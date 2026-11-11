"""
Phase 3: compare post-hoc calibrators beyond temperature scaling.

Question: is the transfer gap (and the KD-vs-scratch difference) specific to
temperature scaling, or does it also show up with richer calibrators fit on
the same source validation set?

Calibrators (all fit on cifar10_val only, 5,000 images, 10 classes):
    ts    temperature scaling, our fixed fit (bounded search on log T), the
          paper's method. Must reproduce Phase 2 exactly.
    vs    vector scaling (Guo et al. 2017): z / t + b, per-class t and b,
          no regularization. probmetrics VectorScalingCalibrator.
    ms    matrix scaling (Guo et al. 2017): W z + b, no regularization.
          probmetrics MatrixScalingCalibrator.
    svs   structured vector scaling (Berta et al., AISTATS 2026), ridge-
          regularized, probmetrics 'svs' with its defaults.
    sms   structured matrix scaling (Berta et al., AISTATS 2026), ridge-
          regularized, probmetrics 'sms' with its defaults (the CalArena
          'logistic' default for multiclass).
    dir   Dirichlet calibration with ODIR regularization (Kull et al. 2019),
          lambda = mu = 1e-3 (probmetrics 'dircal' defaults). The official
          dirichletcal package needs Python >= 3.12 and jax, so this is our
          own re-implementation of its exact objective (FullDirichletCalibrator,
          reg_lambda=1e-3, reg_mu=1e-3, ref_row=True), solved with an exact
          Newton / trust-region method. It was checked against dirichletcal
          0.5.3 on real logits (max probability difference ~1e-6).

Evaluation (same protocol as Phase 2, same A/B split from
results/phase2/split_seed0.npz):
    source:  calibrator fit on cifar10_val, applied to each target domain;
             scored on the full set (absolute calibration) and on halves A
             and B (averaged) for the gaps.
    own oracle (cross-fitted): the SAME calibrator fit on half A of the
             target and scored on half B, and the other way round, averaged.
    gap_own  = source (half-averaged) - own oracle   (per calibrator)
    gap_tsor = source (half-averaged) - cross-fitted TS oracle (Phase 2's
               reference, the same for all calibrators)
Vector/matrix/Dirichlet scaling can change the predicted class, so accuracy
after calibration is reported too.

STL-9: calibrators are fit on 10 classes. A source calibrator is applied to
the unmasked 10-class STL logits and the frog output is then removed (set to
-1e4, i.e. probabilities renormalised over the 9 shared classes). The own
oracle on STL-9 is fit on the 9 shared classes only. For TS both are
identical to Phase 2's masking.

Checks (the script stops if one fails):
  0. inputs: Phase 2 CSV and split file found by md5; split equals a fresh
     seed-0 split; all logit files present; probmetrics version 1.3.0.
  1. TS reproduces Phase 2 for every row (source T, full-set and half-set
     source metrics, cross-fitted oracle metrics).
  2. probmetrics' own temperature scaling agrees with our T on every
     checkpoint (relative difference < 1e-3).
  3. every source fit converged: its NLL on cifar10_val is no worse than the
     model it contains (TS for vs/ms/dir; probmetrics' ts-mix start for
     svs/sms), and Dirichlet's gradient is ~0.
  4. no NaN/inf in any calibrated output.
Own-oracle fits that fail the same convergence test are not stopped on;
they are counted, printed, and left out (NaN) of the own-oracle numbers.

Usage, from the calib-compress folder:
    python3 src/phase3_calibrators.py --quick      (3 checkpoints, a few min)
    python3 src/phase3_calibrators.py              (all 33 checkpoints)
Options: --workers N (parallel processes, default 4), --no-own-oracle.
"""
import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")      # one thread per worker process

import argparse  # noqa: E402
import hashlib  # noqa: E402
import multiprocessing as mp  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
import warnings  # noqa: E402
from concurrent.futures import ProcessPoolExecutor, as_completed  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.optimize import minimize, minimize_scalar  # noqa: E402
from scipy.special import log_softmax, logsumexp, softmax  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from metrics import compute_all_metrics  # noqa: E402  (same metric code as Phases 1-2)

FROG_IDX = 6
N_CLASSES = 10
LOGITS = "logits"
P2_DIR = "results/phase2"
OUT = "results/phase3"
P2_CSV_MD5 = "452ebc2256d5fb41bc672d4b75e1a912"     # calibration_results_phase2.csv (11 Nov 2026)
P2_SPLIT_MD5 = "e820ac2c60e18abb1f846d8d2749ce25"   # split_seed0.npz
CAL = ["ece", "adaptive_ece", "classwise_ece", "nll", "brier"]
CALIBRATORS = ["ts", "vs", "ms", "svs", "sms", "dir"]
ORDER = ["dense", "denseft", "unstr30", "unstr50", "unstr70", "unstr90",
         "str15", "str30", "int8", "kd", "scratch"]
PRUNE = ["unstr30", "unstr50", "unstr70", "unstr90", "str15", "str30", "int8"]
T_MIN, T_MAX = 0.05, 100.0
TOL_REPRO = 1e-6
TOL_ADAPTIVE = 1e-4   # see check 1
DIR_LAMBDA = DIR_MU = 1e-3
TINY64 = np.finfo(np.float64).tiny
QUICK = ("resnet10_kd_s0", "resnet10_scratch_s0", "resnet18_dense_s0")


class FitFailed(Exception):
    pass


# ------------------------------------------------------------------ basics
def nll_at_T(z, y, T):
    s = np.asarray(z, dtype=np.float64) / T
    return float((logsumexp(s, axis=1) - s[np.arange(len(y)), y]).mean())


def fit_T(z, y):
    """Same fixed fit as Phases 1 and 2."""
    lo, hi = np.log(T_MIN), np.log(T_MAX)
    r = minimize_scalar(lambda lt: nll_at_T(z, y, np.exp(lt)), bounds=(lo, hi),
                        method="bounded", options={"xatol": 1e-6, "maxiter": 500})
    if not r.success:
        raise FitFailed(f"T fit did not converge: {r.message}")
    if not (lo + 1e-3 < r.x < hi - 1e-3):
        raise FitFailed(f"T hit the search bound: {np.exp(r.x):.4g}")
    return float(np.exp(r.x))


def nll_of(logits, y):
    ls = log_softmax(np.asarray(logits, dtype=np.float64), axis=1)
    return float(-ls[np.arange(len(y)), y].mean())


def md5(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def stop(msg):
    sys.exit(f"STOP: {msg}\nNothing more was written. Send this output to Claude.")


def load_raw(ckpt, domain):
    """Unmasked logits (float32, as stored) and labels."""
    d = np.load(f"{LOGITS}/{ckpt}__{domain}.npz")
    return d["logits"], d["labels"].astype(np.int64)


def make_split(labels, seed):
    """Identical to Phase 2's class-balanced split."""
    rng = np.random.default_rng(seed)
    a = []
    for c in np.unique(labels):
        idx = np.flatnonzero(labels == c)
        idx = rng.permutation(idx)
        a.append(idx[: len(idx) // 2])
    a = np.sort(np.concatenate(a))
    b = np.setdiff1d(np.arange(len(labels)), a)
    return a, b


# ------------------------------------------------------------- calibrators
def _probs_to_logits(q):
    return np.log(np.clip(q, 1e-300, None))


class Dirichlet:
    """Exact objective of dirichletcal's FullDirichletCalibrator with ODIR
    (reg_lambda=lam on off-diagonal weights, reg_mu=mu on the intercept,
    ref_row=True so the last row is fixed at 0), on log of clipped softmax
    probabilities. Convex; solved with scipy trust-exact (Newton)."""

    def __init__(self, lam=DIR_LAMBDA, mu=DIR_MU):
        self.lam, self.mu = lam, mu

    @staticmethod
    def features(z):
        p = softmax(np.asarray(z, dtype=np.float64), axis=1)
        x = np.log(np.clip(p, TINY64, 1 - TINY64))
        return np.hstack([x, np.ones((len(x), 1))])

    def fit(self, z, y):
        X = self.features(z)
        n, m = X.shape
        k = m - 1
        Y = np.zeros((n, k))
        Y[np.arange(n), y] = 1
        R = np.full((k - 1, m), self.lam)
        R[:, k] = self.mu
        R[np.arange(k - 1), np.arange(k - 1)] = 0.0
        XX = np.einsum("nj,nb->njb", X, X)

        def unpack(v):
            V = v.reshape(k - 1, m)
            return V, log_softmax(X @ np.vstack([V, np.zeros((1, m))]).T, axis=1)

        def f(v):
            V, ls = unpack(v)
            loss = -ls[np.arange(n), y].mean() + (R * V ** 2).sum()
            g = ((np.exp(ls) - Y).T @ X / n)[: k - 1] + 2 * R * V
            return loss, g.ravel()

        def hess(v):
            _, ls = unpack(v)
            P = np.exp(ls)[:, : k - 1]
            A = -np.einsum("ni,na->nia", P, P)
            A[:, np.arange(k - 1), np.arange(k - 1)] += P
            H = np.einsum("nia,njb->ijab", A, XX, optimize=True) / n
            return H.reshape((k - 1) * m, (k - 1) * m) + np.diag(2 * R.ravel())

        V0 = np.hstack([np.eye(k), np.zeros((k, 1))])
        V0 = (V0 - V0[-1])[: k - 1]
        r = minimize(f, V0.ravel(), jac=True, hess=hess, method="trust-exact",
                     options={"gtol": 1e-8, "maxiter": 1000})
        self.grad_max_ = float(np.abs(r.jac).max())
        if not np.isfinite(r.fun) or self.grad_max_ > 1e-6:
            raise FitFailed(f"Dirichlet fit did not converge (max |grad| {self.grad_max_:.1e}, {r.message})")
        self.W_ = np.vstack([r.x.reshape(k - 1, m), np.zeros((1, m))])
        return self

    def logits(self, z):
        return self.features(z) @ self.W_.T


def fit_calibrator(name, z, y):
    """Fit one calibrator on (z, y). Returns (apply_fn, info). apply_fn maps
    logits with the same number of columns to calibrated logits. Raises
    FitFailed if the convergence check fails."""
    z64 = np.asarray(z, dtype=np.float64)
    info = {}
    if name == "ts":
        T = fit_T(z, y)
        info["T"] = T
        return (lambda x: x / T), info

    T = fit_T(z, y)
    ref_nll = nll_at_T(z, y, T)           # TS is nested in vs, ms and dir
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        if name in ("vs", "ms"):
            import torch
            from probmetrics.calibrators import MatrixScalingCalibrator, VectorScalingCalibrator
            from probmetrics.distributions import CategoricalLogits
            c = (VectorScalingCalibrator if name == "vs" else MatrixScalingCalibrator)()
            c.fit_torch(CategoricalLogits(torch.tensor(z64)), torch.as_tensor(y, dtype=torch.long))

            def apply(x, c=c):
                out = c.predict_proba_torch(CategoricalLogits(torch.tensor(np.asarray(x, dtype=np.float64))))
                return out.get_logits().detach().cpu().numpy().astype(np.float64)
        elif name in ("svs", "sms"):
            from probmetrics.calibrators import get_calibrator
            c = get_calibrator(name, **{f"{name}_print_init_info": False})
            c.fit(softmax(z64, axis=1), y)
            # the fit starts from probmetrics' own ts-mix solution (all deltas 0)
            ref_nll = nll_of(_probs_to_logits(c.ts_.predict_proba(softmax(z64, axis=1))), y)

            def apply(x, c=c):
                return _probs_to_logits(c.predict_proba(softmax(np.asarray(x, dtype=np.float64), axis=1)))
        elif name == "dir":
            c = Dirichlet().fit(z64, y)
            info["grad_max"] = c.grad_max_
            apply = c.logits
        else:
            raise ValueError(name)
    msgs = [str(x.message) for x in w if "Not updating" in str(x.message)]
    if msgs:
        raise FitFailed(f"{name}: {msgs[0][:120]}")
    fit_nll = nll_of(apply(z), y)
    info["fit_nll"], info["ref_nll"] = fit_nll, ref_nll
    tol = 1e-4 if name in ("svs", "sms") else 1e-6   # svs/sms work in float32 internally
    if not np.isfinite(fit_nll) or fit_nll > ref_nll + tol:
        raise FitFailed(f"{name}: fit NLL {fit_nll:.6f} is worse than its starting model {ref_nll:.6f}")
    return apply, info


def drop_frog(z):
    return np.delete(z, FROG_IDX, axis=1)


def add_frog(z9):
    z9 = np.asarray(z9)
    return np.insert(z9, FROG_IDX, -1e4, axis=1)


def remap9(y):
    return np.where(y > FROG_IDX, y - 1, y)


def cal_metrics(z, y):
    if not np.all(np.isfinite(z[:, np.arange(z.shape[1]) != FROG_IDX])):
        raise FloatingPointError("non-finite calibrated logits")
    m = compute_all_metrics(z, y)
    return {k: float(m[k]) for k in ["accuracy"] + CAL}


# ----------------------------------------------------------- one checkpoint
def run_checkpoint(ckpt, domains, split, own_oracle):
    """All calibrators x all domains for one checkpoint. Returns
    (rows, source_fit_rows, failure_rows)."""
    warnings.filterwarnings("ignore")
    try:
        import torch
        torch.set_num_threads(1)
    except ImportError:
        pass
    zv, yv = load_raw(ckpt, "cifar10_val")
    src = {}
    fits = []
    for name in CALIBRATORS:
        try:
            apply, info = fit_calibrator(name, zv, yv)
        except FitFailed as e:
            return None, None, [{"checkpoint": ckpt, "where": "SOURCE", "calibrator": name, "error": str(e)}]
        src[name] = apply
        vm = cal_metrics(apply(zv), yv)
        fits.append({"checkpoint": ckpt, "calibrator": name, **info,
                     "val_accuracy": vm["accuracy"], "val_ece": vm["ece"], "val_nll": vm["nll"]})
    # check 2 input: probmetrics' own TS on the source set
    from probmetrics.calibrators import get_calibrator
    pts = get_calibrator("temp-scaling")
    pts.fit(softmax(np.asarray(zv, dtype=np.float64), axis=1), yv)
    inv_t = getattr(pts, "invtemp_", None)
    fits[0]["probmetrics_T"] = float(1.0 / inv_t) if inv_t is not None else np.nan

    rows, fails = [], []
    for d in domains:
        z, y = load_raw(ckpt, d)
        stl = d == "stl9"
        if stl:
            assert not (y == FROG_IDX).any()
        a, b = split["stl9" if stl else "cifar10"]
        raw = z.copy()
        if stl:
            raw[:, FROG_IDX] = -1e4
        raw_full = cal_metrics(raw, y)
        raw_h = [cal_metrics(raw[a], y[a]), cal_metrics(raw[b], y[b])]
        for name in CALIBRATORS:
            # source calibrator on the whole domain, then frog removed for STL
            if name == "ts":
                zc = src[name](raw)                    # Phase 2: mask, then divide by T
            else:
                zc = src[name](z)
                if stl:
                    zc = zc.copy()
                    zc[:, FROG_IDX] = -1e4
            full = cal_metrics(zc, y)
            sh = [cal_metrics(zc[a], y[a]), cal_metrics(zc[b], y[b])]
            row = {"checkpoint": ckpt, "domain": d, "calibrator": name, "n_samples": len(y),
                   "accuracy_raw": raw_full["accuracy"], "accuracy_src_full": full["accuracy"],
                   "accuracy_src_half": (sh[0]["accuracy"] + sh[1]["accuracy"]) / 2}
            for m in CAL:
                row[f"raw_full_{m}"] = raw_full[m]
                row[f"raw_half_{m}"] = (raw_h[0][m] + raw_h[1][m]) / 2
                row[f"src_full_{m}"] = full[m]
                row[f"src_half_{m}"] = (sh[0][m] + sh[1][m]) / 2
            # own cross-fitted oracle
            ok = True
            if own_oracle:
                xh = []
                for fit_idx, ev_idx, h in [(b, a, "A"), (a, b, "B")]:   # score A with fit on B, etc.
                    try:
                        if name == "ts":
                            fz = raw[fit_idx]
                            f_apply, _ = fit_calibrator(name, fz, y[fit_idx])
                            zo = f_apply(raw[ev_idx])
                        elif stl:
                            f_apply, _ = fit_calibrator(name, drop_frog(z[fit_idx]), remap9(y[fit_idx]))
                            zo = add_frog(f_apply(drop_frog(z[ev_idx])))
                        else:
                            f_apply, _ = fit_calibrator(name, z[fit_idx], y[fit_idx])
                            zo = f_apply(z[ev_idx])
                        xh.append(cal_metrics(zo, y[ev_idx]))
                    except (FitFailed, FloatingPointError) as e:
                        ok = False
                        fails.append({"checkpoint": ckpt, "domain": d, "calibrator": name,
                                      "where": f"oracle fit for half {h}", "error": str(e)[:200]})
                row["own_oracle_ok"] = ok
                row["accuracy_xfit"] = (xh[0]["accuracy"] + xh[1]["accuracy"]) / 2 if ok else np.nan
                for m in CAL:
                    row[f"xfit_{m}"] = (xh[0][m] + xh[1][m]) / 2 if ok else np.nan
                    row[f"gap_own_{m}"] = row[f"src_half_{m}"] - row[f"xfit_{m}"]
            rows.append(row)
    return rows, fits, fails


# --------------------------------------------------------------- summaries
def add_meta(df, p2):
    meta = p2[["checkpoint", "arch", "method", "seed"]].drop_duplicates()
    df = df.merge(meta, on="checkpoint", how="left")
    df["sev"] = np.where(df.domain.str.startswith("cifar10c_"), df.domain.str[-1], "")
    df["corr"] = df.domain.str.extract(r"cifar10c_(.+)_sev\d", expand=False)
    return df


def summarise(res, have_oracle):
    """Long table of headline numbers: (setting, quantity, calibrator, arm,
    statistic) -> value."""
    out = []

    def put(setting, qty, cal, arm, stat, val):
        out.append({"setting": setting, "quantity": qty, "calibrator": cal, "arm": arm,
                    "stat": stat, "value": val})

    subsets = {"c10c": res[res.domain.str.startswith("cifar10c_")],
               "stl9": res[res.domain == "stl9"],
               "c10test": res[res.domain == "cifar10_test"]}
    qtys = {"abs_ece": "src_full_ece", "abs_nll": "src_full_nll", "accuracy": "accuracy_src_full",
            "gap_tsor_ece": "gap_tsor_ece"}
    if have_oracle:
        qtys.update({"gap_own_ece": "gap_own_ece", "own_oracle_ece": "xfit_ece"})
    for setting, d in subsets.items():
        for qn, col in qtys.items():
            ps = d.groupby(["calibrator", "method", "seed"])[col].mean()
            for (cal, arm), s in ps.groupby(level=[0, 1]):
                s = s.droplevel([0, 1])
                put(setting, qn, cal, arm, "mean", s.mean())
                put(setting, qn, cal, arm, "sd", s.std())
                for seed, v in s.items():
                    put(setting, qn, cal, arm, f"seed{seed}", v)
            if setting == "c10test":
                continue
            for cal in CALIBRATORS:
                if cal not in ps.index.get_level_values(0):
                    continue
                pc = ps.loc[cal]
                arms = set(pc.index.get_level_values(0))
                if {"kd", "scratch"} <= arms:
                    diff = pc.loc["kd"] - pc.loc["scratch"]
                    for seed, v in diff.items():
                        put(setting, qn, cal, "kd_minus_scratch", f"seed{seed}", v)
                    put(setting, qn, cal, "kd_minus_scratch", "seeds_positive_of_3", int((diff > 0).sum()))
                    put(setting, qn, cal, "kd_over_scratch", "ratio", pc.loc["kd"].mean() / pc.loc["scratch"].mean())
                if {"kd", "dense"} <= arms:
                    for seed, v in (pc.loc["kd"] - pc.loc["dense"]).items():
                        put(setting, qn, cal, "kd_minus_dense", f"seed{seed}", v)
                for p in PRUNE:
                    if {p, "denseft"} <= arms:
                        dd = pc.loc[p] - pc.loc["denseft"]
                        put(setting, qn, cal, f"{p}_minus_denseft", "seeds_positive_of_3", int((dd > 0).sum()))
                        put(setting, qn, cal, f"{p}_minus_denseft", "mean", dd.mean())
        # recovered fraction of the TS-oracle improvement, by calibrator and arm
        if setting != "c10test":
            g = d.groupby(["calibrator", "method"])[["raw_half_ece", "src_half_ece", "xfit_ts_ece"]].mean()
            for (cal, arm), r in g.iterrows():
                put(setting, "recovered_frac_of_ts_oracle", cal, arm, "all_seeds",
                    (r.raw_half_ece - r.src_half_ece) / (r.raw_half_ece - r.xfit_ts_ece))
        # other metrics: KD minus scratch, seeds positive
        if setting != "c10test":
            for cal in CALIBRATORS:
                dc = d[d.calibrator == cal]
                for m in CAL[1:]:          # ECE itself is covered above
                    for qn, col in [("abs", f"src_full_{m}"), ("gap_tsor", f"gap_tsor_{m}")] + \
                                   ([("gap_own", f"gap_own_{m}")] if have_oracle else []):
                        pm = dc.groupby(["method", "seed"])[col].mean().unstack("seed")
                        if {"kd", "scratch"} <= set(pm.index):
                            put(setting, f"metric_{qn}_{m}", cal, "kd_minus_scratch", "seeds_positive_of_3",
                                int((pm.loc["kd"] - pm.loc["scratch"] > 0).sum()))
    c10c = subsets["c10c"]
    if len(c10c):
        for cal in CALIBRATORS:
            dc = c10c[c10c.calibrator == cal]
            if not {"kd", "scratch"} <= set(dc.method):
                continue
            for qn, col in [("abs_ece", "src_full_ece"), ("gap_tsor_ece", "gap_tsor_ece")] + \
                           ([("gap_own_ece", "gap_own_ece")] if have_oracle else []):
                sv = dc.groupby(["method", "sev"])[col].mean().unstack()
                for s in sv.columns:
                    put("c10c", qn, cal, "kd_minus_scratch", f"sev{s}", sv.loc["kd", s] - sv.loc["scratch", s])
                cc = dc.groupby(["method", "corr"])[col].mean().unstack()
                cc = cc.loc["kd"] - cc.loc["scratch"]
                put("c10c", qn, cal, "kd_minus_scratch", "corruptions_positive_of_15", int((cc > 0).sum()))
    return pd.DataFrame(out)


def pivot(summ, setting, qty, stat="mean", arms=ORDER):
    s = summ[(summ.setting == setting) & (summ.quantity == qty) & (summ.stat == stat)]
    t = s.pivot_table(index="arm", columns="calibrator", values="value")
    t = t.reindex([a for a in arms if a in t.index])
    return t[[c for c in CALIBRATORS if c in t.columns]]


# -------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--no-own-oracle", action="store_true")
    args = ap.parse_args()
    own_oracle = not args.no_own_oracle
    t0 = time.time()

    # ---- Step 0: inputs
    print("== Step 0: inputs ==")
    if not os.path.isdir(LOGITS):
        sys.exit("Run this from the calib-compress folder (need logits/ and results/).")
    try:
        from importlib.metadata import version
        import probmetrics.calibrators  # noqa: F401  (also imports numba, torch, torchmin)
        import torchmin  # noqa: F401
        pv = version("probmetrics")
    except Exception as e:  # noqa: BLE001
        stop(f"probmetrics could not be imported ({type(e).__name__}: {e}). Did Step 4 (pip install) finish?")
    if pv != "1.3.0":
        stop(f"probmetrics version is {pv}, expected 1.3.0")
    import torch
    print(f"probmetrics {pv}, torch {torch.__version__}, numpy {np.__version__}")
    p2_csv = f"{P2_DIR}/calibration_results_phase2.csv"
    p2_split = f"{P2_DIR}/split_seed0.npz"
    for path, want in [(p2_csv, P2_CSV_MD5), (p2_split, P2_SPLIT_MD5)]:
        if not os.path.isfile(path):
            stop(f"{path} not found (it is written by the Phase 2 run)")
        if md5(path) != want:
            stop(f"{path} has md5 {md5(path)}, expected {want}")
    p2 = pd.read_csv(p2_csv)
    print(f"Phase 2 CSV: {p2_csv} ({len(p2)} rows, md5 {P2_CSV_MD5[:8]}..., matches)")
    ckpts = sorted(p2.checkpoint.unique())
    if args.quick:
        ckpts = [c for c in ckpts if c in QUICK]
        print(f"(--quick: only {ckpts})")
    domains = sorted(p2.domain.unique())
    missing = [f"{c}__{d}" for c in ckpts for d in ["cifar10_val"] + domains
               if not os.path.exists(f"{LOGITS}/{c}__{d}.npz")]
    if missing:
        stop(f"{len(missing)} logit files are missing, e.g. {missing[:3]}")
    print(f"logit files: all {len(ckpts)} checkpoints x {len(domains) + 1} domains present")
    sp = np.load(p2_split)
    split = {"cifar10": (sp["cifar10_A"], sp["cifar10_B"]), "stl9": (sp["stl9_A"], sp["stl9_B"])}
    y_c10 = load_raw(ckpts[0], "cifar10_test")[1]
    y_stl = load_raw(ckpts[0], "stl9")[1]
    for key, y in [("cifar10", y_c10), ("stl9", y_stl)]:
        a2, b2 = make_split(y, 0)
        if not (np.array_equal(a2, split[key][0]) and np.array_equal(b2, split[key][1])):
            stop(f"the saved {key} split does not equal a fresh seed-0 split")
    print(f"A/B split: {p2_split} (md5 {P2_SPLIT_MD5[:8]}..., equals a fresh seed-0 split; "
          f"CIFAR-10 {len(split['cifar10'][0])}/{len(split['cifar10'][1])}, "
          f"STL-9 {len(split['stl9'][0])}/{len(split['stl9'][1])})")
    os.makedirs(OUT, exist_ok=True)

    # ---- Step 1: fit and evaluate
    print(f"\n== Step 1: fit {len(CALIBRATORS)} calibrators on cifar10_val, evaluate on {len(domains)} domains "
          f"({'with' if own_oracle else 'without'} own cross-fitted oracle), {args.workers} workers ==",
          flush=True)
    rows, fits, fails = [], [], []
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=mp.get_context("spawn")) as ex:
        futs = {ex.submit(run_checkpoint, c, domains, split, own_oracle): c for c in ckpts}
        for k, f in enumerate(as_completed(futs), 1):
            r, fi, fa = f.result()
            if r is None:
                stop(f"source fit failed: {fa[0]}")
            rows += r
            fits += fi
            fails += fa
            print(f"  {k}/{len(ckpts)} {futs[f]} done  ({(time.time() - t0) / 60:.1f} min)", flush=True)
    res = pd.DataFrame(rows)
    fits = pd.DataFrame(fits)
    fails = pd.DataFrame(fails)

    # ---- Step 2: checks
    print("\n== Step 2: checks ==")
    p2i = p2.set_index(["checkpoint", "domain"])
    ts = res[res.calibrator == "ts"].set_index(["checkpoint", "domain"])
    worst = 0.0
    where = None
    pairs = [("accuracy_src_full", "accuracy")]
    for m in CAL:
        pairs += [(f"src_full_{m}", f"src_full_{m}"), (f"src_half_{m}", f"src_half_{m}"),
                  (f"raw_half_{m}", f"raw_half_{m}"), (f"raw_full_{m}", f"raw_full_{m}")]
        if own_oracle:
            pairs += [(f"xfit_{m}", f"xfit_{m}"), (f"gap_own_{m}", f"gap_{m}")]
    ref = p2i.loc[ts.index]
    worst_a, where_a = 0.0, None
    for mine, theirs in pairs:
        dd = (ts[mine] - ref[theirs]).abs() / np.maximum(1.0, ref[theirs].abs())
        if "adaptive" in mine:
            # adaptive ECE sorts confidences, and many are tied at 1.0 in float32, so its
            # value depends on how the sort breaks ties (can differ across machines)
            if dd.max() > worst_a:
                worst_a, where_a = dd.max(), (dd.idxmax(), mine)
        elif dd.max() > worst:
            worst, where = dd.max(), (dd.idxmax(), mine)
    tsrc = fits[fits.calibrator == "ts"].set_index("checkpoint")
    p2T = p2.groupby("checkpoint").source_T.first()
    dT = ((tsrc["T"] - p2T.loc[tsrc.index]).abs() / p2T.loc[tsrc.index]).max()
    print(f"check 1, TS reproduces Phase 2: worst metric diff {worst:.1e} {where if where else ''}")
    print(f"         adaptive ECE worst diff {worst_a:.1e} {where_a if where_a else ''}; "
          f"worst source-T rel diff {dT:.1e}")
    if worst > TOL_REPRO or worst_a > TOL_ADAPTIVE or dT > 1e-9:
        stop("temperature scaling does not reproduce the Phase 2 numbers")
    print("CHECK 1 PASSED: TS numbers reproduce Phase 2.")
    dpm = ((tsrc["probmetrics_T"] - tsrc["T"]).abs() / tsrc["T"]).max()
    print(f"check 2, probmetrics temperature scaling vs ours on cifar10_val: worst rel diff in T {dpm:.1e}")
    if not dpm < 1e-3:
        stop("probmetrics' temperature scaling disagrees with ours by more than 0.1%")
    print("CHECK 2 PASSED.")
    print("check 3, every source fit converged (fit NLL <= its starting model): yes, for all "
          f"{len(fits)} fits (the script stops otherwise)")
    if (fits.calibrator == "dir").any():
        print(f"         Dirichlet: worst |gradient| at the solution {fits.grad_max.max():.1e}")
    print("check 4, no NaN/inf in calibrated outputs: yes (the script stops otherwise)")
    n_fail = len(fails)
    if own_oracle:
        print(f"own-oracle fits that failed the convergence test (left out as NaN): {n_fail}")
        if n_fail:
            print(fails.groupby("calibrator").size().to_string())

    # gap vs the cross-fitted TS oracle (Phase 2 reference)
    xts = ts[[f"xfit_{m}" for m in CAL]] if own_oracle else \
        ref[[f"xfit_{m}" for m in CAL]]
    xts = xts.rename(columns={f"xfit_{m}": f"xfit_ts_{m}" for m in CAL}).reset_index()
    res = res.merge(xts, on=["checkpoint", "domain"], how="left")
    for m in CAL:
        res[f"gap_tsor_{m}"] = res[f"src_half_{m}"] - res[f"xfit_ts_{m}"]
    res = add_meta(res, p2)
    lead = ["checkpoint", "arch", "method", "seed", "domain", "calibrator"]
    res = res[lead + [c for c in res.columns if c not in lead]]
    res.drop(columns=["sev", "corr"]).to_csv(f"{OUT}/calibration_results_phase3.csv", index=False)
    fits.to_csv(f"{OUT}/source_fits.csv", index=False)
    if n_fail:
        fails.to_csv(f"{OUT}/oracle_fit_failures.csv", index=False)
    print(f"wrote {OUT}/calibration_results_phase3.csv ({len(res)} rows = {res.checkpoint.nunique()} checkpoints x "
          f"{res.domain.nunique()} domains x {res.calibrator.nunique()} calibrators), source_fits.csv")

    # ---- Step 3: summary
    print("\n== Step 3: summary (mean over 3 seeds; full tables in results/phase3/summary_phase3.csv) ==")
    summ = summarise(res, own_oracle)
    summ.to_csv(f"{OUT}/summary_phase3.csv", index=False)
    pd.set_option("display.width", 200)
    pd.set_option("display.max_rows", 500)
    ff = lambda v: f"{v:.4f}"  # noqa: E731
    print("\nIn-domain sanity (CIFAR-10 test), ECE after source-fit calibration:")
    print(pivot(summ, "c10test", "abs_ece").to_string(float_format=ff))
    for setting, label in [("c10c", "CIFAR-10-C (mean of 75 conditions)"), ("stl9", "STL-9")]:
        print(f"\n--- {label} ---")
        print("Accuracy after source-fit calibration (raw accuracy for ts):")
        print(pivot(summ, setting, "accuracy").to_string(float_format=ff))
        print("ECE after source-fit calibration (absolute, full set):")
        print(pivot(summ, setting, "abs_ece").to_string(float_format=ff))
        print("Gap to the cross-fitted TS oracle (source half ECE - TS oracle ECE):")
        print(pivot(summ, setting, "gap_tsor_ece").to_string(float_format=ff))
        if own_oracle:
            print("Gap to each calibrator's own cross-fitted oracle:")
            print(pivot(summ, setting, "gap_own_ece").to_string(float_format=ff))
        print("Fraction of the TS-oracle improvement recovered:")
        print(pivot(summ, setting, "recovered_frac_of_ts_oracle", stat="all_seeds").to_string(float_format=ff))
        print("KD minus scratch, per seed (s0 / s1 / s2), and seeds positive:")
        qs = ["abs_ece", "gap_tsor_ece"] + (["gap_own_ece"] if own_oracle else [])
        for q in qs:
            s = summ[(summ.setting == setting) & (summ.quantity == q) & (summ.arm == "kd_minus_scratch")]
            for cal in CALIBRATORS:
                v = s[s.calibrator == cal].set_index("stat").value
                if len(v) == 0:
                    continue
                seeds = " / ".join(f"{v[k]:+.4f}" for k in sorted(v.index) if k.startswith("seed") and len(k) == 5)
                print(f"  {q:13s} {cal:4s}: {seeds}   positive {int(v.get('seeds_positive_of_3', -1))}/3")
    print(f"\nALL DONE in {(time.time() - t0) / 60:.1f} min.")


if __name__ == "__main__":
    main()

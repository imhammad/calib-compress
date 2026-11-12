"""
Dirichlet calibration with cross-validated ODIR regularization (Paolo's
request, 12 Nov 2026).

Phase 3 used fixed lambda = mu = 1e-3 (the probmetrics 'dircal' defaults).
Here, for every checkpoint, lambda (off-diagonal weights) and mu (intercept)
are chosen by 5-fold stratified cross-validation ON THE SOURCE VALIDATION
SET ONLY (cifar10_val, 5,000 images), by mean held-out NLL, over the grid
{1e-5, 1e-4, 1e-3, 1e-2, 1e-1} x {1e-5, ..., 1e-1} (25 pairs; the grid of
probmetrics' 'dircal-cv' and in the spirit of Kull et al. 2019). The
calibrator is then refit on all of cifar10_val with the chosen pair and
evaluated exactly like the other calibrators in pooled_crossfit.py:

    source:      full target set (absolute numbers)
    own oracle:  pooled out-of-fold predictions for 5 A/B splits, averaged.
                 The oracle is a supplementary diagnostic. It refits the
                 Dirichlet weights on target halves but keeps the lambda, mu
                 chosen on the source, so no target data is used to choose
                 hyperparameters.
    gap_own  = source - own pooled oracle
    gap_tsor = source - pooled TS oracle (taken from results/pooled)

No target data is touched by the cross-validation.

Checks (the script stops if one fails):
  0. inputs: pooled CSV found by md5; logit files present.
  1. with lambda = mu = 1e-3 the source fit reproduces the Phase 3 / pooled
     'dir' numbers on the clean test set (same code path).
  2. every fit converged (convex objective, |gradient| ~ 0 at the solution).
  3. pooled oracle covers every image once.
Also reported: how often the chosen pair sits on the edge of the grid.

Usage, from the calib-compress folder:
    python3 src/dirichlet_cv.py --quick
    python3 src/dirichlet_cv.py
"""
import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse  # noqa: E402
import multiprocessing as mp  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
import warnings  # noqa: E402
from concurrent.futures import ProcessPoolExecutor, as_completed  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import phase3_calibrators as p3  # noqa: E402
from phase3_calibrators import (CAL, FROG_IDX, Dirichlet, FitFailed, add_frog, cal_metrics,  # noqa: E402
                                drop_frog, load_raw, make_split, md5, nll_of,
                                remap9, stop)

OUT = "results/dirichlet_cv"
POOLED_CSV = "results/pooled/calibration_results_pooled.csv"
POOLED_MD5 = "c22671212296ddddcd28229ddd1196a3"     # Hammad's pooled run, 11 Nov 2026
GRID = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1]
N_FOLDS = 5
N_SPLITS = 5
QUICK = ("resnet10_kd_s0", "resnet10_scratch_s0", "resnet18_dense_s0")


def fit_dir(z, y, lam, mu):
    """Dirichlet-ODIR fit. The objective is convex, so a gradient of ~0 (checked
    inside Dirichlet.fit, which raises FitFailed otherwise) means the optimum
    was found. (No comparison with temperature scaling here: with ref_row=True
    the ODIR penalty is not zero at the TS solution, so for large lambda the
    fitted NLL can legitimately be above the TS NLL.)"""
    c = Dirichlet(lam, mu).fit(z, y)
    if not np.isfinite(nll_of(c.logits(z), y)):
        raise FitFailed("Dirichlet fit gave a non-finite NLL")
    return c


def stratified_folds(y, n_folds, seed=0):
    rng = np.random.default_rng(seed)
    fold = np.empty(len(y), dtype=int)
    for c in np.unique(y):
        idx = rng.permutation(np.flatnonzero(y == c))
        fold[idx] = np.arange(len(idx)) % n_folds
    return fold


def run_checkpoint(ckpt, domains, splits):
    warnings.filterwarnings("ignore")
    zv, yv = load_raw(ckpt, "cifar10_val")
    zv = np.asarray(zv, dtype=np.float64)
    # ---- cross-validation on the source validation set only
    fold = stratified_folds(yv, N_FOLDS)
    cv = []
    for lam in GRID:
        for mu in GRID:
            nlls = []
            for k in range(N_FOLDS):
                tr, te = fold != k, fold == k
                c = fit_dir(zv[tr], yv[tr], lam, mu)
                nlls.append(nll_of(c.logits(zv[te]), yv[te]))
            cv.append({"checkpoint": ckpt, "lambda": lam, "mu": mu,
                       "cv_nll": float(np.mean(nlls)), "cv_nll_sd": float(np.std(nlls, ddof=1))})
    best = min(cv, key=lambda r: r["cv_nll"])
    lam, mu = best["lambda"], best["mu"]
    src = fit_dir(zv, yv, lam, mu)
    default = fit_dir(zv, yv, 1e-3, 1e-3)
    vm = cal_metrics(src.logits(zv), yv)
    sel = {"checkpoint": ckpt, "lambda": lam, "mu": mu, "cv_nll": best["cv_nll"],
           "cv_nll_default_1e-3": [r["cv_nll"] for r in cv if r["lambda"] == 1e-3 and r["mu"] == 1e-3][0],
           "val_ece_in_sample": vm["ece"], "val_nll_in_sample": vm["nll"]}

    rows = []
    for d in domains:
        z, y = load_raw(ckpt, d)
        stl = d == "stl9"

        def masked(c, x):
            out = c.logits(x).copy()
            if stl:
                out[:, FROG_IDX] = -1e4
            return out

        full = cal_metrics(masked(src, z), y)
        row = {"checkpoint": ckpt, "domain": d, "calibrator": "dir_cv", "n_samples": len(y),
               "lambda": lam, "mu": mu, "accuracy_src_full": full["accuracy"]}
        for m in CAL:
            row[f"src_full_{m}"] = full[m]
        if d == "cifar10_test":
            row["default_src_full_ece_check"] = cal_metrics(masked(default, z), y)["ece"]
        per_split = []
        for sp in splits:
            a, b = sp["stl9" if stl else "cifar10"]
            zo = np.full((len(y), z.shape[1]), np.nan)
            filled = np.zeros(len(y), dtype=int)
            for fit_idx, ev_idx in [(b, a), (a, b)]:
                if stl:
                    c = fit_dir(drop_frog(z[fit_idx]), remap9(y[fit_idx]), lam, mu)
                    zo[ev_idx] = add_frog(c.logits(drop_frog(z[ev_idx])))
                else:
                    c = fit_dir(z[fit_idx], y[fit_idx], lam, mu)
                    zo[ev_idx] = c.logits(z[ev_idx])
                filled[ev_idx] += 1
            if not (filled == 1).all():
                raise RuntimeError("pooled oracle does not cover each image once")
            per_split.append(cal_metrics(zo, y))
        for m in CAL + ["accuracy"]:
            vals = [x[m] for x in per_split]
            row[f"xfit_pooled_{m}"] = float(np.mean(vals))
            for s, v in enumerate(vals):
                row[f"xfit_pooled_{m}_split{s}"] = v
        for m in CAL:
            row[f"gap_own_{m}"] = row[f"src_full_{m}"] - row[f"xfit_pooled_{m}"]
        rows.append(row)
    return rows, cv, sel


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()
    t0 = time.time()

    print("== Step 0: inputs ==")
    if not os.path.isdir(p3.LOGITS):
        sys.exit("Run this from the calib-compress folder (need logits/ and results/).")
    if not os.path.isfile(POOLED_CSV):
        stop(f"{POOLED_CSV} not found")
    if md5(POOLED_CSV) != POOLED_MD5:
        stop(f"{POOLED_CSV} has md5 {md5(POOLED_CSV)}, expected {POOLED_MD5}")
    pooled = pd.read_csv(POOLED_CSV)
    print(f"pooled CSV: {POOLED_CSV} ({len(pooled)} rows, md5 {POOLED_MD5[:8]}..., matches)")
    ckpts = sorted(pooled.checkpoint.unique())
    if args.quick:
        ckpts = [c for c in ckpts if c in QUICK]
        print(f"(--quick: only {ckpts})")
    domains = sorted(pooled.domain.unique())
    missing = [f"{c}__{d}" for c in ckpts for d in ["cifar10_val"] + domains
               if not os.path.exists(f"{p3.LOGITS}/{c}__{d}.npz")]
    if missing:
        stop(f"{len(missing)} logit files are missing, e.g. {missing[:3]}")
    y_c10 = load_raw(ckpts[0], "cifar10_test")[1]
    y_stl = load_raw(ckpts[0], "stl9")[1]
    splits = [{"cifar10": make_split(y_c10, s), "stl9": make_split(y_stl, s)} for s in range(N_SPLITS)]
    print(f"{len(ckpts)} checkpoints; grid {len(GRID)}x{len(GRID)} (lambda, mu in {GRID}), "
          f"{N_FOLDS}-fold CV on cifar10_val only; {N_SPLITS} A/B splits for the pooled oracle")
    os.makedirs(OUT, exist_ok=True)

    print(f"\n== Step 1: cross-validate, refit, evaluate ({args.workers} workers) ==", flush=True)
    rows, cvs, sels = [], [], []
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=mp.get_context("spawn")) as ex:
        futs = {ex.submit(run_checkpoint, c, domains, splits): c for c in ckpts}
        for k, f in enumerate(as_completed(futs), 1):
            try:
                r, cv, sel = f.result()
            except Exception as e:  # noqa: BLE001
                stop(f"{futs[f]}: {type(e).__name__}: {e}")
            rows += r
            cvs += cv
            sels.append(sel)
            print(f"  {k}/{len(ckpts)} {futs[f]}: chose lambda {sel['lambda']:g}, mu {sel['mu']:g}  "
                  f"({(time.time() - t0) / 60:.1f} min)", flush=True)
    res = pd.DataFrame(rows)
    sel = pd.DataFrame(sels).sort_values("checkpoint")
    pd.DataFrame(cvs).to_csv(f"{OUT}/cv_grid.csv", index=False)
    sel.to_csv(f"{OUT}/selected_hyperparameters.csv", index=False)

    print("\n== Step 2: checks ==")
    pd_dir = pooled[pooled.calibrator == "dir"].set_index(["checkpoint", "domain"])
    chk = res[res.domain == "cifar10_test"].set_index(["checkpoint", "domain"])
    d1 = (chk["default_src_full_ece_check"] - pd_dir.loc[chk.index, "src_full_ece"]).abs().max()
    print(f"check 1, lambda = mu = 1e-3 reproduces the pooled 'dir' ECE on the clean test set: worst diff {d1:.1e}")
    if d1 > 1e-6:
        stop("the default Dirichlet fit does not reproduce the earlier run")
    print("CHECK 1 PASSED.")
    print("check 2, every fit converged: yes (the script stops otherwise)")
    print("check 3, pooled oracle covers every image once: yes (the script stops otherwise)")
    edge = ((sel["lambda"].isin([GRID[0], GRID[-1]])) | (sel["mu"].isin([GRID[0], GRID[-1]]))).sum()
    print(f"chosen (lambda, mu) pairs, count over {len(sel)} checkpoints:")
    print(sel.groupby(["lambda", "mu"]).size().to_string())
    print(f"checkpoints whose choice sits on the edge of the grid: {edge} of {len(sel)}")
    print(f"mean CV NLL: chosen {sel.cv_nll.mean():.5f} vs default (1e-3, 1e-3) {sel['cv_nll_default_1e-3'].mean():.5f}")

    # ---- combine with ts and default dir from the pooled run
    res = res.drop(columns=["default_src_full_ece_check"])
    ts = pooled[pooled.calibrator == "ts"].set_index(["checkpoint", "domain"])
    idx = res.set_index(["checkpoint", "domain"]).index
    for m in CAL:
        res[f"xfit_ts_{m}"] = ts.loc[idx, f"xfit_pooled_{m}"].values
        res[f"gap_tsor_{m}"] = res[f"src_full_{m}"] - res[f"xfit_ts_{m}"]
        res[f"raw_full_{m}"] = ts.loc[idx, f"raw_full_{m}"].values
    meta = pooled[["checkpoint", "arch", "method", "seed"]].drop_duplicates()
    res = res.merge(meta, on="checkpoint", how="left")
    lead = ["checkpoint", "arch", "method", "seed", "domain", "calibrator"]
    res = res[lead + [c for c in res.columns if c not in lead]]
    res.to_csv(f"{OUT}/calibration_results_dirichlet_cv.csv", index=False)
    print(f"wrote {OUT}/calibration_results_dirichlet_cv.csv ({len(res)} rows), cv_grid.csv, selected_hyperparameters.csv")

    print("\n== Step 3: summary (mean over seeds) ==")
    keep = ["checkpoint", "method", "seed", "domain", "calibrator", "accuracy_src_full",
            "src_full_ece", "src_full_nll", "gap_own_ece", "gap_tsor_ece", "xfit_pooled_ece"]
    old = pooled[pooled.calibrator.isin(["ts", "dir"]) & pooled.checkpoint.isin(ckpts)].copy()
    old["gap_tsor_ece"] = old["src_full_ece"] - ts.loc[old.set_index(["checkpoint", "domain"]).index, "xfit_pooled_ece"].values
    allr = pd.concat([old[keep], res[keep]], ignore_index=True)
    pd.set_option("display.width", 200)
    ff = lambda v: f"{v:.4f}"  # noqa: E731
    order = [a for a in p3.ORDER if a in set(allr.method)]
    out = []
    for setting, selr in [("clean CIFAR-10 test", allr.domain == "cifar10_test"),
                          ("CIFAR-10-C (mean of 75 conditions)", allr.domain.str.startswith("cifar10c_")),
                          ("STL-9", allr.domain == "stl9")]:
        d = allr[selr]
        print(f"\n--- {setting} ---")
        for q, label in [("src_full_ece", "ECE after source-fit calibration (absolute)"),
                         ("src_full_nll", "NLL after source-fit calibration"),
                         ("accuracy_src_full", "Accuracy after source-fit calibration"),
                         ("gap_own_ece", "Gap to own pooled oracle (supplementary diagnostic)")]:
            ps = d.groupby(["calibrator", "method", "seed"])[q].mean()
            t = ps.groupby(level=[0, 1]).mean().unstack(0).reindex(order)[["ts", "dir", "dir_cv"]]
            print(label + ":")
            print(t.to_string(float_format=ff))
            for cal in ["ts", "dir", "dir_cv"]:
                for arm in order:
                    for seed, v in ps[cal][arm].items():
                        out.append({"setting": setting, "quantity": q, "calibrator": cal, "arm": arm,
                                    "seed": seed, "value": v})
        if setting.startswith("clean"):
            continue
        pc = d.groupby(["calibrator", "checkpoint"]).src_full_ece.mean().unstack(0)
        print(f"dir_cv minus ts, absolute ECE: mean {(pc.dir_cv - pc.ts).mean():+.4f}, "
              f"lower than ts on {int((pc.dir_cv < pc.ts).sum())} of {len(pc)} checkpoints; "
              f"dir_cv minus default dir: mean {(pc.dir_cv - pc['dir']).mean():+.4f}, "
              f"lower on {int((pc.dir_cv < pc['dir']).sum())} of {len(pc)}")
        if {"kd", "scratch"} <= set(d.method):
            print("KD minus scratch, per seed (s0 / s1 / s2):")
            for q in ["src_full_ece", "gap_tsor_ece", "gap_own_ece"]:
                ps = d.groupby(["calibrator", "method", "seed"])[q].mean()
                for cal in ["ts", "dir", "dir_cv"]:
                    dd = ps[cal]["kd"] - ps[cal]["scratch"]
                    print(f"  {q:13s} {cal:6s}: " + " / ".join(f"{v:+.4f}" for v in dd)
                          + f"   positive {int((dd > 0).sum())}/{len(dd)}"
                          + f"   ratio {ps[cal]['kd'].mean() / ps[cal]['scratch'].mean():.3f}")
    pd.DataFrame(out).to_csv(f"{OUT}/summary_dirichlet_cv.csv", index=False)
    print(f"\nALL DONE in {(time.time() - t0) / 60:.1f} min.")


if __name__ == "__main__":
    main()

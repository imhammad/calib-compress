"""
Phase 2: fix the evaluation protocol (cross-fitted oracle temperature).

Problem (raised by Paolo, 9 Nov 2026): in Phase 1 the oracle temperature is
fit on a target domain and then scored on the SAME images, so the oracle ECE
is slightly optimistic and the transfer gap slightly too large.

Fix: split the images of each target domain into two class-balanced halves,
A and B. The same split is used for the CIFAR-10 test set and all 75
CIFAR-10-C conditions (they hold the same 10,000 images in the same order),
and a separate split for STL-9 (7,200 images). Then, for every checkpoint
and domain:
    oracle (cross-fitted): fit T on A, score on B; fit T on B, score on A;
                           average the two halves.
    source:                T fit on cifar10_val as before, scored on the
                           same two halves and averaged.
    gap = source - oracle(cross-fitted), for ECE, adaptive ECE, classwise
          ECE, NLL and Brier.
Also reported: the same-half (in-sample) oracle, so the optimism can be
measured; absolute calibration after source TS on the full set; macro-F1,
balanced accuracy, per-class precision/recall/F1 and confusion matrices.

Nothing in results/ from earlier phases is changed. Everything is written to
results/phase2/.

Checks (the script stops if one fails):
  0. all 2,574 logit files are present; the Phase 1 (fixed-fit) CSV is found
     by its md5.
  1. every CIFAR-10-C condition has exactly the CIFAR-10 test labels in the
     same order; STL-9 labels are the same for every checkpoint.
  2. the split is exactly class-balanced and A and B do not overlap.
  3. full-set numbers reproduce the Phase 1 CSV (source T, oracle T, all
     raw / source-TS / oracle-TS metrics).
  4. on every half, the same-half oracle NLL is <= the source NLL and <= the
     other half's T (the fit minimises NLL on that half).
  5. per-class metrics agree with accuracy (mean recall = balanced accuracy).

Usage, from the calib-compress folder:
    python3 src/phase2_crossfit.py
Options:
    --split-seed N     seed of the main A/B split (default 0)
    --extra-splits K   repeat the gap computation with K more split seeds,
                       to show the result does not depend on one split
                       (default 4; 0 to skip)
    --quick            only 3 checkpoints (kd_s0, scratch_s0, dense_s0), for a
                       fast trial; skips the comparison tables
"""
import argparse
import hashlib
import os
import sys
import time

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.special import logsumexp

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from metrics import compute_all_metrics  # noqa: E402  (same metric code as Phase 1)

FROG_IDX = 6
N_CLASSES = 10
LOGITS = "logits"
OUT = "results/phase2"
# md5 of the Phase 1 fixed-fit CSV (calibration_results_fixed.csv, 9 Nov 2026)
PHASE1_MD5 = "09e6c1586856e2c79fe78ef84a9d9cdf"
PHASE1_CANDIDATES = ["results/calibration_results_fixed.csv", "results/calibration_results.csv"]
CAL = ["ece", "adaptive_ece", "classwise_ece", "nll", "brier"]   # calibration metrics
ORDER = ["dense", "denseft", "unstr30", "unstr50", "unstr70", "unstr90",
         "str15", "str30", "int8", "kd", "scratch"]
T_MIN, T_MAX = 0.05, 100.0
TOL_METRIC = 1e-6
TOL_T = 1e-5


# ---------------------------------------------------------------- basics
def nll_at_T(z, y, T):
    s = np.asarray(z, dtype=np.float64) / T
    return float((logsumexp(s, axis=1) - s[np.arange(len(y)), y]).mean())


def fit_T(z, y):
    """Same fixed fit as Phase 1: bounded 1-D search on log T."""
    lo, hi = np.log(T_MIN), np.log(T_MAX)
    r = minimize_scalar(lambda lt: nll_at_T(z, y, np.exp(lt)), bounds=(lo, hi),
                        method="bounded", options={"xatol": 1e-6, "maxiter": 500})
    if not r.success:
        raise RuntimeError(f"fit did not converge: {r.message}")
    if not (lo + 1e-3 < r.x < hi - 1e-3):
        raise RuntimeError(f"temperature hit the search bound: T={np.exp(r.x):.4g}")
    return float(np.exp(r.x))


def cal_metrics(z, y):
    m = compute_all_metrics(z, y)
    return {k: float(m[k]) for k in CAL}


def load(ckpt, domain):
    d = np.load(f"{LOGITS}/{ckpt}__{domain}.npz")
    z, y = d["logits"], d["labels"].astype(np.int64)
    if domain == "stl9":
        z = z.copy()
        z[:, FROG_IDX] = -1e4
        assert not (y == FROG_IDX).any()
    return z, y


def md5(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def stop(msg):
    sys.exit(f"STOP: {msg}\nNothing more was written. Send this output to Claude.")


def make_split(labels, seed):
    """Class-balanced split: within each class, a random half goes to A."""
    rng = np.random.default_rng(seed)
    a = []
    for c in np.unique(labels):
        idx = np.flatnonzero(labels == c)
        idx = rng.permutation(idx)
        a.append(idx[: len(idx) // 2])
    a = np.sort(np.concatenate(a))
    b = np.setdiff1d(np.arange(len(labels)), a)
    return a, b


def check_split(labels, a, b, name):
    if len(np.intersect1d(a, b)) or len(a) + len(b) != len(labels):
        stop(f"{name} split: halves overlap or do not cover all images")
    ca = np.bincount(labels[a], minlength=N_CLASSES)
    cb = np.bincount(labels[b], minlength=N_CLASSES)
    if np.abs(ca - cb).max() > 1:
        stop(f"{name} split is not class-balanced: {ca} vs {cb}")
    return ca, cb


def classification(z, y):
    """Argmax-based metrics (do not depend on temperature)."""
    pred = z.argmax(axis=1)
    cm = np.bincount(y * N_CLASSES + pred, minlength=N_CLASSES ** 2).reshape(N_CLASSES, N_CLASSES)
    support = cm.sum(axis=1)
    predicted = cm.sum(axis=0)
    tp = np.diag(cm).astype(float)
    with np.errstate(divide="ignore", invalid="ignore"):
        prec = np.where(predicted > 0, tp / predicted, 0.0)
        rec = np.where(support > 0, tp / support, np.nan)
        f1 = np.where(prec + np.nan_to_num(rec) > 0, 2 * prec * rec / (prec + rec), 0.0)
    present = support > 0          # STL-9: frog has no images and is never predicted
    if (predicted[~present] > 0).any():
        stop("a class with no images was predicted (frog mask failed?)")
    summary = {
        "accuracy": float(tp.sum() / len(y)),
        "balanced_accuracy": float(np.nanmean(rec[present])),
        "macro_f1": float(np.mean(f1[present])),
        "macro_precision": float(np.mean(prec[present])),
    }
    per_class = [{"class": c, "support": int(support[c]), "predicted": int(predicted[c]),
                  "precision": float(prec[c]), "recall": float(rec[c]), "f1": float(f1[c])}
                 for c in range(N_CLASSES) if present[c]]
    return summary, per_class, cm


def crossfit_row(z, y, a, b, T_src):
    """Half-based numbers for one checkpoint-domain pair and one split."""
    za, ya, zb, yb = z[a], y[a], z[b], y[b]
    T_a, T_b = fit_T(za, ya), fit_T(zb, yb)
    # check 4: each half's own T minimises NLL on that half
    for zz, yy, T_own, T_other, h in [(za, ya, T_a, T_b, "A"), (zb, yb, T_b, T_a, "B")]:
        n_own = nll_at_T(zz, yy, T_own)
        if n_own > nll_at_T(zz, yy, T_src) + 1e-9 or n_own > nll_at_T(zz, yy, T_other) + 1e-9:
            stop(f"half {h}: same-half oracle NLL is not the smallest (T_own={T_own}, "
                 f"T_other={T_other}, T_src={T_src})")
    raw = [cal_metrics(za, ya), cal_metrics(zb, yb)]
    src = [cal_metrics(za / T_src, ya), cal_metrics(zb / T_src, yb)]
    xfit = [cal_metrics(za / T_b, ya), cal_metrics(zb / T_a, yb)]     # fit on other half
    insm = [cal_metrics(za / T_a, ya), cal_metrics(zb / T_b, yb)]     # fit on same half
    row = {"T_A": T_a, "T_B": T_b}
    for m in CAL:
        row[f"raw_half_{m}"] = (raw[0][m] + raw[1][m]) / 2
        row[f"src_half_{m}"] = (src[0][m] + src[1][m]) / 2
        row[f"xfit_{m}"] = (xfit[0][m] + xfit[1][m]) / 2
        row[f"insample_half_{m}"] = (insm[0][m] + insm[1][m]) / 2
        row[f"gap_{m}"] = row[f"src_half_{m}"] - row[f"xfit_{m}"]
    row["src_A_ece"], row["src_B_ece"] = src[0]["ece"], src[1]["ece"]
    row["xfit_A_ece"], row["xfit_B_ece"] = xfit[0]["ece"], xfit[1]["ece"]
    return row


# ---------------------------------------------------------------- summaries
def tidy(df):
    df = df.copy()
    df["sev"] = np.where(df.domain.str.startswith("cifar10c_"), df.domain.str[-1], "")
    df["corr"] = df.domain.str.extract(r"cifar10c_(.+)_sev\d", expand=False)
    return df


def headline(df, gap, src, raw, orc, label):
    """Paper headline numbers from one results table. gap/src/raw/orc map a
    metric name to the column holding it."""
    df = tidy(df)
    c10c = df[df.domain.str.startswith("cifar10c_")]
    stl = df[df.domain == "stl9"]
    c10 = df[df.domain == "cifar10_test"]
    out = {}
    for name, d in [("c10c", c10c), ("stl9", stl), ("c10test", c10)]:
        ps = d.groupby(["method", "seed"])[gap["ece"]].mean().unstack("seed")
        for m in ORDER:
            if m not in ps.index:
                continue
            out[f"gap_{name}/{m}/mean"] = ps.loc[m].mean()
            out[f"gap_{name}/{m}/sd"] = ps.loc[m].std()
            for s in ps.columns:
                out[f"gap_{name}/{m}/seed{s}"] = ps.loc[m, s]
        if name == "c10test":
            continue
        if {"kd", "scratch"} <= set(ps.index):
            out[f"kd_over_scratch_ratio/{name}"] = ps.loc["kd"].mean() / ps.loc["scratch"].mean()
        for x, y in [("kd", "scratch"), ("kd", "dense"), ("denseft", "dense"), ("int8", "dense")]:
            if {x, y} <= set(ps.index):
                for s in ps.columns:
                    out[f"{x}_minus_{y}/{name}/seed{s}"] = ps.loc[x, s] - ps.loc[y, s]
        for p in ["unstr30", "unstr50", "unstr70", "unstr90", "str15", "str30", "int8"]:
            if {p, "denseft"} <= set(ps.index):
                dd = ps.loc[p] - ps.loc["denseft"]
                out[f"{p}_minus_denseft/{name}/seeds_positive_of_3"] = int((dd > 0).sum())
        # KD minus scratch for the other calibration metrics: seeds positive
        for m in CAL:
            pm = d.groupby(["method", "seed"])[gap[m]].mean().unstack("seed")
            if {"kd", "scratch"} <= set(pm.index):
                out[f"kd_minus_scratch_{m}/{name}/seeds_positive_of_3"] = int((pm.loc["kd"] - pm.loc["scratch"] > 0).sum())
        # absolute calibration after source TS (ECE), per arm, mean over seeds
        pa = d.groupby(["method", "seed"])[src["ece"]].mean().unstack("seed")
        for m in ["dense", "kd", "scratch"]:
            if m in pa.index:
                out[f"abs_srcts_ece_{name}/{m}/mean"] = pa.loc[m].mean()
                for s in pa.columns:
                    out[f"abs_srcts_ece_{name}/{m}/seed{s}"] = pa.loc[m, s]
        # recovered fraction of the oracle improvement
        for m in ORDER:
            r = d[d.method == m][[raw["ece"], src["ece"], orc["ece"]]].mean()
            if len(d[d.method == m]):
                out[f"recovered_frac_{name}/{m}"] = (r.iloc[0] - r.iloc[1]) / (r.iloc[0] - r.iloc[2])
    if len(c10c) and {"kd", "scratch"} <= set(c10c.method):
        sv = c10c.groupby(["method", "sev"])[gap["ece"]].mean().unstack()
        for s in sv.columns:
            out[f"kd_minus_scratch_sev{s}/c10c"] = sv.loc["kd", s] - sv.loc["scratch", s]
        dc = c10c.groupby(["method", "corr"])[gap["ece"]].mean().unstack()
        dc = dc.loc["kd"] - dc.loc["scratch"]
        out["kd_minus_scratch_corruptions_positive_of_15/c10c"] = int((dc > 0).sum())
        out["kd_minus_scratch_smallest_corruption/c10c"] = dc.min()
        out["kd_minus_scratch_smallest_corruption_name/c10c"] = dc.idxmin()
        out["kd_minus_scratch_largest_corruption/c10c"] = dc.max()
        out["kd_minus_scratch_largest_corruption_name/c10c"] = dc.idxmax()
    return pd.Series(out, name=label)


def phase1_view(p1):
    """Phase 1 CSV in the same shape (full-set, in-sample oracle)."""
    g = p1.copy()
    for m in CAL:
        g[f"gap_{m}"] = g[f"sourcets_{m}"] - g[f"oraclets_{m}"]
    return g


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split-seed", type=int, default=0)
    ap.add_argument("--extra-splits", type=int, default=4)
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    t0 = time.time()
    os.makedirs(OUT, exist_ok=True)

    # ---- Step 0: inputs
    print("== Step 0: inputs ==")
    if not os.path.isdir(LOGITS):
        sys.exit("Run this from the calib-compress folder (need logits/ and results/).")
    p1_path = next((p for p in PHASE1_CANDIDATES if os.path.isfile(p) and md5(p) == PHASE1_MD5), None)
    if p1_path is None:
        stop("could not find the Phase 1 fixed-fit CSV (md5 09e6c158...) in results/")
    p1 = pd.read_csv(p1_path)
    print(f"Phase 1 CSV: {p1_path} ({len(p1)} rows, md5 {PHASE1_MD5[:8]}..., matches)")
    ckpts = sorted(p1.checkpoint.unique())
    if args.quick:
        ckpts = [c for c in ckpts if c in ("resnet10_kd_s0", "resnet10_scratch_s0", "resnet18_dense_s0")]
        print(f"(--quick: only {ckpts})")
    domains = sorted(p1.domain.unique())
    missing = [f"{c}__{d}" for c in ckpts for d in ["cifar10_val"] + domains
               if not os.path.exists(f"{LOGITS}/{c}__{d}.npz")]
    if missing:
        stop(f"{len(missing)} logit files are missing, e.g. {missing[:3]}")
    print(f"logit files: all {len(ckpts)} checkpoints x {len(domains) + 1} domains present")

    # ---- Step 1: labels and the split
    print("\n== Step 1: labels and the A/B split ==")
    y_c10 = load(ckpts[0], "cifar10_test")[1]
    y_stl = load(ckpts[0], "stl9")[1]
    n_c10c = 0
    for c in ckpts:
        for d in domains:
            if d == "stl9":
                if not np.array_equal(load(c, d)[1], y_stl):
                    stop(f"{c} stl9 labels differ from {ckpts[0]}")
            else:
                if not np.array_equal(load(c, d)[1], y_c10):
                    stop(f"{c} {d}: labels are not the CIFAR-10 test labels in the same order")
                n_c10c += d.startswith("cifar10c_")
    print(f"CIFAR-10 test + {n_c10c // len(ckpts)} CIFAR-10-C conditions: identical labels, same order "
          f"({len(y_c10)} images) for all {len(ckpts)} checkpoints")
    print(f"STL-9: identical labels for all checkpoints ({len(y_stl)} images)")

    def build(seed):
        a1, b1 = make_split(y_c10, seed)
        a2, b2 = make_split(y_stl, seed)
        return {"cifar10": (a1, b1), "stl9": (a2, b2)}

    split = build(args.split_seed)
    ca, cb = check_split(y_c10, *split["cifar10"], "CIFAR-10")
    sa, sb = check_split(y_stl, *split["stl9"], "STL-9")
    print(f"split seed {args.split_seed}: CIFAR-10 A/B = {len(split['cifar10'][0])}/{len(split['cifar10'][1])}, "
          f"per class {ca.min()}-{ca.max()} / {cb.min()}-{cb.max()}")
    print(f"                STL-9 A/B = {len(split['stl9'][0])}/{len(split['stl9'][1])}, "
          f"per class {sa[sa > 0].min()}-{sa.max()} / {sb[sb > 0].min()}-{sb.max()} (frog: 0)")
    np.savez(f"{OUT}/split_seed{args.split_seed}.npz",
             cifar10_A=split["cifar10"][0], cifar10_B=split["cifar10"][1],
             stl9_A=split["stl9"][0], stl9_B=split["stl9"][1])
    print(f"saved {OUT}/split_seed{args.split_seed}.npz")

    def which(d):
        return "stl9" if d == "stl9" else "cifar10"

    # ---- Step 2: main computation
    print("\n== Step 2: full-set check against Phase 1, cross-fitted oracle, new metrics ==")
    rows, pcs, cms, cm_keys = [], [], [], []
    worst = {"T": 0.0, "metric": 0.0}
    where = {}
    p1i = p1.set_index(["checkpoint", "domain"])
    for k, c in enumerate(ckpts, 1):
        zv, yv = load(c, "cifar10_val")
        T_src = fit_T(zv, yv)
        for d in domains:
            z, y = load(c, d)
            ref = p1i.loc[(c, d)]
            # check 3: full-set numbers reproduce Phase 1
            T_or = fit_T(z, y)
            full = {"raw": compute_all_metrics(z, y), "sourcets": compute_all_metrics(z / T_src, y),
                    "oraclets": compute_all_metrics(z / T_or, y)}
            for name, val in [("source_T", T_src), ("oracle_T", T_or)]:
                dd = abs(val - ref[name]) / ref[name]
                if dd > worst["T"]:
                    worst["T"], where["T"] = dd, (c, d, name)
            for pre, vals in full.items():
                for m, v in vals.items():
                    # relative for large values: the CSV stores float32 NLLs as text
                    dd = abs(v - ref[f"{pre}_{m}"]) / max(1.0, abs(ref[f"{pre}_{m}"]))
                    if dd > worst["metric"]:
                        worst["metric"], where["metric"] = dd, (c, d, f"{pre}_{m}")
            arch, method, seed = ref["arch"], ref["method"], int(ref["seed"])
            row = {"checkpoint": c, "arch": arch, "method": method, "seed": seed, "domain": d,
                   "n_samples": len(y), "source_T": T_src, "oracle_T_full": T_or}
            summ, pc, cm = classification(z, y)
            if abs(summ["accuracy"] - full["raw"]["accuracy"]) > 1e-12:
                stop(f"{c} {d}: accuracy from confusion matrix differs")
            row.update(summ)
            for m in CAL:
                row[f"raw_full_{m}"] = full["raw"][m]
                row[f"src_full_{m}"] = full["sourcets"][m]
                row[f"oracle_insample_full_{m}"] = full["oraclets"][m]
            row.update(crossfit_row(z, y, *split[which(d)], T_src))
            rows.append(row)
            for r in pc:
                pcs.append({"checkpoint": c, "method": method, "seed": seed, "domain": d, **r})
            cms.append(cm)
            cm_keys.append(f"{c}__{d}")
        print(f"  {k}/{len(ckpts)} {c}: source T {T_src:.4f}  ({(time.time() - t0) / 60:.1f} min)", flush=True)
    print(f"full-set reproduction of Phase 1: worst T rel diff {worst['T']:.1e} {where.get('T', '')}")
    print(f"                                  worst metric abs diff {worst['metric']:.1e} {where.get('metric', '')}")
    if worst["T"] > TOL_T or worst["metric"] > TOL_METRIC:
        stop("the full-set numbers do not reproduce the Phase 1 CSV (tolerances T 1e-5, metrics 1e-6)")
    print("CHECK PASSED: full-set numbers reproduce the Phase 1 CSV.")

    res = pd.DataFrame(rows)
    pcdf = pd.DataFrame(pcs)
    # check 5: balanced accuracy = mean per-class recall; balanced data -> equals accuracy
    chk = pcdf.groupby(["checkpoint", "domain"]).recall.mean()
    bal = res.set_index(["checkpoint", "domain"]).balanced_accuracy.loc[chk.index]
    if (chk - bal).abs().max() > 1e-12:
        stop("balanced accuracy does not equal the mean per-class recall")
    print(f"balanced accuracy minus accuracy: max |diff| {(res.balanced_accuracy - res.accuracy).abs().max():.1e} "
          f"(0 expected: every domain is class-balanced)")
    n_neg = int((res.gap_ece < 0).sum())
    print(f"rows where the cross-fitted ECE gap is negative (source beats held-out oracle): {n_neg} of {len(res)}")
    opt = (res.xfit_ece - res.insample_half_ece)
    print(f"optimism of the same-half oracle (held-out ECE minus same-half ECE): "
          f"mean {opt.mean():.4f}, median {opt.median():.4f}, min {opt.min():.4f}, max {opt.max():.4f}")

    res.to_csv(f"{OUT}/calibration_results_phase2.csv", index=False)
    pcdf.to_csv(f"{OUT}/per_class_metrics.csv", index=False)
    np.savez_compressed(f"{OUT}/confusion_matrices.npz", keys=np.array(cm_keys), counts=np.stack(cms))
    print(f"wrote {OUT}/calibration_results_phase2.csv ({len(res)} rows), per_class_metrics.csv "
          f"({len(pcdf)} rows), confusion_matrices.npz ({len(cms)} matrices, rows = true class, counts)")

    if args.quick:
        print(f"\n(--quick: stopping before the comparison tables.)  {(time.time() - t0) / 60:.1f} min")
        return

    # ---- Step 3: other split seeds (robustness)
    extra = {}
    if args.extra_splits:
        print(f"\n== Step 3: {args.extra_splits} more split seeds (robustness of the gaps) ==")
        for s in range(args.split_seed + 1, args.split_seed + 1 + args.extra_splits):
            sp = build(s)
            check_split(y_c10, *sp["cifar10"], "CIFAR-10")
            check_split(y_stl, *sp["stl9"], "STL-9")
            er = []
            for c in ckpts:
                T_src = res.loc[res.checkpoint == c, "source_T"].iloc[0]
                for d in domains:
                    z, y = load(c, d)
                    r = crossfit_row(z, y, *sp[which(d)], T_src)
                    er.append({"checkpoint": c, "method": c.split("_")[1], "seed": int(c[-1]), "domain": d,
                               **{kk: r[kk] for kk in r if kk.startswith(("gap_", "src_half_", "raw_half_", "xfit_"))}})
            extra[s] = pd.DataFrame(er)
            print(f"  split seed {s} done  ({(time.time() - t0) / 60:.1f} min)", flush=True)

    # ---- Step 4: comparison with Phase 1
    print("\n== Step 4: Phase 1 (same-image oracle) vs Phase 2 (cross-fitted oracle) ==")
    p1v = phase1_view(p1[p1.checkpoint.isin(ckpts)])
    cols1 = dict(gap={m: f"gap_{m}" for m in CAL}, src={m: f"sourcets_{m}" for m in CAL},
                 raw={m: f"raw_{m}" for m in CAL}, orc={m: f"oraclets_{m}" for m in CAL})
    cols2 = dict(gap={m: f"gap_{m}" for m in CAL}, src={m: f"src_full_{m}" for m in CAL},
                 raw={m: f"raw_half_{m}" for m in CAL}, orc={m: f"xfit_{m}" for m in CAL})
    h1 = headline(p1v, label="phase1_same_image_oracle", **cols1)
    h2 = headline(res, label="phase2_crossfit",
                  gap=cols2["gap"], src=cols2["src"], raw=cols2["raw"], orc=cols2["orc"])
    # recovered fraction must use half-based source ECE (like for like with xfit and raw_half)
    h2rf = headline(res, label="tmp", gap=cols2["gap"], src={m: f"src_half_{m}" for m in CAL},
                    raw=cols2["raw"], orc=cols2["orc"])
    for k2 in h2.index:
        if k2.startswith("recovered_frac_"):
            h2[k2] = h2rf[k2]
    table = pd.concat([h1, h2], axis=1)
    num = pd.to_numeric(table.phase1_same_image_oracle, errors="coerce")
    num2 = pd.to_numeric(table.phase2_crossfit, errors="coerce")
    table["change"] = num2 - num
    if extra:
        hs = [headline(e, label=f"split{s}", gap=cols2["gap"], src={m: f"src_half_{m}" for m in CAL},
                       raw=cols2["raw"], orc=cols2["orc"]) for s, e in extra.items()]
        hs = pd.concat([h2] + hs, axis=1)
        hn = hs.apply(pd.to_numeric, errors="coerce")
        table["all_splits_min"] = hn.min(axis=1)
        table["all_splits_max"] = hn.max(axis=1)
        pd.concat([pd.DataFrame({"split_seed": s, **e.iloc[:, :4].to_dict("list"),
                                 "gap_ece": e.gap_ece}) for s, e in extra.items()]) \
            .to_csv(f"{OUT}/extra_splits_gap_ece.csv", index=False)
    table.to_csv(f"{OUT}/phase1_vs_phase2.csv")
    pd.set_option("display.max_rows", 1000)
    pd.set_option("display.width", 200)
    pd.set_option("display.max_colwidth", 40)
    show = [k for k in table.index if ("/mean" in k or "ratio" in k or "minus_scratch/" in k
                                       or "recovered_frac_c10c" in k or "positive" in k or "sev" in k
                                       or "corruption" in k)]
    print(table.loc[show].to_string(float_format=lambda v: f"{v:.4f}"))
    print(f"saved {OUT}/phase1_vs_phase2.csv (all {len(table)} numbers, incl. per-seed values)")
    print(f"\nALL DONE in {(time.time() - t0) / 60:.1f} min.")


if __name__ == "__main__":
    main()

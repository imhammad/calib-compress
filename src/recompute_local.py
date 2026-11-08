"""
Recompute calibration_results.csv with the fixed temperature fit, from the
logits already on disk. Runs on CPU (written for Hammad's Mac). Never
overwrites the old CSV.

Step 1  Reproduce v1. For every row of the old CSV, recompute the raw
        metrics and rerun the OLD temperature fit (torch L-BFGS, lr=0.01,
        no line search) on the same logits, and compare with the old CSV.
        Also re-apply the old CSV's temperatures and compare the scaled
        metrics. If anything is off beyond tiny float noise, stop here.
Step 2  Recompute with the fixed fit (bounded 1-D search on log T), check
        oracle NLL <= source NLL on every row, cross-check a sample of fits
        against L-BFGS with a strong-Wolfe line search, and write
        results/calibration_results_fixed.csv.
Step 3  Old vs new headline numbers -> results/old_vs_new.csv, and rerun the
        paper scripts on the new CSV inside results_fixed_run/ (the old
        results/ folder and figures are left untouched).

Usage, from the calib-compress folder:
    python3 src/recompute_local.py
Options:
    --quick      reproduce only 150 random rows in step 1 (for a fast trial)
    --force      continue to step 2 even if step 1 finds differences
"""
import argparse
import hashlib
import os
import shutil
import subprocess
import sys
import time

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy.optimize import minimize_scalar
from scipy.special import logsumexp

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from metrics import compute_all_metrics  # noqa: E402  (metric code is unchanged by the fix)

FROG_IDX = 6
OLD_CSV = "results/calibration_results.csv"
NEW_CSV = "results/calibration_results_fixed.csv"
LOGITS = "logits"
METRICS = ["accuracy", "nll", "brier", "ece", "adaptive_ece", "classwise_ece"]
T_MIN, T_MAX = 0.05, 100.0
TOL_METRIC = 1e-6   # raw / scaled metrics: same numpy code on the same data
TOL_T = 1e-4        # old L-BFGS fit: float32, can move in the 7th digit with thread count


# ---------------------------------------------------------------- fitters
def fit_temperature_old(logits, labels, max_iter=200, lr=0.01):
    """Exact copy of the fit used for v1 (does not converge)."""
    logits_t = torch.tensor(logits, dtype=torch.float32)
    labels_t = torch.tensor(labels, dtype=torch.int64)
    log_T = torch.zeros(1, requires_grad=True)
    opt = torch.optim.LBFGS([log_T], lr=lr, max_iter=max_iter)

    def closure():
        opt.zero_grad()
        loss = F.cross_entropy(logits_t / torch.exp(log_T), labels_t)
        loss.backward()
        return loss

    opt.step(closure)
    return torch.exp(log_T).item()


def nll_at_T(logits, labels, T):
    z = np.asarray(logits, dtype=np.float64) / T
    return float((logsumexp(z, axis=1) - z[np.arange(len(labels)), labels]).mean())


def fit_temperature_fixed(logits, labels):
    """Bounded 1-D search on log T; NLL is convex in 1/T, so this finds the optimum."""
    lo, hi = np.log(T_MIN), np.log(T_MAX)
    r = minimize_scalar(lambda lt: nll_at_T(logits, labels, np.exp(lt)),
                        bounds=(lo, hi), method="bounded", options={"xatol": 1e-6, "maxiter": 500})
    if not r.success:
        raise RuntimeError(f"fit did not converge: {r.message}")
    if not (lo + 1e-3 < r.x < hi - 1e-3):
        raise RuntimeError(f"temperature hit the search bound: T={np.exp(r.x):.4g}")
    return float(np.exp(r.x))


def fit_temperature_lbfgs_check(logits, labels):
    """Independent check: L-BFGS, lr=1, strong-Wolfe line search, float64."""
    z = torch.tensor(np.asarray(logits), dtype=torch.float64)
    y = torch.tensor(np.asarray(labels), dtype=torch.int64)
    log_T = torch.zeros(1, dtype=torch.float64, requires_grad=True)
    opt = torch.optim.LBFGS([log_T], lr=1.0, max_iter=100, line_search_fn="strong_wolfe",
                            tolerance_grad=1e-12, tolerance_change=1e-14)

    def closure():
        opt.zero_grad()
        loss = F.cross_entropy(z / torch.exp(log_T), y)
        loss.backward()
        return loss

    opt.step(closure)
    return float(torch.exp(log_T).item())


# ---------------------------------------------------------------- helpers
def load(ckpt, domain):
    d = np.load(f"{LOGITS}/{ckpt}__{domain}.npz")
    z, y = d["logits"], d["labels"]
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


def rel_diff(a, b):
    return abs(a - b) / max(abs(b), 1e-12)


def headline(df):
    """The paper's temperature-scaling numbers (plus accuracy as a control)."""
    order = ["dense", "denseft", "unstr30", "unstr50", "unstr70", "unstr90",
             "str15", "str30", "int8", "kd", "scratch"]
    df = df.copy()
    df["gap"] = df.sourcets_ece - df.oraclets_ece
    c10 = df[df.domain == "cifar10_test"]
    c10c = df[df.domain.str.startswith("cifar10c_")].copy()
    c10c["sev"] = c10c.domain.str[-1].astype(int)
    c10c["corr"] = c10c.domain.str.extract(r"cifar10c_(.+)_sev\d", expand=False)
    stl = df[df.domain == "stl9"]
    out = {}
    for m in order:
        out[f"acc_cifar10/{m}"] = c10[c10.method == m].raw_accuracy.mean() * 100
        out[f"T_src/{m}"] = c10[c10.method == m].source_T.mean()
        out[f"oracle_T_mean_c10c/{m}"] = c10c[c10c.method == m].oracle_T.mean()
        out[f"gap_c10c/{m}"] = c10c[c10c.method == m].gap.mean()
        out[f"gap_stl9/{m}"] = stl[stl.method == m].gap.mean()
        r = c10c[c10c.method == m][["raw_ece", "sourcets_ece", "oraclets_ece"]].mean()
        out[f"recovered_frac_c10c/{m}"] = (r.raw_ece - r.sourcets_ece) / (r.raw_ece - r.oraclets_ece)
    for name, d in [("c10c", c10c), ("stl9", stl)]:
        ps = d.groupby(["method", "seed"]).gap.mean().unstack("seed")
        out[f"kd_over_scratch_ratio/{name}"] = ps.loc["kd"].mean() / ps.loc["scratch"].mean()
        for a, b in [("kd", "scratch"), ("kd", "dense"), ("denseft", "dense")]:
            for s in [0, 1, 2]:
                out[f"{a}_minus_{b}_seed{s}/{name}"] = ps.loc[a, s] - ps.loc[b, s]
    sv = c10c.groupby(["method", "sev"]).gap.mean().unstack()
    for s in range(1, 6):
        out[f"kd_minus_scratch_sev{s}/c10c"] = sv.loc["kd", s] - sv.loc["scratch", s]
    diff = c10c.groupby(["method", "corr"]).gap.mean().unstack()
    diff = diff.loc["kd"] - diff.loc["scratch"]
    out["kd_minus_scratch_corruptions_positive_of_15/c10c"] = int((diff > 0).sum())
    out["kd_minus_scratch_smallest_corruption/c10c"] = diff.min()
    out["kd_minus_scratch_largest_corruption/c10c"] = diff.max()
    out["rows_oracle_nll_above_source_nll"] = int((df.oraclets_nll > df.sourcets_nll + 1e-6).sum())
    return pd.Series(out)


# ---------------------------------------------------------------- main
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--quick", action="store_true")
    p.add_argument("--force", action="store_true")
    args = p.parse_args()
    t0 = time.time()

    # ---- Step 0: inputs
    print("== Step 0: inputs ==")
    if not os.path.isfile(OLD_CSV) or not os.path.isdir(LOGITS):
        sys.exit(f"Run this from the calib-compress folder (need {OLD_CSV} and {LOGITS}/).")
    old = pd.read_csv(OLD_CSV)
    n_files = len([f for f in os.listdir(LOGITS) if f.endswith(".npz")])
    print(f"old CSV: {len(old)} rows, md5 {md5(OLD_CSV)}")
    print(f"logit files in {LOGITS}/: {n_files}")
    ckpts = sorted(old.checkpoint.unique())
    missing = [f"{c}__{d}" for c in ckpts for d in ["cifar10_val"] + sorted(old.domain.unique())
               if not os.path.exists(f"{LOGITS}/{c}__{d}.npz")]
    if missing:
        sys.exit(f"STOP: {len(missing)} logit files are missing, e.g. {missing[:3]}")
    print(f"all {len(ckpts)} checkpoints x {old.domain.nunique() + 1} domains present")

    # ---- Step 1: reproduce v1
    print("\n== Step 1: reproduce the old numbers from these logits ==")
    rows = old if not args.quick else old.sample(150, random_state=0)
    if args.quick:
        print("(--quick: checking 150 random rows only)")
    worst = {"raw": 0.0, "scaled": 0.0, "source_T": 0.0, "oracle_T": 0.0}
    where = {}
    src_old_fit = {}
    for c in ckpts:
        zv, yv = load(c, "cifar10_val")
        src_old_fit[c] = fit_temperature_old(zv, yv)
        d = rel_diff(src_old_fit[c], old[old.checkpoint == c].source_T.iloc[0])
        if d > worst["source_T"]:
            worst["source_T"], where["source_T"] = d, (c, "cifar10_val")
    print(f"source T (old fit) reproduced for all {len(ckpts)} checkpoints; "
          f"worst relative difference {worst['source_T']:.1e}")
    for i, (_, r) in enumerate(rows.iterrows(), 1):
        z, y = load(r.checkpoint, r.domain)
        checks = [("raw", compute_all_metrics(z, y), "raw_"),
                  ("scaled", compute_all_metrics(z / r.source_T, y), "sourcets_"),
                  ("scaled", compute_all_metrics(z / r.oracle_T, y), "oraclets_")]
        for kind, vals, prefix in checks:
            for m in METRICS:
                dd = abs(vals[m] - r[prefix + m])
                if dd > worst[kind]:
                    worst[kind], where[kind] = dd, (r.checkpoint, r.domain, prefix + m)
        d = rel_diff(fit_temperature_old(z, y), r.oracle_T)
        if d > worst["oracle_T"]:
            worst["oracle_T"], where["oracle_T"] = d, (r.checkpoint, r.domain)
        if i % 250 == 0 or i == len(rows):
            print(f"  {i}/{len(rows)} rows checked  ({(time.time() - t0) / 60:.1f} min)", flush=True)
    print(f"raw metrics:                    worst abs difference {worst['raw']:.1e}  {where.get('raw', '')}")
    print(f"metrics at the old CSV's Ts:    worst abs difference {worst['scaled']:.1e}  {where.get('scaled', '')}")
    print(f"oracle T (old fit, refit here): worst rel difference {worst['oracle_T']:.1e}  {where.get('oracle_T', '')}")
    ok1 = (worst["raw"] <= TOL_METRIC and worst["scaled"] <= TOL_METRIC
           and worst["source_T"] <= TOL_T and worst["oracle_T"] <= TOL_T)
    if ok1:
        print("STEP 1 PASSED: these logits and this code reproduce the old CSV.")
    else:
        print("STEP 1 FAILED: the old CSV is not reproduced (tolerances: metrics 1e-6, T 1e-4).")
        if not args.force:
            sys.exit("Stopped before writing anything. Send this output to Claude.")
        print("--force given: continuing anyway.")

    # ---- Step 2: fixed fit
    print("\n== Step 2: recompute with the fixed temperature fit ==")
    out = []
    for k, c in enumerate(ckpts, 1):
        zv, yv = load(c, "cifar10_val")
        T_src = fit_temperature_fixed(zv, yv)
        for _, r in old[old.checkpoint == c].iterrows():
            z, y = load(c, r.domain)
            T_or = fit_temperature_fixed(z, y)
            n_src, n_or = nll_at_T(z, y, T_src), nll_at_T(z, y, T_or)
            if n_or > n_src + 1e-9:
                sys.exit(f"STOP: {c} {r.domain}: oracle NLL {n_or:.8f} > source NLL {n_src:.8f}")
            row = {k2: r[k2] for k2 in ["checkpoint", "arch", "method", "seed", "domain", "n_samples"]}
            row["source_T"] = T_src
            for m, v in compute_all_metrics(z, y).items():
                row[f"raw_{m}"] = v
            for m, v in compute_all_metrics(z / T_src, y).items():
                row[f"sourcets_{m}"] = v
            row["oracle_T"] = T_or
            for m, v in compute_all_metrics(z / T_or, y).items():
                row[f"oraclets_{m}"] = v
            out.append(row)
        print(f"  {k}/{len(ckpts)} {c}: source T old {src_old_fit[c]:.4f} -> fixed {T_src:.4f}"
              f"  ({(time.time() - t0) / 60:.1f} min)", flush=True)
    new = pd.DataFrame(out)[list(old.columns)]
    assert len(new) == len(old) == 2541, f"expected 2541 rows, got {len(new)}"
    bad = int((new.oraclets_nll > new.sourcets_nll + 1e-6).sum())
    print(f"oracle NLL > source NLL: old CSV {int((old.oraclets_nll > old.sourcets_nll).sum())} rows, "
          f"new {bad} rows")
    if bad:
        sys.exit("STOP: the NLL check failed on the stored metrics. Send this output to Claude.")

    print("cross-check against L-BFGS with line search (15 largest oracle Ts, 25 random rows, 5 source fits):")
    rng = np.random.default_rng(0)
    pick = list(new.oracle_T.sort_values(ascending=False).index[:15]) + \
        list(rng.choice(len(new), 25, replace=False))
    worst_x = 0.0
    for j in pick:
        r = new.iloc[j]
        z, y = load(r.checkpoint, r.domain)
        worst_x = max(worst_x, rel_diff(r.oracle_T, fit_temperature_lbfgs_check(z, y)))
    for c in ckpts[:5]:
        zv, yv = load(c, "cifar10_val")
        worst_x = max(worst_x, rel_diff(new[new.checkpoint == c].source_T.iloc[0],
                                        fit_temperature_lbfgs_check(zv, yv)))
    print(f"  worst relative difference {worst_x:.1e} (must be < 1e-3)")
    if worst_x > 1e-3:
        sys.exit("STOP: the two fitters disagree. Send this output to Claude.")
    new.to_csv(NEW_CSV, index=False)
    print(f"STEP 2 PASSED. Wrote {NEW_CSV} ({len(new)} rows). {OLD_CSV} was not changed.")

    # ---- Step 3: comparison and paper scripts
    print("\n== Step 3: old vs new, and paper scripts on the new CSV ==")
    t = pd.DataFrame({"old_fit": headline(old), "fixed_fit": headline(new)})
    t["change"] = t.fixed_fit - t.old_fit
    t.to_csv("results/old_vs_new.csv")
    pd.set_option("display.max_rows", 500)
    pd.set_option("display.width", 160)
    print(t.round(4).to_string())
    print("saved results/old_vs_new.csv")

    run_dir = "results_fixed_run"
    os.makedirs(f"{run_dir}/results", exist_ok=True)
    shutil.copy(NEW_CSV, f"{run_dir}/results/calibration_results.csv")
    for script in ["check_numbers.py", "summarize_other_metrics.py",
                   "review_numbers2.py", "plot_kd_corruption.py"]:
        path = os.path.abspath(f"src/{script}")
        if not os.path.exists(path):
            print(f"  (skipped {script}: not found in src/)")
            continue
        res = subprocess.run([sys.executable, path], cwd=run_dir, capture_output=True, text=True,
                             env={**os.environ, "MPLBACKEND": "Agg"})
        with open(f"{run_dir}/{script.replace('.py', '_output.txt')}", "w") as f:
            f.write(res.stdout + res.stderr)
        print(f"  {script}: {'ok' if res.returncode == 0 else 'ERROR (see its output file)'}")
    print(f"paper-script outputs and both figures are in {run_dir}/")
    print(f"\nALL DONE in {(time.time() - t0) / 60:.1f} min.")


if __name__ == "__main__":
    main()

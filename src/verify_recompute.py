"""
Checks for the temperature-fit fix (Nov 2026). Compares a recomputed
results CSV against the v1 CSV and against the re-dumped logits.

  1. Same 2,541 (checkpoint, domain) rows in both CSVs.
  2. Regression check on raw (pre-scaling) metrics: these do not depend on
     temperature fitting, so they should match v1 closely. A mismatch means
     the re-dumped logits differ from the ones behind v1.
  3. Logit fingerprint: the OLD (legacy) fitter is deterministic given the
     logits, so running it on the new logits should reproduce v1's
     source_T and oracle_T. This is a tighter check than (2).
  4. Oracle-fit NLL <= source-fit NLL on every row of the new CSV
     (v1 had 144 violating rows).
  5. Cross-check of the new fitter against torch L-BFGS (lr=1, strong-Wolfe
     line search) on a sample of rows, including the largest temperatures.

Usage:
    python3 src/verify_recompute.py --old results/calibration_results_v1.csv \
        --new results/calibration_results.csv --logits-dir logits
Exits non-zero if a hard check (1, 4, 5) fails. Checks 2 and 3 are reported
with tolerances; read their output before trusting the new numbers.
"""
import argparse
import sys

import numpy as np
import pandas as pd

from analyze_calibration import FROG_IDX
from metrics import (fit_temperature, fit_temperature_legacy,
                     fit_temperature_lbfgs_reference)

RAW_COLS = ["raw_accuracy", "raw_nll", "raw_brier", "raw_ece",
            "raw_adaptive_ece", "raw_classwise_ece"]


def load_logits(logits_dir, ckpt, domain):
    d = np.load(f"{logits_dir}/{ckpt}__{domain}.npz")
    z, y = d["logits"], d["labels"]
    if domain == "stl9":
        z = z.copy()
        z[:, FROG_IDX] = -1e4
    return z, y


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--old", required=True)
    p.add_argument("--new", required=True)
    p.add_argument("--logits-dir", default="logits")
    p.add_argument("--n-fingerprint", type=int, default=60,
                   help="random rows on which to re-run the legacy fitter for oracle_T")
    p.add_argument("--n-crosscheck", type=int, default=40)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    rng = np.random.default_rng(args.seed)
    ok = True

    old = pd.read_csv(args.old).set_index(["checkpoint", "domain"]).sort_index()
    new = pd.read_csv(args.new).set_index(["checkpoint", "domain"]).sort_index()

    # 1. Row set
    print("== 1. Row set ==")
    print(f"old rows: {len(old)}   new rows: {len(new)}")
    if len(new) != 2541 or not old.index.equals(new.index):
        print("FAIL: row sets differ (expected the same 2,541 rows)")
        missing = old.index.difference(new.index)
        extra = new.index.difference(old.index)
        print(f"  missing from new: {len(missing)} e.g. {list(missing[:5])}")
        print(f"  extra in new: {len(extra)} e.g. {list(extra[:5])}")
        ok = False
        common = old.index.intersection(new.index)
        old, new = old.loc[common], new.loc[common]
    else:
        print("OK")

    # 2. Raw-metric regression
    print("\n== 2. Raw metrics, new minus v1 (should be ~0) ==")
    for c in RAW_COLS:
        diff = (new[c] - old[c]).abs()
        print(f"  {c:20s} max |diff| {diff.max():.2e}   mean {diff.mean():.2e}   "
              f"rows > 1e-4: {(diff > 1e-4).sum()}")
    acc_diff = (new.raw_accuracy - old.raw_accuracy).abs() * new.n_samples
    print(f"  max predictions that changed in one domain: {acc_diff.round().max():.0f}")
    worst = (new.raw_ece - old.raw_ece).abs().sort_values(ascending=False).head(5)
    print("  largest raw ECE differences:")
    for (ck, dm), v in worst.items():
        print(f"    {ck:24s} {dm:34s} {v:.2e}")

    # 3. Legacy-fitter fingerprint
    print("\n== 3. Legacy fitter on new logits vs v1 temperatures ==")
    src_d = []
    for ck in sorted(new.index.get_level_values(0).unique()):
        z, y = load_logits(args.logits_dir, ck, "cifar10_val")
        t_leg = fit_temperature_legacy(z, y)
        t_old = old.loc[(ck, "cifar10_test"), "source_T"]
        src_d.append(abs(t_leg - t_old))
    src_d = np.array(src_d)
    print(f"  source_T, all {len(src_d)} checkpoints: max |diff| {src_d.max():.2e}, "
          f"exact (< 1e-5) in {(src_d < 1e-5).sum()}")
    idx = rng.choice(len(new), size=min(args.n_fingerprint, len(new)), replace=False)
    orc_d = []
    for i in idx:
        ck, dm = new.index[i]
        z, y = load_logits(args.logits_dir, ck, dm)
        orc_d.append(abs(fit_temperature_legacy(z, y) - old.loc[(ck, dm), "oracle_T"]))
    orc_d = np.array(orc_d)
    print(f"  oracle_T, {len(orc_d)} random rows: max |diff| {orc_d.max():.2e}, "
          f"exact (< 1e-5) in {(orc_d < 1e-5).sum()}")
    print("  (exact matches mean the logits are the same ones behind v1)")

    # 4. Oracle NLL <= source NLL
    print("\n== 4. Oracle-fit NLL <= source-fit NLL ==")
    bad_old = (old.oraclets_nll > old.sourcets_nll).sum()
    bad_new = (new.oraclets_nll > new.sourcets_nll + 1e-6).sum()
    print(f"  violating rows: v1 {bad_old}, new {bad_new}")
    if bad_new:
        ok = False
        print("FAIL")
        print(new[new.oraclets_nll > new.sourcets_nll + 1e-6]
              [["source_T", "oracle_T", "sourcets_nll", "oraclets_nll"]].head(10))
    else:
        print("OK")

    # 5. Cross-check against L-BFGS with line search
    print("\n== 5. New fitter vs L-BFGS (lr=1, strong-Wolfe) ==")
    n_top = args.n_crosscheck // 3
    top = list(new.oracle_T.sort_values(ascending=False).index[:n_top])
    rest = [new.index[i] for i in rng.choice(len(new), size=args.n_crosscheck - n_top, replace=False)]
    worst_rel = 0.0
    for ck, dm in top + rest:
        z, y = load_logits(args.logits_dir, ck, dm)
        t_new = fit_temperature(z, y)
        t_ref = fit_temperature_lbfgs_reference(z, y)
        if abs(t_new - new.loc[(ck, dm), "oracle_T"]) > 1e-9:
            print(f"  FAIL: CSV oracle_T differs from refit for {ck} {dm}")
            ok = False
        rel = abs(t_new - t_ref) / t_ref
        worst_rel = max(worst_rel, rel)
        if (ck, dm) in top[:5]:
            print(f"  {ck:24s} {dm:34s} new {t_new:.5f}  lbfgs {t_ref:.5f}")
    for ck in sorted(new.index.get_level_values(0).unique())[:5]:
        z, y = load_logits(args.logits_dir, ck, "cifar10_val")
        t_new, t_ref = fit_temperature(z, y), fit_temperature_lbfgs_reference(z, y)
        worst_rel = max(worst_rel, abs(t_new - t_ref) / t_ref)
    print(f"  {len(top) + len(rest)} oracle fits + 5 source fits: "
          f"worst relative difference {worst_rel:.2e}")
    if worst_rel > 1e-3:
        ok = False
        print("FAIL: fitters disagree by more than 0.1%")
    else:
        print("OK")

    print("\nALL HARD CHECKS PASSED" if ok else "\nSOME CHECKS FAILED")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()

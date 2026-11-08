"""
Side-by-side table of the paper's temperature-scaling numbers computed
from the v1 CSV (old, unconverged fit) and the recomputed CSV (fixed fit).
Accuracy and raw metrics are included as a control; they should not move.

Usage:
    python3 src/compare_old_new.py --old results/calibration_results_v1.csv \
        --new results/calibration_results.csv [--out results/old_vs_new.csv]
"""
import argparse

import pandas as pd

ORDER = ["dense", "denseft", "unstr30", "unstr50", "unstr70", "unstr90",
         "str15", "str30", "int8", "kd", "scratch"]


def headline(path):
    df = pd.read_csv(path)
    df["gap"] = df.sourcets_ece - df.oraclets_ece
    c10 = df[df.domain == "cifar10_test"]
    c10c = df[df.domain.str.startswith("cifar10c_")].copy()
    c10c["sev"] = c10c.domain.str[-1].astype(int)
    c10c["corr"] = c10c.domain.str.extract(r"cifar10c_(.+)_sev\d", expand=False)
    stl = df[df.domain == "stl9"]
    out = {}

    def per_seed(d, col):
        return d.groupby(["method", "seed"])[col].mean().unstack("seed")

    for m in ORDER:
        out[f"acc_c10/{m}"] = c10[c10.method == m].raw_accuracy.mean() * 100
        out[f"raw_ece_c10c/{m}"] = c10c[c10c.method == m].raw_ece.mean()
        out[f"T_src/{m}"] = c10[c10.method == m].source_T.mean()
        out[f"oracle_T_c10c_mean/{m}"] = c10c[c10c.method == m].oracle_T.mean()
        out[f"ece_src_c10/{m}"] = c10[c10.method == m].sourcets_ece.mean()
        out[f"gap_c10c/{m}"] = c10c[c10c.method == m].gap.mean()
        out[f"gap_stl9/{m}"] = stl[stl.method == m].gap.mean()
        r = c10c[c10c.method == m][["raw_ece", "sourcets_ece", "oraclets_ece"]].mean()
        out[f"recovered_frac_c10c/{m}"] = (r.raw_ece - r.sourcets_ece) / (r.raw_ece - r.oraclets_ece)

    for name, d in [("c10c", c10c), ("stl9", stl)]:
        ps = per_seed(d, "gap")
        out[f"kd_over_scratch_ratio/{name}"] = ps.loc["kd"].mean() / ps.loc["scratch"].mean()
        for a, b in [("kd", "scratch"), ("kd", "dense"), ("denseft", "dense")]:
            for s in [0, 1, 2]:
                out[f"{a}_minus_{b}_seed{s}/{name}"] = ps.loc[a, s] - ps.loc[b, s]

    sv = c10c.groupby(["method", "sev"]).gap.mean().unstack()
    for s in range(1, 6):
        out[f"kd_minus_scratch_sev{s}/c10c"] = sv.loc["kd", s] - sv.loc["scratch", s]
    pc = c10c.groupby(["method", "corr"]).gap.mean().unstack()
    diff = pc.loc["kd"] - pc.loc["scratch"]
    out["kd_minus_scratch_corruptions_positive/c10c"] = int((diff > 0).sum())
    out["kd_minus_scratch_min_corruption/c10c"] = diff.min()
    out["kd_minus_scratch_max_corruption/c10c"] = diff.max()
    out["oracle_nll_gt_source_nll_rows"] = int((df.oraclets_nll > df.sourcets_nll + 1e-6).sum())
    return pd.Series(out)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--old", required=True)
    p.add_argument("--new", required=True)
    p.add_argument("--out", default="results/old_vs_new.csv")
    args = p.parse_args()
    t = pd.DataFrame({"v1_old_fit": headline(args.old), "fixed_fit": headline(args.new)})
    t["change"] = t.fixed_fit - t.v1_old_fit
    pd.set_option("display.width", 160)
    pd.set_option("display.max_rows", 500)
    print(t.round(4).to_string())
    t.to_csv(args.out)
    print(f"\nSaved {args.out}")


if __name__ == "__main__":
    main()

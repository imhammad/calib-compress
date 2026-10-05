"""
Summarizes NLL, Brier, adaptive ECE and classwise ECE alongside ECE,
so we can check whether the ECE-based findings hold on other metrics.
Transfer gap = source-fit metric minus oracle-fit metric.

Usage:
    python3 src/summarize_other_metrics.py
"""
import pandas as pd

pd.set_option("display.width", 140)
pd.set_option("display.max_columns", 20)

df = pd.read_csv("results/calibration_results.csv")
metrics = ["ece", "adaptive_ece", "classwise_ece", "nll", "brier"]

for m in metrics:
    df[f"gap_{m}"] = df[f"sourcets_{m}"] - df[f"oraclets_{m}"]

c10c = df[df.domain.str.startswith("cifar10c_")].copy()
stl = df[df.domain == "stl9"].copy()

print("=" * 70)
print("1. Mean transfer gap per method, CIFAR-10-C (avg over 75 conditions, 3 seeds)")
print("=" * 70)
tab = c10c.groupby(["arch", "method"])[[f"gap_{m}" for m in metrics]].mean().round(4)
tab = tab.sort_values("gap_ece")
print(tab.to_string())

print()
print("=" * 70)
print("2. Mean RAW metrics per method, CIFAR-10-C")
print("=" * 70)
raw = c10c.groupby(["arch", "method"])[[f"raw_{m}" for m in metrics]].mean().round(4)
print(raw.to_string())


def paired(data, a, b, label):
    print(f"\n{label}")
    for m in metrics:
        diffs = []
        for s in [0, 1, 2]:
            va = data[(data.method == a) & (data.seed == s)][f"gap_{m}"].mean()
            vb = data[(data.method == b) & (data.seed == s)][f"gap_{m}"].mean()
            diffs.append(va - vb)
        signs = "".join("+" if d > 0 else "-" for d in diffs)
        print(f"  {m:14s} " + "  ".join(f"{d:+.4f}" for d in diffs) + f"   signs: {signs}")


print()
print("=" * 70)
print("3. Paired per-seed differences in transfer gap (positive = first is worse)")
print("=" * 70)
paired(c10c, "kd", "scratch", "KD - scratch, CIFAR-10-C:")
paired(stl, "kd", "scratch", "KD - scratch, STL-9:")
paired(c10c, "denseft", "dense", "denseft - dense, CIFAR-10-C:")
paired(stl, "denseft", "dense", "denseft - dense, STL-9:")

"""
Final numbers before the rewrite:
1. KD vs its own dense teacher (same seed), on CIFAR-10-C and STL-9.
2. Gap-vs-severity figure for dense teacher, KD student, scratch student.
"""
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("results/calibration_results.csv")
df["gap"] = df.sourcets_ece - df.oraclets_ece
c10c = df[df.domain.str.startswith("cifar10c_")].copy()
stl = df[df.domain == "stl9"].copy()

print("== KD minus its own dense teacher (same seed), ECE transfer gap ==")
for name, d in [("CIFAR-10-C", c10c), ("STL-9", stl)]:
    diffs = []
    for s in [0, 1, 2]:
        kd = d[(d.method == "kd") & (d.seed == s)].gap.mean()
        te = d[(d.method == "dense") & (d.seed == s)].gap.mean()
        diffs.append(kd - te)
    print(f"  {name:11s}", [round(x, 4) for x in diffs])

print("\n== Source temperature, KD vs its own teacher, per seed ==")
t = df[df.domain == "cifar10_test"]
for s in [0, 1, 2]:
    kd_T = t[(t.method == "kd") & (t.seed == s)].source_T.values[0]
    te_T = t[(t.method == "dense") & (t.seed == s)].source_T.values[0]
    sc_T = t[(t.method == "scratch") & (t.seed == s)].source_T.values[0]
    print(f"  seed {s}: teacher={te_T:.3f}  kd={kd_T:.3f}  scratch={sc_T:.3f}")

print("\n== KD / scratch gap ratio ==")
for name, d in [("CIFAR-10-C", c10c), ("STL-9", stl)]:
    r = d[d.method == "kd"].gap.mean() / d[d.method == "scratch"].gap.mean()
    print(f"  {name:11s} {r:.3f}")

# Figure: gap vs severity, mean +/- std over seeds
c10c["sev"] = c10c.domain.str[-1].astype(int)
g = c10c.groupby(["method", "seed", "sev"]).gap.mean().reset_index()
labels = {
    "dense": "ResNet-18, dense (teacher)",
    "kd": "ResNet-10, distilled",
    "scratch": "ResNet-10, from scratch",
}
fig, ax = plt.subplots(figsize=(5.5, 3.8))
for m, lab in labels.items():
    s = g[g.method == m].groupby("sev").gap.agg(["mean", "std"])
    ax.errorbar(s.index, s["mean"], yerr=s["std"], marker="o", capsize=3, label=lab)
ax.set_xlabel("CIFAR-10-C corruption severity")
ax.set_ylabel("Transfer gap (source-fit ECE $-$ oracle-fit ECE)")
ax.set_xticks([1, 2, 3, 4, 5])
ax.legend(frameon=False)
fig.tight_layout()
plt.savefig("results/gap_vs_severity.pdf")
plt.savefig("results/gap_vs_severity.png", dpi=200)
print("\nSaved results/gap_vs_severity.pdf and .png")

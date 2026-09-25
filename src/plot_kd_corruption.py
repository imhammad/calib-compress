"""
Generates the per-corruption-type KD vs scratch transfer-gap figure
referenced in the paper (Figure: fig:kd-corruption).

Usage:
    python3 src/plot_kd_corruption.py
"""
import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv("results/calibration_results.csv")
shift = df[df.domain.str.startswith("cifar10c_")].copy()
shift["transfer_gap"] = shift["sourcets_ece"] - shift["oraclets_ece"]
shift["corruption"] = shift["domain"].str.extract(r"cifar10c_(.+)_sev\d")

kd = shift[shift.method == "kd"].groupby("corruption").transfer_gap.mean()
sc = shift[shift.method == "scratch"].groupby("corruption").transfer_gap.mean()
diff = (kd - sc).sort_values(ascending=True)  # ascending so largest bar ends up on top

fig, ax = plt.subplots(figsize=(6, 5))
colors = ["#d62728" if v > 0 else "#2ca02c" for v in diff.values]
ax.barh(diff.index, diff.values, color=colors)
ax.axvline(0, color="black", linewidth=0.8)
ax.set_xlabel("KD $-$ scratch transfer gap ($\\Delta$ ECE)")
ax.set_title("Knowledge distillation's calibration-transfer penalty\nby corruption type (CIFAR-10-C)")
plt.tight_layout()

plt.savefig("results/kd_corruption_gap.pdf")
plt.savefig("results/kd_corruption_gap.png", dpi=200)
print("Saved results/kd_corruption_gap.pdf and .png")
print()
print(diff.round(4).to_string())

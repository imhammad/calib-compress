"""
Recomputes every number used in main.tex from results/calibration_results.csv,
so the paper can be checked against the data before submission.

Usage (from the calib-compress folder):
    python3 src/check_numbers.py
"""
import pandas as pd

pd.set_option("display.width", 160)
df = pd.read_csv("results/calibration_results.csv")
df["gap"] = df.sourcets_ece - df.oraclets_ece
order = ["dense", "denseft", "unstr30", "unstr50", "unstr70", "unstr90",
         "str15", "str30", "int8", "kd", "scratch"]

c10 = df[df.domain == "cifar10_test"]
c10c = df[df.domain.str.startswith("cifar10c_")].copy()
stl = df[df.domain == "stl9"]


def per_seed(d, col):
    return d.groupby(["method", "seed"])[col].mean().unstack("seed")


print("== Table 2: main summary ==")
t = pd.DataFrame(index=order)
acc = per_seed(c10, "raw_accuracy") * 100
t["acc"] = acc.mean(axis=1).round(2).astype(str) + " +/- " + acc.std(axis=1).round(2).astype(str)
t["ece_raw"] = c10.groupby("method").raw_ece.mean().round(4)
t["ece_src"] = c10.groupby("method").sourcets_ece.mean().round(4)
t["T_src"] = c10.groupby("method").source_T.mean().round(2)
t["c10c_acc"] = (c10c.groupby("method").raw_accuracy.mean() * 100).round(1)
g = per_seed(c10c, "gap")
t["c10c_gap"] = g.mean(axis=1).round(4).astype(str) + " +/- " + g.std(axis=1).round(4).astype(str)
t["stl_gap"] = stl.groupby("method").gap.mean().round(4)
t["stl_acc"] = (stl.groupby("method").raw_accuracy.mean() * 100).round(1)
print(t.to_string())

print("\n== RQ1: raw / source-fit / oracle-fit ECE on CIFAR-10-C, and recovery fraction ==")
r = c10c.groupby("method")[["raw_ece", "sourcets_ece", "oraclets_ece"]].mean().loc[order]
r["recovered"] = ((r.raw_ece - r.sourcets_ece) / (r.raw_ece - r.oraclets_ece)).round(3)
print(r.round(4).to_string())

print("\n== Per-seed temperatures (teacher = dense, same seed) ==")
for s in [0, 1, 2]:
    row = {m: round(c10[(c10.method == m) & (c10.seed == s)].source_T.values[0], 3)
           for m in ["dense", "kd", "scratch"]}
    print(f"seed {s}: {row}")

print("\n== Paired per-seed gap differences ==")
for name, d in [("CIFAR-10-C", c10c), ("STL-9", stl)]:
    ps = per_seed(d, "gap")
    for a, b in [("kd", "scratch"), ("kd", "dense"), ("denseft", "dense")]:
        print(f"{name:11s} {a} - {b}:", (ps.loc[a] - ps.loc[b]).round(4).tolist())
    print(f"{name:11s} kd/scratch ratio: {ps.loc['kd'].mean() / ps.loc['scratch'].mean():.3f}")

print("\n== KD - scratch by severity ==")
c10c["sev"] = c10c.domain.str[-1].astype(int)
sv = c10c.groupby(["method", "sev"]).gap.mean().unstack()
print((sv.loc["kd"] - sv.loc["scratch"]).round(4).to_string())

print("\n== KD - scratch per corruption type ==")
c10c["corr"] = c10c.domain.str.extract(r"cifar10c_(.+)_sev\d", expand=False)
pc = c10c.groupby(["method", "corr"]).gap.mean().unstack()
d = (pc.loc["kd"] - pc.loc["scratch"]).sort_values(ascending=False)
print(d.round(4).to_string())
print("positive in", int((d > 0).sum()), "of", len(d))

print("\n== Raw ECE under shift, KD vs scratch ==")
print(c10c[c10c.method.isin(["kd", "scratch"])].groupby("method").raw_ece.mean().round(4).to_string())

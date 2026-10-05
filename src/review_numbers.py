import pandas as pd
pd.set_option("display.width", 140)
df = pd.read_csv("results/calibration_results.csv")
c10 = df[df.domain == "cifar10_test"]
c10c = df[df.domain.str.startswith("cifar10c_")].copy()
c10c["gap"] = c10c.sourcets_ece - c10c.oraclets_ece

print("== In-domain (CIFAR-10 test), mean/std over seeds ==")
print(c10.groupby("method")[["raw_accuracy", "raw_ece", "sourcets_ece", "source_T"]].agg(["mean", "std"]).round(4))

print("\n== CIFAR-10-C accuracy, mean over conditions and seeds ==")
print(c10c.groupby("method").raw_accuracy.mean().round(4))

print("\n== Transfer gap per method, mean/std over seeds ==")
per_seed = c10c.groupby(["method", "seed"]).gap.mean()
print(per_seed.groupby("method").agg(["mean", "std"]).round(4))

print("\n== Pruned minus denseft, per seed (negative = pruned better) ==")
ft = per_seed["denseft"]
for m in ["unstr30", "unstr50", "unstr70", "unstr90", "str15", "str30"]:
    print(f"{m:8s}", (per_seed[m] - ft).round(4).tolist())

print("\n== Transfer gap by severity ==")
c10c["sev"] = c10c.domain.str[-1].astype(int)
print(c10c[c10c.method.isin(["dense", "kd", "scratch"])].groupby(["method", "sev"]).gap.mean().unstack().round(4))

stl = df[df.domain == "stl9"].copy()
stl["gap"] = stl.sourcets_ece - stl.oraclets_ece
print("\n== STL-9 after frog masking: acc and gap ==")
print(stl.groupby("method")[["raw_accuracy", "gap"]].mean().round(4))
for a, b in [("kd", "scratch"), ("denseft", "dense")]:
    d = [stl[(stl.method == a) & (stl.seed == s)].gap.mean() - stl[(stl.method == b) & (stl.seed == s)].gap.mean() for s in [0, 1, 2]]
    print(f"{a} - {b}:", [round(x, 4) for x in d])

"""
Pooled cross-fitting (Paolo's refinement, 11 Nov 2026), for Phase 2 (TS) and
Phase 3 (all six calibrators) in one run.

Before (Phases 2 and 3): ECE was computed on each half (A scored with the
calibrator fit on B, B with the one fit on A) and the two half-ECEs were
averaged; the source calibrator was scored on the same halves.

Now: the out-of-fold predictions of both halves are put back together into
one prediction for every image of the domain, and ECE, adaptive ECE and
classwise ECE are computed ONCE on that full set. NLL and Brier are plain
per-image averages, so for equal halves the pooled value equals the average
of the halves. The source calibrator is scored on the same full set (as for
the absolute numbers). This is done for 5 random A/B splits (seeds 0-4, the
same splits as Phase 2), the 5 values are averaged per model, and summaries
report mean +- sd over the 3 trained seeds.

    oracle_pooled(split s) = metric( concat[ A calibrated with fit(B),
                                             B calibrated with fit(A) ] )
    gap_own   = source(full set) - mean over 5 splits of oracle_pooled (same calibrator)
    gap_tsor  = source(full set) - mean over 5 splits of TS oracle_pooled

The calibrators, their fitting code and the STL-9 handling are imported
unchanged from phase3_calibrators.py.

Checks (the script stops if one fails):
  0. inputs: Phase 2 CSV and Phase 3 CSV found by md5, split seed 0 equals
     results/phase2/split_seed0.npz, all logit files present.
  1. split seed 0: the TS half temperatures T_A, T_B equal Phase 2's, and
     the pooled NLL and Brier equal Phase 2's half-averaged ones (they must,
     since the halves are equal in size).
  2. source-fit numbers (full set) of all six calibrators equal Phase 3's.
  3. the pooled oracle has exactly one prediction per image (A and B cover
     the domain once).

Usage, from the calib-compress folder:
    python3 src/pooled_crossfit.py --quick     (3 checkpoints)
    python3 src/pooled_crossfit.py
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
from phase3_calibrators import (CAL, CALIBRATORS, FROG_IDX, FitFailed, add_frog,  # noqa: E402
                                cal_metrics, drop_frog, fit_calibrator, load_raw,
                                make_split, md5, remap9, stop)

OUT = "results/pooled"
P3_CSV = "results/phase3/calibration_results_phase3.csv"
P3_CSV_MD5 = "9e6358a63ee2eb00fa7adc2b66db59eb"     # Hammad's Phase 3 run, 11 Nov 2026
N_SPLITS = 5
QUICK = ("resnet10_kd_s0", "resnet10_scratch_s0", "resnet18_dense_s0")


def run_checkpoint(ckpt, domains, splits):
    warnings.filterwarnings("ignore")
    try:
        import torch
        torch.set_num_threads(1)
    except ImportError:
        pass
    zv, yv = load_raw(ckpt, "cifar10_val")
    src = {name: fit_calibrator(name, zv, yv)[0] for name in CALIBRATORS}
    rows, fails, halfT = [], [], []
    for d in domains:
        z, y = load_raw(ckpt, d)
        stl = d == "stl9"
        raw = z.copy()
        if stl:
            raw[:, FROG_IDX] = -1e4
        raw_full = cal_metrics(raw, y)
        for name in CALIBRATORS:
            if name == "ts":
                zc = src[name](raw)
            else:
                zc = src[name](z).copy()
                if stl:
                    zc[:, FROG_IDX] = -1e4
            full = cal_metrics(zc, y)
            row = {"checkpoint": ckpt, "domain": d, "calibrator": name, "n_samples": len(y),
                   "accuracy_raw": raw_full["accuracy"], "accuracy_src_full": full["accuracy"]}
            for m in CAL:
                row[f"raw_full_{m}"] = raw_full[m]
                row[f"src_full_{m}"] = full[m]
            per_split = []
            for s, sp in enumerate(splits):
                a, b = sp["stl9" if stl else "cifar10"]
                zo = np.full((len(y), z.shape[1]), np.nan)
                filled = np.zeros(len(y), dtype=int)
                try:
                    for fit_idx, ev_idx in [(b, a), (a, b)]:
                        if name == "ts":
                            f_apply, info = fit_calibrator(name, raw[fit_idx], y[fit_idx])
                            zo[ev_idx] = f_apply(raw[ev_idx])
                            if s == 0:
                                halfT.append({"checkpoint": ckpt, "domain": d,
                                              "fit_on": "B" if fit_idx is b else "A", "T": info["T"]})
                        elif stl:
                            f_apply, _ = fit_calibrator(name, drop_frog(z[fit_idx]), remap9(y[fit_idx]))
                            zo[ev_idx] = add_frog(f_apply(drop_frog(z[ev_idx])))
                        else:
                            f_apply, _ = fit_calibrator(name, z[fit_idx], y[fit_idx])
                            zo[ev_idx] = f_apply(z[ev_idx])
                        filled[ev_idx] += 1
                    if not (filled == 1).all():
                        stop(f"{ckpt} {d} split {s}: pooled oracle does not cover each image once")
                    pm = cal_metrics(zo, y)
                    # NLL and Brier as averages of the two halves (identical for equal halves)
                    ha = cal_metrics(zo[a], y[a])
                    hb = cal_metrics(zo[b], y[b])
                    pm["nll_halfavg"] = (ha["nll"] + hb["nll"]) / 2
                    pm["brier_halfavg"] = (ha["brier"] + hb["brier"]) / 2
                    per_split.append(pm)
                except (FitFailed, FloatingPointError) as e:
                    fails.append({"checkpoint": ckpt, "domain": d, "calibrator": name, "split": s,
                                  "error": str(e)[:200]})
                    per_split.append(None)
            ok = [x for x in per_split if x is not None]
            row["n_splits_ok"] = len(ok)
            for m in CAL + ["accuracy"]:
                vals = [x[m] for x in ok]
                row[f"xfit_pooled_{m}"] = float(np.mean(vals)) if vals else np.nan
                row[f"xfit_pooled_{m}_split_sd"] = float(np.std(vals, ddof=1)) if len(vals) > 1 else np.nan
                for s, x in enumerate(per_split):
                    row[f"xfit_pooled_{m}_split{s}"] = x[m] if x is not None else np.nan
            for m in ["nll", "brier"]:
                row[f"xfit_split0_{m}_halfavg"] = per_split[0][f"{m}_halfavg"] if per_split[0] else np.nan
            for m in CAL:
                row[f"gap_own_{m}"] = row[f"src_full_{m}"] - row[f"xfit_pooled_{m}"]
            rows.append(row)
    return rows, fails, halfT


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()
    t0 = time.time()

    print("== Step 0: inputs ==")
    if not os.path.isdir(p3.LOGITS):
        sys.exit("Run this from the calib-compress folder (need logits/ and results/).")
    try:
        from importlib.metadata import version
        import probmetrics.calibrators  # noqa: F401
        if version("probmetrics") != "1.3.0":
            stop(f"probmetrics version is {version('probmetrics')}, expected 1.3.0")
    except ImportError as e:
        stop(f"probmetrics could not be imported ({e})")
    p2_csv = f"{p3.P2_DIR}/calibration_results_phase2.csv"
    p2_split = f"{p3.P2_DIR}/split_seed0.npz"
    for path, want in [(p2_csv, p3.P2_CSV_MD5), (p2_split, p3.P2_SPLIT_MD5), (P3_CSV, P3_CSV_MD5)]:
        if not os.path.isfile(path):
            stop(f"{path} not found")
        if md5(path) != want:
            stop(f"{path} has md5 {md5(path)}, expected {want}")
    print("Phase 2 CSV, Phase 2 split and Phase 3 CSV found (md5s match)")
    p2 = pd.read_csv(p2_csv)
    p3r = pd.read_csv(P3_CSV)
    ckpts = sorted(p2.checkpoint.unique())
    if args.quick:
        ckpts = [c for c in ckpts if c in QUICK]
        print(f"(--quick: only {ckpts})")
    domains = sorted(p2.domain.unique())
    missing = [f"{c}__{d}" for c in ckpts for d in ["cifar10_val"] + domains
               if not os.path.exists(f"{p3.LOGITS}/{c}__{d}.npz")]
    if missing:
        stop(f"{len(missing)} logit files are missing, e.g. {missing[:3]}")
    y_c10 = load_raw(ckpts[0], "cifar10_test")[1]
    y_stl = load_raw(ckpts[0], "stl9")[1]
    splits = []
    for s in range(N_SPLITS):
        sp = {}
        for key, y in [("cifar10", y_c10), ("stl9", y_stl)]:
            a, b = make_split(y, s)
            if len(np.intersect1d(a, b)) or len(a) + len(b) != len(y) or len(a) != len(b):
                stop(f"split seed {s} {key}: halves overlap, miss images or differ in size")
            sp[key] = (a, b)
        splits.append(sp)
    saved = np.load(p2_split)
    if not (np.array_equal(splits[0]["cifar10"][0], saved["cifar10_A"])
            and np.array_equal(splits[0]["stl9"][0], saved["stl9_A"])):
        stop("split seed 0 does not equal results/phase2/split_seed0.npz")
    print(f"{len(ckpts)} checkpoints x {len(domains)} domains x {len(CALIBRATORS)} calibrators x "
          f"{N_SPLITS} splits (seed 0 = Phase 2's split; CIFAR-10 5000/5000, STL-9 3600/3600)")
    os.makedirs(OUT, exist_ok=True)

    print(f"\n== Step 1: pooled cross-fitted oracles, {args.workers} workers ==", flush=True)
    rows, fails, halfT = [], [], []
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=mp.get_context("spawn")) as ex:
        futs = {ex.submit(run_checkpoint, c, domains, splits): c for c in ckpts}
        for k, f in enumerate(as_completed(futs), 1):
            r, fa, ht = f.result()
            rows += r
            fails += fa
            halfT += ht
            print(f"  {k}/{len(ckpts)} {futs[f]} done  ({(time.time() - t0) / 60:.1f} min)", flush=True)
    res = pd.DataFrame(rows)

    print("\n== Step 2: checks ==")
    # check 1: Phase 2 half temperatures and NLL/Brier (split seed 0, TS)
    ht = pd.DataFrame(halfT).pivot_table(index=["checkpoint", "domain"], columns="fit_on", values="T")
    p2i = p2.set_index(["checkpoint", "domain"]).loc[ht.index]
    dT = max(((ht["A"] - p2i["T_A"]).abs() / p2i["T_A"]).max(), ((ht["B"] - p2i["T_B"]).abs() / p2i["T_B"]).max())
    ts = res[res.calibrator == "ts"].set_index(["checkpoint", "domain"])
    p2t = p2.set_index(["checkpoint", "domain"]).loc[ts.index]
    dN = max((ts["xfit_split0_nll_halfavg"] - p2t["xfit_nll"]).abs().max(),
             (ts["xfit_split0_brier_halfavg"] - p2t["xfit_brier"]).abs().max())
    dP = max((ts["xfit_pooled_nll_split0"] - ts["xfit_split0_nll_halfavg"]).abs().max(),
             (ts["xfit_pooled_brier_split0"] - ts["xfit_split0_brier_halfavg"]).abs().max())
    print(f"check 1, split seed 0 vs Phase 2: half temperatures worst rel diff {dT:.1e}; "
          f"half-averaged NLL/Brier worst diff {dN:.1e}; pooled minus half-averaged NLL/Brier {dP:.1e}")
    if dT > 1e-9 or dN > 1e-6 or dP > 1e-9:
        stop("split seed 0 does not reproduce Phase 2's half-set fits")
    print("CHECK 1 PASSED.")
    # check 2: source-fit numbers equal Phase 3
    mine = res.set_index(["checkpoint", "domain", "calibrator"])
    ref = p3r.set_index(["checkpoint", "domain", "calibrator"]).loc[mine.index]
    is_ts = mine.index.get_level_values("calibrator") == "ts"
    worst = {"ts": (0.0, None), "other": (0.0, None)}
    for col in ["accuracy_src_full"] + [f"src_full_{m}" for m in CAL] + [f"raw_full_{m}" for m in CAL]:
        dd = (mine[col] - ref[col]).abs() / np.maximum(1.0, ref[col].abs())
        for key, sel in [("ts", is_ts), ("other", ~is_ts)]:
            if sel.any() and dd[sel].max() > worst[key][0]:
                worst[key] = (dd[sel].max(), (dd[sel].idxmax(), col))
    print(f"check 2, source-fit full-set numbers vs Phase 3: TS worst diff {worst['ts'][0]:.1e}; "
          f"other calibrators worst diff {worst['other'][0]:.1e} {worst['other'][1] or ''}")
    # On the same Mac these should be ~0. The torch BFGS fits of vs/ms can differ
    # slightly between machines, hence the looser limit for them.
    if worst["ts"][0] > 1e-6 or worst["other"][0] > 5e-3:
        stop("source-fit numbers do not reproduce Phase 3")
    print("CHECK 2 PASSED.")
    print("check 3, each pooled oracle covers every image exactly once: yes (the script stops otherwise)")
    print(f"oracle fits that failed (left out of that split's average): {len(fails)}")
    if fails:
        pd.DataFrame(fails).to_csv(f"{OUT}/oracle_fit_failures.csv", index=False)
        print(pd.DataFrame(fails).groupby("calibrator").size().to_string())

    # gap vs the pooled TS oracle; aliases so phase3_calibrators.summarise() can be reused
    xts = ts[[f"xfit_pooled_{m}" for m in CAL]].rename(
        columns={f"xfit_pooled_{m}": f"xfit_ts_{m}" for m in CAL}).reset_index()
    res = res.merge(xts, on=["checkpoint", "domain"], how="left")
    for m in CAL:
        res[f"gap_tsor_{m}"] = res[f"src_full_{m}"] - res[f"xfit_ts_{m}"]
    res = p3.add_meta(res, p2)
    lead = ["checkpoint", "arch", "method", "seed", "domain", "calibrator"]
    res = res[lead + [c for c in res.columns if c not in lead]]
    res.drop(columns=["sev", "corr"]).to_csv(f"{OUT}/calibration_results_pooled.csv", index=False)
    print(f"wrote {OUT}/calibration_results_pooled.csv ({len(res)} rows)")

    alias = res.copy()
    for m in CAL:
        alias[f"xfit_{m}"] = alias[f"xfit_pooled_{m}"]
    alias["raw_half_ece"] = alias["raw_full_ece"]
    alias["src_half_ece"] = alias["src_full_ece"]
    summ = p3.summarise(alias, True)
    # robustness over splits: KD/scratch ratio of the TS gap for each split separately
    for s in range(N_SPLITS):
        alias_s = res[res.calibrator == "ts"].copy()
        alias_s["g"] = alias_s["src_full_ece"] - alias_s[f"xfit_pooled_ece_split{s}"]
        for setting, dsel in [("c10c", alias_s.domain.str.startswith("cifar10c_")), ("stl9", alias_s.domain == "stl9")]:
            ps = alias_s[dsel].groupby(["method", "seed"]).g.mean()
            if {"kd", "scratch"} <= set(ps.index.get_level_values(0)):
                summ.loc[len(summ)] = [setting, "gap_ts_ece_by_split", "ts", "kd_over_scratch", f"split{s}",
                                       ps["kd"].mean() / ps["scratch"].mean()]
    summ.to_csv(f"{OUT}/summary_pooled.csv", index=False)

    print("\n== Step 3: summary (pooled oracle, mean over 5 splits, then mean over 3 seeds) ==")
    pd.set_option("display.width", 200)
    pd.set_option("display.max_rows", 500)
    ff = lambda v: f"{v:.4f}"  # noqa: E731
    print("\nClean CIFAR-10 test, gap to own oracle (should be ~0, in-domain):")
    print(p3.pivot(summ, "c10test", "gap_own_ece").to_string(float_format=ff))
    for setting, label in [("c10c", "CIFAR-10-C (mean of 75 conditions)"), ("stl9", "STL-9")]:
        print(f"\n--- {label} ---")
        print("Gap to the pooled cross-fitted TS oracle:")
        print(p3.pivot(summ, setting, "gap_tsor_ece").to_string(float_format=ff))
        print("Gap to each calibrator's own pooled cross-fitted oracle:")
        print(p3.pivot(summ, setting, "gap_own_ece").to_string(float_format=ff))
        print("Fraction of the TS-oracle improvement recovered:")
        print(p3.pivot(summ, setting, "recovered_frac_of_ts_oracle", stat="all_seeds").to_string(float_format=ff))
        print("KD minus scratch, per seed (s0 / s1 / s2), seeds positive, and KD/scratch ratio:")
        for q in ["gap_tsor_ece", "gap_own_ece"]:
            s = summ[(summ.setting == setting) & (summ.quantity == q)]
            for cal in CALIBRATORS:
                v = s[(s.calibrator == cal) & (s.arm == "kd_minus_scratch")].set_index("stat").value
                rt = s[(s.calibrator == cal) & (s.arm == "kd_over_scratch")].value
                seeds = " / ".join(f"{v[k]:+.4f}" for k in sorted(v.index) if k.startswith("seed") and len(k) == 5)
                print(f"  {q:13s} {cal:4s}: {seeds}   positive {int(v.get('seeds_positive_of_3', -1))}/3"
                      f"   ratio {rt.iloc[0]:.3f}" if len(rt) else "")
        r = summ[(summ.setting == setting) & (summ.quantity == "gap_ts_ece_by_split")].value
        if len(r):
            print(f"  TS gap KD/scratch ratio over the 5 splits: {r.min():.3f} to {r.max():.3f}")
    print(f"\nALL DONE in {(time.time() - t0) / 60:.1f} min.")


if __name__ == "__main__":
    main()

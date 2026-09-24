"""
Phase 7: the real calibration analysis. Reads every dumped logit file,
computes raw + source-TS + oracle-TS metrics for every (checkpoint,
domain) pair, and writes results/calibration_results.csv -- the single
source of truth for every calibration number in the paper.

Usage:
    python3 src/analyze_calibration.py
"""
import glob
import os
import re

import numpy as np
import pandas as pd

from metrics import compute_all_metrics, fit_temperature, apply_temperature


def parse_filename(path):
    """logits/<checkpoint_name>__<domain_name>.npz -> (checkpoint_name, domain_name)"""
    base = os.path.basename(path).replace(".npz", "")
    checkpoint_name, domain_name = base.split("__", 1)
    return checkpoint_name, domain_name


def classify_checkpoint(name):
    """Extracts (arch, method, param, seed) from a checkpoint name for
    easy grouping/plotting later."""
    m = re.match(r"(resnet\d+)_([a-z0-9]+)_s(\d)$", name)
    arch, method, seed = m.group(1), m.group(2), int(m.group(3))
    return arch, method, seed


def main(logits_dir="logits"):
    logit_files = sorted(glob.glob(f"{logits_dir}/*.npz"))
    print(f"Found {len(logit_files)} logit files")

    # organize by checkpoint -> {domain: (logits, labels)}
    by_checkpoint = {}
    for f in logit_files:
        ckpt_name, domain_name = parse_filename(f)
        data = np.load(f)
        by_checkpoint.setdefault(ckpt_name, {})[domain_name] = (data["logits"], data["labels"])

    print(f"Found {len(by_checkpoint)} checkpoints")

    rows = []
    for ckpt_name, domains in sorted(by_checkpoint.items()):
        arch, method, seed = classify_checkpoint(ckpt_name)
        print(f"Processing {ckpt_name} ({len(domains)} domains)...")

        if "cifar10_val" not in domains:
            print(f"  WARNING: no cifar10_val for {ckpt_name}, skipping temperature fitting")
            source_T = None
        else:
            val_logits, val_labels = domains["cifar10_val"]
            source_T = fit_temperature(val_logits, val_labels)

        for domain_name, (logits, labels) in domains.items():
            if domain_name == "cifar10_val":
                continue  # val set itself is not a reported evaluation domain

            raw_metrics = compute_all_metrics(logits, labels)

            row = {
                "checkpoint": ckpt_name,
                "arch": arch,
                "method": method,
                "seed": seed,
                "domain": domain_name,
                "n_samples": len(labels),
                "source_T": source_T,
            }
            for k, v in raw_metrics.items():
                row[f"raw_{k}"] = v

            if source_T is not None:
                source_ts_logits = apply_temperature(logits, source_T)
                source_ts_metrics = compute_all_metrics(source_ts_logits, labels)
                for k, v in source_ts_metrics.items():
                    row[f"sourcets_{k}"] = v

            oracle_T = fit_temperature(logits, labels)
            oracle_ts_logits = apply_temperature(logits, oracle_T)
            oracle_ts_metrics = compute_all_metrics(oracle_ts_logits, labels)
            row["oracle_T"] = oracle_T
            for k, v in oracle_ts_metrics.items():
                row[f"oraclets_{k}"] = v

            rows.append(row)

    df = pd.DataFrame(rows)
    os.makedirs("results", exist_ok=True)
    df.to_csv("results/calibration_results.csv", index=False)
    print(f"\nDone. {len(df)} rows written to results/calibration_results.csv")
    print(f"Columns: {list(df.columns)}")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--logits-dir", default="logits")
    args = p.parse_args()
    main(logits_dir=args.logits_dir)

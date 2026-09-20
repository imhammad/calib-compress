"""
Dump raw logits + labels for every (checkpoint, domain) pair to
logits/<checkpoint_name>__<domain_name>.npz.

This is the ONE place every model ever gets run for evaluation from
here on. Every calibration metric (ECE, NLL, Brier, temperature
scaling) in Phase 7 reads these .npz files -- nothing downstream ever
re-runs a model. Get this right once; everything after is fast,
local, and re-runnable on the Mac.

Usage:
    python3 src/dump_logits.py --checkpoints resnet18_dense_s0 \
        --domains cifar10_test,cifar10c_gaussian_noise_sev3
    python3 src/dump_logits.py --checkpoints all --domains all
"""
import argparse
import os

import numpy as np
import torch
from torch.utils.data import DataLoader

from data import CIFAR10Test, CIFAR10C, STL9, STANDARD_CORRUPTIONS
from models import resnet18_cifar, resnet10_cifar
from prune_structured import rebuild_pruned_skeleton


# ---------------------------------------------------------------------
# Checkpoint manifest: every checkpoint we've trained, and how to load it.
# ---------------------------------------------------------------------

def build_manifest():
    m = []
    seeds = [0, 1, 2]

    for s in seeds:
        m.append({"name": f"resnet18_dense_s{s}", "kind": "plain", "arch": "resnet18",
                   "path": f"ckpt/resnet18_dense_s{s}.pt"})
        m.append({"name": f"resnet18_denseft_s{s}", "kind": "plain", "arch": "resnet18",
                   "path": f"ckpt/resnet18_denseft_s{s}.pt"})
        for sp in [30, 50, 70, 90]:
            m.append({"name": f"resnet18_unstr{sp}_s{s}", "kind": "plain", "arch": "resnet18",
                       "path": f"ckpt/resnet18_unstr{sp}_s{s}.pt"})
        for ratio_tag, ratio in [("15", 0.15), ("30", 0.30)]:
            m.append({"name": f"resnet18_str{ratio_tag}_s{s}", "kind": "structured", "arch": "resnet18",
                       "path": f"ckpt/resnet18_str{ratio_tag}_s{s}.pt",
                       "dense_ckpt": f"ckpt/resnet18_dense_s{s}.pt", "sparsity": ratio})
        m.append({"name": f"resnet10_kd_s{s}", "kind": "plain", "arch": "resnet10",
                   "path": f"ckpt/resnet10_kd_s{s}.pt"})
        m.append({"name": f"resnet10_scratch_s{s}", "kind": "plain", "arch": "resnet10",
                   "path": f"ckpt/resnet10_scratch_s{s}.pt"})
        m.append({"name": f"resnet18_int8_s{s}", "kind": "int8", "arch": "resnet18",
                   "path": f"ckpt/resnet18_int8_s{s}.pt"})

    return {entry["name"]: entry for entry in m}


ARCH_BUILDERS = {"resnet18": resnet18_cifar, "resnet10": resnet10_cifar}


def load_model(entry, device):
    """Returns (model, device_to_use_for_this_model) -- INT8 models are
    forced to CPU regardless of what device everything else runs on."""
    if entry["kind"] == "plain":
        model = ARCH_BUILDERS[entry["arch"]]().to(device)
        ckpt = torch.load(entry["path"], map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()
        return model, device

    if entry["kind"] == "structured":
        model = rebuild_pruned_skeleton(entry["dense_ckpt"], ARCH_BUILDERS[entry["arch"]],
                                          entry["sparsity"], device)
        ckpt = torch.load(entry["path"], map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()
        return model, device

    if entry["kind"] == "int8":
        # Quantized models only run on CPU; the checkpoint stores the
        # whole wrapped module object, not a state_dict.
        ckpt = torch.load(entry["path"], map_location="cpu")
        model = ckpt["model"]
        model.eval()
        return model, "cpu"

    raise ValueError(f"unknown checkpoint kind: {entry['kind']}")


# ---------------------------------------------------------------------
# Domain manifest
# ---------------------------------------------------------------------

def build_domain_list():
    domains = ["cifar10_test", "stl9"]
    for corruption in STANDARD_CORRUPTIONS:
        for severity in [1, 2, 3, 4, 5]:
            domains.append(f"cifar10c_{corruption}_sev{severity}")
    return domains


def get_domain_dataset(domain_name):
    if domain_name == "cifar10_test":
        return CIFAR10Test()
    if domain_name == "stl9":
        return STL9()
    if domain_name.startswith("cifar10c_"):
        rest = domain_name[len("cifar10c_"):]
        corruption, sev_str = rest.rsplit("_sev", 1)
        return CIFAR10C(corruption, severity=int(sev_str))
    raise ValueError(f"unknown domain: {domain_name}")


# ---------------------------------------------------------------------
# Core dump loop
# ---------------------------------------------------------------------

def dump_for_checkpoint(entry, domain_names, device, out_dir="logits", batch_size=256):
    model, model_device = load_model(entry, device)
    print(f"  loaded ({entry['kind']}, arch={entry['arch']}) on {model_device}")

    for domain_name in domain_names:
        out_path = f"{out_dir}/{entry['name']}__{domain_name}.npz"
        if os.path.exists(out_path):
            print(f"  [skip, exists] {domain_name}")
            continue

        ds = get_domain_dataset(domain_name)
        loader = DataLoader(ds, batch_size=batch_size, shuffle=False)

        all_logits, all_labels = [], []
        with torch.no_grad():
            for images, labels in loader:
                images = images.to(model_device)
                logits = model(images)
                all_logits.append(logits.cpu().numpy())
                all_labels.append(np.array(labels))

        logits_arr = np.concatenate(all_logits, axis=0).astype(np.float32)
        labels_arr = np.concatenate(all_labels, axis=0).astype(np.int64)

        np.savez(out_path, logits=logits_arr, labels=labels_arr)
        print(f"  [saved] {domain_name}: logits {logits_arr.shape}, labels {labels_arr.shape}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoints", required=True,
                    help="comma-separated checkpoint names, or 'all'")
    p.add_argument("--domains", required=True,
                    help="comma-separated domain names, or 'all'")
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Default device: {device}")

    os.makedirs("logits", exist_ok=True)

    manifest = build_manifest()
    all_domains = build_domain_list()
    print(f"Full manifest: {len(manifest)} checkpoints, {len(all_domains)} domains")

    if args.checkpoints == "all":
        ckpt_names = list(manifest.keys())
    else:
        ckpt_names = args.checkpoints.split(",")

    if args.domains == "all":
        domain_names = all_domains
    else:
        domain_names = args.domains.split(",")

    for name in ckpt_names:
        if name not in manifest:
            print(f"WARNING: '{name}' not in manifest, skipping")
            continue
        print(f"\n=== {name} ===")
        dump_for_checkpoint(manifest[name], domain_names, device)

    print("\nDone.")


if __name__ == "__main__":
    main()

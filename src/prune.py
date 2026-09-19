"""
Unstructured global magnitude pruning for the calibration-under-
compression project.

Loads a trained dense checkpoint, prunes the requested fraction of
weights (by magnitude, globally across all Conv2d/Linear layers, not
per-layer), bakes the resulting zeros in permanently, then fine-tunes
using the EXACT SAME recovery schedule as Phase 2's fine-tuning
control (src/finetune.py's fine_tune() function). This is what makes
"pruned vs. Phase-2-control" a fair, matched comparison rather than
"pruned vs. never-fine-tuned-dense".

Usage:
    python3 src/prune.py --start-ckpt ckpt/resnet18_dense_s0.pt \
        --arch resnet18 --seed 0 --sparsity 0.5 --epochs 20
"""
import argparse
import json

import torch
import torch.nn as nn
import torch.nn.utils.prune as prune_utils

from models import resnet18_cifar, resnet10_cifar, mobilenetv2_cifar
from finetune import fine_tune, get_device

ARCH_BUILDERS = {
    "resnet18": resnet18_cifar,
    "resnet10": resnet10_cifar,
    "mobilenetv2": mobilenetv2_cifar,
}


def apply_global_unstructured_pruning(model, sparsity):
    """
    Prunes `sparsity` fraction of weights globally across every
    Conv2d and Linear layer's .weight, by L1 magnitude. Zeros are
    baked in permanently (prune.remove) so the resulting state_dict
    is a normal, clean state_dict -- no pruning-specific wrapper
    objects, loadable by the plain architecture like any other
    checkpoint.
    """
    prunable = [
        (module, "weight")
        for module in model.modules()
        if isinstance(module, (nn.Conv2d, nn.Linear))
    ]

    prune_utils.global_unstructured(
        prunable,
        pruning_method=prune_utils.L1Unstructured,
        amount=sparsity,
    )

    # bake the zeros in permanently, remove the reparametrization
    for module, name in prunable:
        prune_utils.remove(module, name)

    return model


def measure_sparsity(model):
    """Returns the actual fraction of zeroed weights across prunable layers."""
    total, zeros = 0, 0
    for module in model.modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            w = module.weight.data
            total += w.numel()
            zeros += (w == 0).sum().item()
    return zeros / total


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--start-ckpt", required=True)
    p.add_argument("--arch", required=True, choices=ARCH_BUILDERS.keys())
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--sparsity", type=float, required=True,
                    help="fraction of weights to zero out, e.g. 0.5 for 50%")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--lr", type=float, default=0.01)
    p.add_argument("--max-train-images", type=int, default=None,
                    help="If set, use only this many training images (smoke tests only)")
    args = p.parse_args()

    torch.manual_seed(args.seed)
    device = get_device()
    sparsity_tag = f"unstr{int(args.sparsity * 100)}"
    run_name = f"{args.arch}_{sparsity_tag}_s{args.seed}"

    print(f"Run: {run_name}")
    print(f"Device: {device}")
    print(f"Starting from: {args.start_ckpt}")
    print(f"Target sparsity: {args.sparsity:.0%}")

    model = ARCH_BUILDERS[args.arch]().to(device)
    start_ckpt = torch.load(args.start_ckpt, map_location=device)
    model.load_state_dict(start_ckpt["model_state_dict"])
    print(f"Loaded weights from epoch {start_ckpt['epoch']} of the source checkpoint")

    model = apply_global_unstructured_pruning(model, args.sparsity)
    achieved = measure_sparsity(model)
    print(f"Achieved sparsity: {achieved:.4f} (target was {args.sparsity:.4f})")

    # sanity check: global L1 pruning should hit the target almost exactly
    assert abs(achieved - args.sparsity) < 0.01, \
        f"Sparsity mismatch! Wanted {args.sparsity}, got {achieved}"

    # measure accuracy immediately after pruning, BEFORE any recovery
    # training -- this is the real check that pruning actually did
    # something, uncontaminated by the fine-tuning that follows
    from torch.utils.data import DataLoader
    from data import CIFAR10Subset, EVAL_TRANSFORM
    val_ds = CIFAR10Subset("splits/cifar10_val_idx.npy", train=False, transform=EVAL_TRANSFORM)
    val_loader = DataLoader(val_ds, batch_size=256, shuffle=False)
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            preds = model(images).argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += images.size(0)
    pre_recovery_val_acc = correct / total
    print(f"Val acc immediately after pruning, BEFORE recovery: {pre_recovery_val_acc:.4f}")

    print(f"\nFine-tuning for {args.epochs} epochs (recovery)...")
    history = fine_tune(model, run_name, device, epochs=args.epochs, lr=args.lr,
                         max_train_images=args.max_train_images)

    with open(f"results/{run_name}_train_log.json", "w") as f:
        json.dump({"achieved_sparsity": achieved, "pre_recovery_val_acc": pre_recovery_val_acc, "history": history}, f, indent=2)

    print(f"\nDone. Checkpoint: ckpt/{run_name}.pt")
    print(f"Log: results/{run_name}_train_log.json")


if __name__ == "__main__":
    main()

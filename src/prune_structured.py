"""
Structured (filter/channel) pruning for the calibration-under-
compression project, using torch-pruning to correctly propagate
pruning through ResNet's residual connections.

Unlike unstructured pruning (src/prune.py), this ACTUALLY removes
channels -- the resulting model has fewer real parameters and is
genuinely smaller/faster on hardware, not just sparser. This is also
what makes it fragile: pruning a channel from conv2 in a BasicBlock
requires pruning the matching channel from the shortcut path too, or
the residual addition breaks. torch-pruning's DependencyGraph handles
this automatically via its "importance + group" pruning API.

Usage:
    python3 src/prune_structured.py --start-ckpt ckpt/resnet18_dense_s0.pt \
        --arch resnet18 --seed 0 --sparsity 0.3 --epochs 20
"""
import argparse
import json

import torch
import torch_pruning as tp

from models import resnet18_cifar
from finetune import fine_tune, get_device

# Structured pruning is only wired up for resnet18 in this script --
# MobileNetV2's depthwise-separable structure needs different handling
# and is out of scope for this arm.
ARCH_BUILDERS = {
    "resnet18": resnet18_cifar,
}


def apply_structured_pruning(model, sparsity, example_input):
    """
    Prunes `sparsity` fraction of channels globally, using torch-pruning's
    MagnitudePruner with a dependency graph built from a real forward
    pass (example_input). This ensures every layer coupled to a pruned
    channel -- including residual shortcuts -- gets pruned consistently.
    """
    imp = tp.importance.MagnitudeImportance(p=1)  # L1 norm, same criterion as unstructured

    # The final classifier layer (fc) must never be pruned -- its
    # output dimension is fixed at 10 (the number of classes).
    ignored_layers = [model.fc]

    pruner = tp.pruner.MagnitudePruner(
        model,
        example_inputs=example_input,
        importance=imp,
        pruning_ratio=sparsity,
        ignored_layers=ignored_layers,
    )
    pruner.step()
    return model


def rebuild_pruned_skeleton(dense_ckpt_path, arch_builder, sparsity, device):
    """
    Reconstructs a correctly-shaped pruned model skeleton by re-running
    the same deterministic structured pruning starting from the same
    dense checkpoint. Structured pruning changes real layer shapes, so
    a plain freshly-built full-size model can not load_state_dict() a
    pruned checkpoint directly -- this rebuilds the right shape first.
    Returns an uninitialized-weights model with the CORRECT shapes,
    ready to have the real trained state_dict loaded into it.
    """
    skeleton = arch_builder().to(device)
    dense_ckpt = torch.load(dense_ckpt_path, map_location=device)
    skeleton.load_state_dict(dense_ckpt["model_state_dict"])
    example_input = torch.randn(1, 3, 32, 32).to(device)
    skeleton = apply_structured_pruning(skeleton, sparsity, example_input)
    return skeleton


def count_params(model):
    return sum(p.numel() for p in model.parameters())


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--start-ckpt", required=True)
    p.add_argument("--arch", required=True, choices=ARCH_BUILDERS.keys())
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--sparsity", type=float, required=True,
                    help="fraction of channels to remove, e.g. 0.3 for 30%")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--lr", type=float, default=0.01)
    p.add_argument("--max-train-images", type=int, default=None)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    device = get_device()
    sparsity_tag = f"str{int(args.sparsity * 100)}"
    run_name = f"{args.arch}_{sparsity_tag}_s{args.seed}"

    print(f"Run: {run_name}")
    print(f"Device: {device}")
    print(f"Starting from: {args.start_ckpt}")
    print(f"Target channel sparsity: {args.sparsity:.0%}")

    model = ARCH_BUILDERS[args.arch]().to(device)
    start_ckpt = torch.load(args.start_ckpt, map_location=device)
    model.load_state_dict(start_ckpt["model_state_dict"])
    print(f"Loaded weights from epoch {start_ckpt['epoch']} of the source checkpoint")

    params_before = count_params(model)
    print(f"Parameters before pruning: {params_before:,}")

    example_input = torch.randn(1, 3, 32, 32).to(device)
    model = apply_structured_pruning(model, args.sparsity, example_input)

    params_after = count_params(model)
    actual_reduction = 1 - (params_after / params_before)
    print(f"Parameters after pruning: {params_after:,}")
    print(f"Actual parameter reduction: {actual_reduction:.4f} (target was {args.sparsity:.4f})")

    # Confirm the model still produces correctly-shaped output and
    # doesn't crash -- this is the real proof the dependency graph
    # was resolved correctly (shape mismatches would error here)
    with torch.no_grad():
        test_out = model(example_input)
    assert test_out.shape == (1, 10), f"Broken output shape: {test_out.shape}"
    print(f"Forward pass check OK, output shape: {tuple(test_out.shape)}")

    # measure accuracy immediately after pruning, before any recovery
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
        json.dump({
            "params_before": params_before,
            "params_after": params_after,
            "actual_param_reduction": actual_reduction,
            "pre_recovery_val_acc": pre_recovery_val_acc,
            "history": history,
        }, f, indent=2)

    print(f"\nDone. Checkpoint: ckpt/{run_name}.pt")
    print(f"Log: results/{run_name}_train_log.json")


if __name__ == "__main__":
    main()

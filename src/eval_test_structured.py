"""
Evaluate a STRUCTURED-pruned checkpoint on the real CIFAR-10 test set.

Structured pruning changes real layer shapes, so we can't just build
a plain resnet18_cifar() and load_state_dict() -- the shapes won't
match. Instead we deterministically rebuild the same pruned skeleton
by re-pruning from the same dense checkpoint (see
rebuild_pruned_skeleton in prune_structured.py), then load the real
trained weights into that correctly-shaped skeleton.
"""
import argparse
import torch
from torch.utils.data import DataLoader

from data import CIFAR10Test
from prune_structured import rebuild_pruned_skeleton
from models import resnet18_cifar


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True, help="the trained pruned checkpoint to evaluate")
    p.add_argument("--dense-ckpt", required=True, help="the ORIGINAL dense checkpoint this was pruned from")
    p.add_argument("--sparsity", type=float, required=True, help="the channel ratio used when pruning")
    args = p.parse_args()

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Device: {device}")

    model = rebuild_pruned_skeleton(args.dense_ckpt, resnet18_cifar, args.sparsity, device)

    trained_ckpt = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(trained_ckpt["model_state_dict"])
    model.eval()
    print(f"Loaded checkpoint from epoch {trained_ckpt['epoch']}")

    test_ds = CIFAR10Test()
    loader = DataLoader(test_ds, batch_size=256, shuffle=False)

    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            logits = model(images)
            correct += (logits.argmax(dim=1) == labels).sum().item()
            total += images.size(0)

    acc = correct / total
    print(f"\nTest accuracy on real CIFAR-10 test set (10,000 images): {acc:.4f} ({acc*100:.2f}%)")


if __name__ == "__main__":
    main()

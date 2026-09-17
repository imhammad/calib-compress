"""
Fine-tuning script for the calibration-under-compression project.

Loads an already-trained checkpoint and continues training for a
short recovery schedule (default 20 epochs, low LR). Used two ways
in this project:
  1. Phase 2 control: fine-tune the DENSE model with no compression
     applied, to isolate "extra training" from "compression" as a
     cause of any calibration change.
  2. Phase 3: fine-tune a model AFTER pruning, using the identical
     schedule, so pruned-vs-Phase-2-control is an apples-to-apples
     comparison.

This script deliberately does NOT prune anything itself -- pruning
(when we get to Phase 3) will be applied to the loaded checkpoint
BEFORE calling into the same fine-tune loop, via a separate script
that imports the fine_tune() function below. For now (Phase 2) we
call it with no modification, which is exactly the control we want.

Usage:
    python3 src/finetune.py --start-ckpt ckpt/resnet18_dense_s0.pt \
        --arch resnet18 --seed 0 --epochs 20 --tag denseft
"""
import argparse
import json
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from data import CIFAR10Subset, TRAIN_TRANSFORM, EVAL_TRANSFORM
from models import resnet18_cifar, resnet10_cifar, mobilenetv2_cifar

ARCH_BUILDERS = {
    "resnet18": resnet18_cifar,
    "resnet10": resnet10_cifar,
    "mobilenetv2": mobilenetv2_cifar,
}


def get_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def evaluate(model, loader, device, criterion):
    model.eval()
    total_loss, total_correct, total_n = 0.0, 0, 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            logits = model(images)
            loss = criterion(logits, labels)
            total_loss += loss.item() * images.size(0)
            total_correct += (logits.argmax(dim=1) == labels).sum().item()
            total_n += images.size(0)
    return total_loss / total_n, total_correct / total_n


def fine_tune(model, run_name, device, epochs=20, lr=0.01,
              batch_size=128, max_train_images=None):
    """
    Fine-tunes `model` in place for `epochs` epochs using a short
    cosine schedule from `lr` down to 0. Checkpoints every epoch to
    ckpt/<run_name>.pt. Returns the training history (list of dicts).
    """
    train_ds = CIFAR10Subset("splits/cifar10_train_idx.npy", train=True, transform=TRAIN_TRANSFORM)
    val_ds = CIFAR10Subset("splits/cifar10_val_idx.npy", train=False, transform=EVAL_TRANSFORM)

    if max_train_images is not None:
        train_ds.images = train_ds.images[:max_train_images]
        train_ds.labels = train_ds.labels[:max_train_images]
        print(f"[smoke test] restricting train set to {len(train_ds)} images")

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=256, shuffle=False, num_workers=0)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9,
                                 nesterov=True, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    history = []
    for epoch in range(epochs):
        t0 = time.time()
        model.train()
        running_loss, running_correct, running_n = 0.0, 0, 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            running_correct += (logits.argmax(dim=1) == labels).sum().item()
            running_n += images.size(0)

        scheduler.step()
        train_loss = running_loss / running_n
        train_acc = running_correct / running_n
        val_loss, val_acc = evaluate(model, val_loader, device, criterion)
        elapsed = time.time() - t0

        print(f"[ft] epoch {epoch+1:3d}/{epochs} | "
              f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
              f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} | "
              f"{elapsed:.1f}s | lr={scheduler.get_last_lr()[0]:.5f}")

        history.append({
            "epoch": epoch + 1, "train_loss": train_loss, "train_acc": train_acc,
            "val_loss": val_loss, "val_acc": val_acc, "elapsed_s": elapsed,
        })

        torch.save({
            "epoch": epoch + 1,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
        }, f"ckpt/{run_name}.pt")

    return history


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--start-ckpt", required=True, help="checkpoint to load and continue training from")
    p.add_argument("--arch", required=True, choices=ARCH_BUILDERS.keys())
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--lr", type=float, default=0.01)
    p.add_argument("--tag", default="denseft")
    p.add_argument("--max-train-images", type=int, default=None,
                    help="If set, use only this many training images (smoke tests only)")
    args = p.parse_args()

    torch.manual_seed(args.seed)
    device = get_device()
    run_name = f"{args.arch}_{args.tag}_s{args.seed}"
    print(f"Run: {run_name}")
    print(f"Device: {device}")
    print(f"Starting from: {args.start_ckpt}")

    model = ARCH_BUILDERS[args.arch]().to(device)
    start_ckpt = torch.load(args.start_ckpt, map_location=device)
    model.load_state_dict(start_ckpt["model_state_dict"])
    print(f"Loaded weights from epoch {start_ckpt['epoch']} of the source checkpoint")

    history = fine_tune(model, run_name, device, epochs=args.epochs, lr=args.lr,
                         max_train_images=args.max_train_images)

    with open(f"results/{run_name}_train_log.json", "w") as f:
        json.dump(history, f, indent=2)

    print(f"\nDone. Checkpoint: ckpt/{run_name}.pt")
    print(f"Log: results/{run_name}_train_log.json")


if __name__ == "__main__":
    main()

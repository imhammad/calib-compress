"""
Training loop for the calibration-under-compression project.

Usage:
    python3 src/train.py --arch resnet18 --seed 0 --epochs 100 --tag dense

Checkpoints every epoch to ckpt/<run_name>.pt (overwritten each epoch,
so a crash loses at most one epoch of progress, not the whole run).
Writes final metrics to results/<run_name>_train_log.json.
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


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


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


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--arch", required=True, choices=ARCH_BUILDERS.keys())
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--epochs", type=int, required=True)
    p.add_argument("--tag", default="dense")  # e.g. "dense", "denseft"
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=0.1)
    p.add_argument("--max-train-images", type=int, default=None,
                    help="If set, use only this many training images (smoke tests only)")
    args = p.parse_args()

    run_name = f"{args.arch}_{args.tag}_s{args.seed}"
    device = get_device()
    print(f"Run: {run_name}")
    print(f"Device: {device}")
    set_seed(args.seed)

    train_ds = CIFAR10Subset("splits/cifar10_train_idx.npy", train=True, transform=TRAIN_TRANSFORM)
    val_ds = CIFAR10Subset("splits/cifar10_val_idx.npy", train=False, transform=EVAL_TRANSFORM)

    if args.max_train_images is not None:
        train_ds.images = train_ds.images[:args.max_train_images]
        train_ds.labels = train_ds.labels[:args.max_train_images]
        print(f"[smoke test] restricting train set to {len(train_ds)} images")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=256, shuffle=False, num_workers=0)

    model = ARCH_BUILDERS[args.arch]().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9,
                                 nesterov=True, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    history = []
    for epoch in range(args.epochs):
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

        print(f"epoch {epoch+1:3d}/{args.epochs} | "
              f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
              f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} | "
              f"{elapsed:.1f}s | lr={scheduler.get_last_lr()[0]:.5f}")

        history.append({
            "epoch": epoch + 1, "train_loss": train_loss, "train_acc": train_acc,
            "val_loss": val_loss, "val_acc": val_acc, "elapsed_s": elapsed,
        })

        # checkpoint every epoch -- crash-safe, overwrites previous epoch's file
        torch.save({
            "epoch": epoch + 1,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "args": vars(args),
        }, f"ckpt/{run_name}.pt")

    with open(f"results/{run_name}_train_log.json", "w") as f:
        json.dump(history, f, indent=2)

    print(f"\nDone. Checkpoint: ckpt/{run_name}.pt")
    print(f"Log: results/{run_name}_train_log.json")


if __name__ == "__main__":
    main()

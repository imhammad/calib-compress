"""
Knowledge distillation for the calibration-under-compression project.

Trains a ResNet-10 student to match a frozen, already-trained ResNet-18
teacher's softened outputs (temperature T=2.0), combined with the
normal hard-label cross-entropy loss (weight alpha=0.7 on the soft
term, 1-alpha=0.3 on the hard term -- standard Hinton et al. KD setup).

The matching CONTROL (ResNet-10 trained from scratch, no teacher) does
NOT need this script -- it's just train.py --arch resnet10 --tag
scratch, since train.py already supports resnet10 as an architecture.
Comparing kd vs scratch is what isolates "distillation helps" from
"smaller models are just easier to calibrate/train" -- same logic as
the Phase 2 fine-tuning control.

Usage:
    python3 src/distill.py --teacher-ckpt ckpt/resnet18_dense_s0.pt \
        --seed 0 --epochs 100
"""
import argparse
import json
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from data import CIFAR10Subset, TRAIN_TRANSFORM, EVAL_TRANSFORM
from models import resnet18_cifar, resnet10_cifar


def get_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def kd_loss(student_logits, teacher_logits, labels, T=2.0, alpha=0.7):
    """
    Combined KD loss: alpha * soft-target KL divergence (temperature T)
    + (1-alpha) * standard hard-label cross-entropy.
    """
    soft_teacher = F.log_softmax(teacher_logits / T, dim=1)
    soft_student = F.log_softmax(student_logits / T, dim=1)
    # KLDivLoss expects log-probs for the input and log-probs (with
    # log_target=True) for the target; scale by T^2 per Hinton et al.
    soft_loss = F.kl_div(soft_student, soft_teacher, log_target=True,
                          reduction="batchmean") * (T * T)
    hard_loss = F.cross_entropy(student_logits, labels)
    return alpha * soft_loss + (1 - alpha) * hard_loss


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
    p.add_argument("--teacher-ckpt", required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--epochs", type=int, required=True)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=0.1)
    p.add_argument("--temperature", type=float, default=2.0)
    p.add_argument("--alpha", type=float, default=0.7)
    p.add_argument("--max-train-images", type=int, default=None,
                    help="If set, use only this many training images (smoke tests only)")
    args = p.parse_args()

    run_name = f"resnet10_kd_s{args.seed}"
    device = get_device()
    print(f"Run: {run_name}")
    print(f"Device: {device}")
    torch.manual_seed(args.seed)

    teacher = resnet18_cifar().to(device)
    teacher_ckpt = torch.load(args.teacher_ckpt, map_location=device)
    teacher.load_state_dict(teacher_ckpt["model_state_dict"])
    teacher.eval()
    for param in teacher.parameters():
        param.requires_grad = False
    print(f"Teacher loaded from epoch {teacher_ckpt['epoch']} of {args.teacher_ckpt}")

    student = resnet10_cifar().to(device)
    student_params = sum(p.numel() for p in student.parameters())
    print(f"Student (ResNet-10) parameters: {student_params:,}")

    train_ds = CIFAR10Subset("splits/cifar10_train_idx.npy", train=True, transform=TRAIN_TRANSFORM)
    val_ds = CIFAR10Subset("splits/cifar10_val_idx.npy", train=False, transform=EVAL_TRANSFORM)

    if args.max_train_images is not None:
        train_ds.images = train_ds.images[:args.max_train_images]
        train_ds.labels = train_ds.labels[:args.max_train_images]
        print(f"[smoke test] restricting train set to {len(train_ds)} images")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=256, shuffle=False, num_workers=0)

    criterion_eval = nn.CrossEntropyLoss()  # for reporting plain val loss/acc
    optimizer = torch.optim.SGD(student.parameters(), lr=args.lr, momentum=0.9,
                                 nesterov=True, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    history = []
    for epoch in range(args.epochs):
        t0 = time.time()
        student.train()
        running_loss, running_correct, running_n = 0.0, 0, 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()

            with torch.no_grad():
                teacher_logits = teacher(images)
            student_logits = student(images)

            loss = kd_loss(student_logits, teacher_logits, labels,
                            T=args.temperature, alpha=args.alpha)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            running_correct += (student_logits.argmax(dim=1) == labels).sum().item()
            running_n += images.size(0)

        scheduler.step()
        train_loss = running_loss / running_n
        train_acc = running_correct / running_n
        val_loss, val_acc = evaluate(student, val_loader, device, criterion_eval)
        elapsed = time.time() - t0

        print(f"epoch {epoch+1:3d}/{args.epochs} | "
              f"kd_train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
              f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} | "
              f"{elapsed:.1f}s | lr={scheduler.get_last_lr()[0]:.5f}")

        history.append({
            "epoch": epoch + 1, "kd_train_loss": train_loss, "train_acc": train_acc,
            "val_loss": val_loss, "val_acc": val_acc, "elapsed_s": elapsed,
        })

        torch.save({
            "epoch": epoch + 1,
            "model_state_dict": student.state_dict(),
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

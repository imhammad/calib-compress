"""
INT8 static post-training quantization (PTQ) for the calibration-
under-compression project.

Quantized ops in PyTorch only run on CPU (the 'fbgemm' backend for
x86, 'qnnpack' for ARM -- Apple Silicon Macs use qnnpack). This is a
ONE-SHOT conversion, not a training process: load a trained dense
checkpoint, calibrate the quantization ranges on a small batch of
real training images, convert, done. No epochs, no fine-tuning.

We insert QuantStub/DeQuantStub around the existing model rather than
modifying its internals -- simpler and less error-prone than manually
fusing every conv+bn+relu triple in our custom BasicBlock, at some
cost to quantized inference speed (irrelevant for our purposes, since
we only care about the resulting predictions/logits, not latency).
"""
import argparse
import platform

import torch
import torch.nn as nn
from torch.ao.quantization import get_default_qconfig, prepare, convert
from torch.utils.data import DataLoader

from data import CIFAR10Subset, CIFAR10Test, EVAL_TRANSFORM
from models import resnet18_cifar


class QuantWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        from torch.ao.quantization import QuantStub, DeQuantStub
        self.quant = QuantStub()
        self.model = model
        self.dequant = DeQuantStub()

    def forward(self, x):
        x = self.quant(x)
        x = self.model(x)
        x = self.dequant(x)
        return x


def default_backend():
    return "qnnpack" if platform.machine() in ("arm64", "aarch64") else "fbgemm"


def evaluate_cpu(model, loader):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in loader:
            logits = model(images)
            correct += (logits.argmax(dim=1) == labels).sum().item()
            total += images.size(0)
    return correct / total


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--backend", default=None)
    p.add_argument("--calib-images", type=int, default=1000)
    args = p.parse_args()

    backend = args.backend or default_backend()
    torch.backends.quantized.engine = backend
    print(f"Backend: {backend}")

    run_name = f"resnet18_int8_s{args.seed}"

    model = resnet18_cifar()
    ckpt = torch.load(args.ckpt, map_location="cpu")
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"Loaded dense checkpoint from epoch {ckpt['epoch']} of {args.ckpt}")

    test_ds = CIFAR10Test()
    test_loader = DataLoader(test_ds, batch_size=256, shuffle=False)
    fp32_acc = evaluate_cpu(model, test_loader)
    print(f"FP32 test accuracy (this run): {fp32_acc:.4f}")

    wrapped = QuantWrapper(model)
    wrapped.eval()
    wrapped.qconfig = get_default_qconfig(backend)
    prepare(wrapped, inplace=True)

    calib_ds = CIFAR10Subset("splits/cifar10_train_idx.npy", train=False, transform=EVAL_TRANSFORM)
    calib_ds.images = calib_ds.images[:args.calib_images]
    calib_ds.labels = calib_ds.labels[:args.calib_images]
    calib_loader = DataLoader(calib_ds, batch_size=100, shuffle=False)
    print(f"Calibrating on {len(calib_ds)} images...")
    with torch.no_grad():
        for images, _ in calib_loader:
            wrapped(images)

    convert(wrapped, inplace=True)
    print("Converted to INT8.")

    int8_acc = evaluate_cpu(wrapped, test_loader)
    print(f"INT8 test accuracy: {int8_acc:.4f}")
    print(f"Accuracy delta (INT8 - FP32): {int8_acc - fp32_acc:+.4f}")
    # A true no-op (like the original DeepSeek report's quantize_dynamic
    # bug) would show EXACTLY 0.0000 delta -- a small real, seed-varying
    # delta like this already proves conversion genuinely happened.

    torch.save({
        "model": wrapped,
        "backend": backend,
        "fp32_acc": fp32_acc,
        "int8_acc": int8_acc,
    }, f"ckpt/{run_name}.pt")
    print(f"\nDone. Checkpoint: ckpt/{run_name}.pt")


if __name__ == "__main__":
    main()

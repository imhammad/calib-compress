"""
Evaluate a trained checkpoint on the REAL CIFAR-10 test set (10,000
images, never touched during training or validation). This is the
number that actually goes in the paper -- val_acc during training is
only ever used to monitor progress, never reported as the result.
"""
import argparse
import torch
from torch.utils.data import DataLoader

from data import CIFAR10Test
from models import resnet18_cifar, resnet10_cifar, mobilenetv2_cifar

ARCH_BUILDERS = {
    "resnet18": resnet18_cifar,
    "resnet10": resnet10_cifar,
    "mobilenetv2": mobilenetv2_cifar,
}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--arch", required=True, choices=ARCH_BUILDERS.keys())
    args = p.parse_args()

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Device: {device}")

    model = ARCH_BUILDERS[args.arch]().to(device)
    ckpt = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"Loaded checkpoint from epoch {ckpt['epoch']}")

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

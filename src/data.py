"""
Shared data loading for the calibration-under-compression project.

Every loader here returns a torch.utils.data.Dataset that yields
(image_tensor, label_int). Normalization uses the standard CIFAR-10
per-channel mean/std everywhere, including for CIFAR-10-C and STL-10,
so a model sees the same input statistics regardless of which domain
it's being evaluated on.
"""
import numpy as np
import torch
from torch.utils.data import Dataset
from datasets import load_dataset
import torchvision.transforms as T

CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR_STD = (0.2470, 0.2435, 0.2616)

TRAIN_TRANSFORM = T.Compose([
    T.RandomCrop(32, padding=4),
    T.RandomHorizontalFlip(),
    T.ToTensor(),
    T.Normalize(CIFAR_MEAN, CIFAR_STD),
])

EVAL_TRANSFORM = T.Compose([
    T.ToTensor(),
    T.Normalize(CIFAR_MEAN, CIFAR_STD),
])


class CIFAR10Subset(Dataset):
    """CIFAR-10 train split, restricted to a fixed set of indices."""

    def __init__(self, indices_path, train=True, transform=None):
        ds = load_dataset("uoft-cs/cifar10", split="train")
        idx = np.load(indices_path)
        self.images = [ds[int(i)]["img"] for i in idx]
        self.labels = [ds[int(i)]["label"] for i in idx]
        self.transform = transform or (TRAIN_TRANSFORM if train else EVAL_TRANSFORM)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        img = self.images[i]
        if self.transform:
            img = self.transform(img)
        return img, self.labels[i]


class CIFAR10Test(Dataset):
    """The real CIFAR-10 test set (10k images) — never touched for training or val."""

    def __init__(self, transform=None):
        ds = load_dataset("uoft-cs/cifar10", split="test")
        self.images = ds["img"]
        self.labels = ds["label"]
        self.transform = transform or EVAL_TRANSFORM

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        img = self.images[i]
        if self.transform:
            img = self.transform(img)
        return img, self.labels[i]


if __name__ == "__main__":
    print("Loading train subset (45k)...")
    tr = CIFAR10Subset("splits/cifar10_train_idx.npy", train=True)
    print(f"  len(tr) = {len(tr)}")
    img, label = tr[0]
    print(f"  sample image shape: {tuple(img.shape)}, label: {label}")

    print("Loading val subset (5k)...")
    va = CIFAR10Subset("splits/cifar10_val_idx.npy", train=False)
    print(f"  len(va) = {len(va)}")

    print("Loading real test set (10k)...")
    te = CIFAR10Test()
    print(f"  len(te) = {len(te)}")

    print("All CIFAR-10 loaders OK.")


# ---------------------------------------------------------------------------
# CIFAR-10-C
#
# Downloaded from https://zenodo.org/record/2535967/files/CIFAR-10-C.tar
# and extracted to data_cache/CIFAR-10-C/. Each <corruption>.npy is shaped
# (50000, 32, 32, 3) uint8: rows 0:10000 are severity 1, 10000:20000 are
# severity 2, ..., 40000:50000 are severity 5. labels.npy (50000,) applies
# identically to every corruption file, since the same 10000 underlying
# test images are reused at every severity.
#
# We deliberately expose only the 15 "standard" corruptions used in the
# original Hendrycks & Dietterich headline tables. The 4 extra/holdout
# corruptions (speckle_noise, gaussian_blur, spatter, saturate) are left
# out of STANDARD_CORRUPTIONS but the files are still on disk if we ever
# want them.
# ---------------------------------------------------------------------------

STANDARD_CORRUPTIONS = [
    "gaussian_noise", "shot_noise", "impulse_noise",
    "defocus_blur", "glass_blur", "motion_blur", "zoom_blur",
    "snow", "frost", "fog", "brightness",
    "contrast", "elastic_transform", "pixelate", "jpeg_compression",
]

C10C_ROOT = "data_cache/CIFAR-10-C"


class CIFAR10C(Dataset):
    """One (corruption, severity) slice of CIFAR-10-C. 10,000 images."""

    def __init__(self, corruption, severity, transform=None, root=C10C_ROOT):
        assert corruption in STANDARD_CORRUPTIONS, f"unknown corruption: {corruption}"
        assert severity in (1, 2, 3, 4, 5), f"severity must be 1-5, got {severity}"

        all_images = np.load(f"{root}/{corruption}.npy")
        all_labels = np.load(f"{root}/labels.npy")

        start = (severity - 1) * 10000
        end = severity * 10000
        self.images = all_images[start:end]  # (10000, 32, 32, 3) uint8
        self.labels = all_labels[start:end]

        self.corruption = corruption
        self.severity = severity
        self.transform = transform or EVAL_TRANSFORM

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        img = self.images[i]  # numpy (32, 32, 3) uint8
        if self.transform:
            # transform expects a PIL-like input; ToTensor accepts numpy arrays directly
            img = self.transform(img)
        return img, int(self.labels[i])


if __name__ == "__main__":
    print("\n--- CIFAR-10-C check ---")
    c = CIFAR10C("gaussian_noise", severity=3)
    print(f"  len = {len(c)}")
    img, label = c[0]
    print(f"  sample image shape: {tuple(img.shape)}, label: {label}")

    c5 = CIFAR10C("gaussian_noise", severity=5)
    print(f"  severity 5, first label: {c5.labels[0]} (severity 1 was: {c.labels[0]}, should match)")

    print(f"  {len(STANDARD_CORRUPTIONS)} standard corruptions configured")
    print("CIFAR-10-C loader OK.")

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

import os as _os
C10C_ROOT = _os.environ.get("CIFAR10C_ROOT", "data_cache/CIFAR-10-C")


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


# ---------------------------------------------------------------------------
# STL-10 -> CIFAR-10 9-class cross-domain shift
#
# Downloaded via torchvision from the official Stanford source to
# data_cache/stl10/. STL-10 test split: 8000 images, 96x96, 10 balanced
# classes (800 each). STL-10's class order (confirmed empirically,
# see src/probe_stl10.py output) is:
#   0 airplane, 1 bird, 2 car, 3 cat, 4 deer, 5 dog,
#   6 horse, 7 monkey, 8 ship, 9 truck
#
# CIFAR-10's class order (from the HF uoft-cs/cifar10 loading script) is:
#   0 airplane, 1 automobile, 2 bird, 3 cat, 4 deer, 5 dog,
#   6 frog, 7 horse, 8 ship, 9 truck
#
# 9 classes overlap. STL's "monkey" has no CIFAR equivalent and is
# DROPPED from this dataset entirely (not remapped to anything).
# CIFAR's "frog" has no STL equivalent -- there is nothing to do about
# that here; it is handled at evaluation time by masking logit index 6
# out of the softmax before computing predictions, so the model is
# never penalised for not predicting a class STL-10 cannot contain.
# ---------------------------------------------------------------------------

STL_TO_CIFAR = {
    0: 0,  # airplane  -> airplane
    1: 2,  # bird      -> bird
    2: 1,  # car       -> automobile
    3: 3,  # cat       -> cat
    4: 4,  # deer      -> deer
    5: 5,  # dog       -> dog
    6: 7,  # horse     -> horse
    # 7 (monkey) intentionally absent -- dropped, not mapped
    8: 8,  # ship      -> ship
    9: 9,  # truck     -> truck
}

CIFAR_CLASSES_MISSING_FROM_STL = [6]  # frog -- mask this logit at eval time

STL_EVAL_TRANSFORM = T.Compose([
    T.Resize((32, 32)),
    T.ToTensor(),
    T.Normalize(CIFAR_MEAN, CIFAR_STD),
])


class STL9(Dataset):
    """STL-10 test set, remapped to CIFAR-10 label space, monkey dropped."""

    def __init__(self, transform=None, root="data_cache/stl10"):
        import torchvision
        raw = torchvision.datasets.STL10(root=root, split="test", download=True)

        self.samples = []  # list of (index_into_raw, cifar_label)
        for i, stl_label in enumerate(raw.labels.tolist()):
            if stl_label in STL_TO_CIFAR:
                self.samples.append((i, STL_TO_CIFAR[stl_label]))

        self.raw = raw
        self.transform = transform or STL_EVAL_TRANSFORM

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        raw_idx, cifar_label = self.samples[idx]
        img, _ = self.raw[raw_idx]  # PIL image, discard original STL label
        if self.transform:
            img = self.transform(img)
        return img, cifar_label


if __name__ == "__main__":
    print("\n--- STL-9 check ---")
    stl = STL9()
    print(f"  len = {len(stl)}  (expect 7200 = 8000 - 800 dropped monkeys)")

    img, label = stl[0]
    print(f"  sample image shape: {tuple(img.shape)}, remapped label: {label}")

    from collections import Counter
    label_counts = Counter(c for _, c in stl.samples)
    print(f"  distinct CIFAR labels present: {sorted(label_counts.keys())}")
    print(f"  label 6 (frog) present? {6 in label_counts}  (should be False)")
    print(f"  per-label counts: {dict(sorted(label_counts.items()))}")

    print("STL-9 loader OK.")

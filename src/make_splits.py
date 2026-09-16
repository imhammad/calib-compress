"""
Generate a fixed, frozen train/val split for CIFAR-10.

Run this ONCE. Commit the two .npy files it produces to git.
Never run this again after that — every script in this project
will load those files and assume they never change. Regenerating
them later would silently invalidate every result computed so far.
"""
import numpy as np
from datasets import load_dataset

SEED = 0
VAL_PER_CLASS = 500  # 500 * 10 classes = 5000 val, 45000 train


def main():
    print("Loading CIFAR-10 from Hugging Face (uoft-cs/cifar10)...")
    ds = load_dataset("uoft-cs/cifar10", split="train")
    labels = np.array(ds["label"])
    n = len(labels)
    print(f"Total training images: {n}")

    rng = np.random.default_rng(SEED)
    val_idx = []
    for c in range(10):
        class_idx = np.where(labels == c)[0]
        rng.shuffle(class_idx)
        val_idx.extend(class_idx[:VAL_PER_CLASS].tolist())

    val_idx = np.array(sorted(val_idx))
    all_idx = np.arange(n)
    train_idx = np.array(sorted(set(all_idx.tolist()) - set(val_idx.tolist())))

    assert len(val_idx) == VAL_PER_CLASS * 10
    assert len(train_idx) == n - len(val_idx)
    assert len(set(train_idx.tolist()) & set(val_idx.tolist())) == 0

    np.save("splits/cifar10_train_idx.npy", train_idx)
    np.save("splits/cifar10_val_idx.npy", val_idx)

    print(f"train: {len(train_idx)}  val: {len(val_idx)}")
    print("Saved to splits/cifar10_train_idx.npy and splits/cifar10_val_idx.npy")

    val_labels = labels[val_idx]
    counts = np.bincount(val_labels, minlength=10)
    print("Val class counts (should all be 500):", counts.tolist())


if __name__ == "__main__":
    main()

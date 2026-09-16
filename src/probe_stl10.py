"""
Throwaway probe — confirms torchvision's official STL-10 test set
downloads correctly and checks the class order, before we wire the
9-class CIFAR remap into data.py.
"""
import torchvision

print("Downloading official STL-10 test split (~2.5 GB, first time only)...")
ds = torchvision.datasets.STL10(root="data_cache/stl10", split="test", download=True)

print("\nNumber of test images:", len(ds))
print("Class names (index order):", ds.classes)

img, label = ds[0]
print("\nFirst image type:", type(img), "size:", img.size)
print("First image label index:", label, "-> class name:", ds.classes[label])

# count how many images per class
from collections import Counter
counts = Counter(ds.labels.tolist())
print("\nPer-class counts:")
for i, name in enumerate(ds.classes):
    print(f"  {i} {name}: {counts[i]}")

# Results Summary

Real, trustworthy runs only. Smoke tests and diagnostics are never
recorded here. Every row must have a checkpoint that passed
verification and a test-set number computed by `eval_test.py`
(never a val_acc number substituted for test).

## Dense baselines

| Arch | Seed | Epochs | Val acc (final) | Test acc | Checkpoint |
|---|---|---|---|---|---|
| ResNet-18 | 0 | 100 | 0.9504 | 0.9482 | `resnet18_dense_s0.pt` |
| ResNet-18 | 1 | 100 | 0.9462 | 0.9430 | `resnet18_dense_s1.pt` |
| ResNet-18 | 2 | 100 | 0.9518 | 0.9475 | `resnet18_dense_s2.pt` |
| **ResNet-18 mean ± std** | — | 100 | 0.9495 ± 0.0030 | **0.9462 ± 0.0028** | — |

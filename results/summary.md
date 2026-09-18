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

## Phase 2 control: fine-tuning, no compression

Same 20-epoch recovery schedule that Phase 3's pruned models will
receive. Purpose: isolate "extra training" from "pruning" as a cause
of any calibration change observed later. Accuracy is not expected to
move much here -- the real comparison is ECE, computed in Phase 7.

| Arch | Seed | Epochs | Val acc (final) | Test acc | Checkpoint |
|---|---|---|---|---|---|
| ResNet-18 (FT) | 0 | +20 | 0.9472 | 0.9454 | `resnet18_denseft_s0.pt` |
| ResNet-18 (FT) | 1 | +20 | 0.9464 | 0.9450 | `resnet18_denseft_s1.pt` |
| ResNet-18 (FT) | 2 | +20 | 0.9492 | 0.9499 | `resnet18_denseft_s2.pt` |
| **ResNet-18 (FT) mean ± std** | — | +20 | 0.9476 ± 0.0015 | **0.9468 ± 0.0027** | — |

Comparison: dense 94.62% ± 0.28% vs. denseft 94.68% ± 0.27% -- no
meaningful accuracy shift from fine-tuning alone, as expected.

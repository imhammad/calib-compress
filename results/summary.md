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

## Phase 3: unstructured global magnitude pruning (ResNet-18)

Pruned from each dense seed's checkpoint, then fine-tuned for 20
epochs using the identical recovery schedule as the Phase 2 control.
"Pre-recovery" is the model's accuracy immediately after pruning,
before any fine-tuning -- this is the real measure of how much damage
pruning did on its own.

| Sparsity | Seed | Pre-recovery val acc | Post-recovery test acc |
|---|---|---|---|
| 30% | 0 | 0.9502 | 0.9472 |
| 30% | 1 | 0.9470 | 0.9431 |
| 30% | 2 | 0.9528 | 0.9459 |
| 50% | 0 | 0.9476 | 0.9471 |
| 50% | 1 | 0.9452 | 0.9423 |
| 50% | 2 | 0.9508 | 0.9477 |
| 70% | 0 | 0.9254 | 0.9477 |
| 70% | 1 | 0.9182 | 0.9431 |
| 70% | 2 | 0.9306 | 0.9444 |
| 90% | 0 | 0.2134 | 0.9473 |
| 90% | 1 | 0.1766 | 0.9413 |
| 90% | 2 | 0.3330 | 0.9449 |

**Key finding:** post-recovery test accuracy is essentially flat
across the entire sparsity range (94.1-94.8%), even though 90%
sparsity causes catastrophic pre-recovery damage (as low as 17.7%
val acc). 20 epochs of fine-tuning fully repairs accuracy at every
sparsity level tested. Whether calibration (ECE) shows the same
flatness, or reveals damage that accuracy hides, is the open question
for Phase 7.

## Phase 3b: structured (channel) pruning via torch-pruning (ResNet-18)

Channel ratio does NOT map linearly to parameter reduction (removing
channels shrinks both a layer's output AND the next layer's input
dimension, compounding the effect) -- reporting the REAL achieved
parameter reduction, not the input channel ratio, to avoid a
misleading comparison to unstructured pruning's sparsity numbers.

| Channel ratio | Real param reduction | Seed | Pre-recovery val acc | Post-recovery test acc |
|---|---|---|---|---|
| 0.15 | 27.95% | 0 | 0.9046 | 0.9463 |
| 0.15 | 27.95% | 1 | 0.9208 | 0.9408 |
| 0.15 | 27.95% | 2 | 0.9218 | 0.9464 |
| 0.30 | 51.14% | 0 | 0.5770 | 0.9453 |
| 0.30 | 51.14% | 1 | 0.8172 | 0.9426 |
| 0.30 | 51.14% | 2 | 0.7438 | 0.9448 |

**Comparison across all Phase 3 arms:** dense 94.62% | denseft (control)
94.68% | unstructured pruning (30-90%) 94.1-94.8% | structured pruning
(28%/51% real reduction) 94.1-94.6%. Post-recovery accuracy is
essentially indistinguishable across every compression method and
severity tested. Real damage before recovery scales with severity as
expected (structured hits harder than unstructured at comparable
parameter reduction, consistent with channel removal being a coarser
operation than individual-weight zeroing), but 20 epochs of
fine-tuning erases the difference every time. This makes the Phase 7
calibration question -- does ECE show the same flatness, or does it
reveal damage accuracy hides -- the central open question of the
paper.

## Phase 5: knowledge distillation (ResNet-18 teacher -> ResNet-10 student)

Student trained with combined KD loss (T=2.0, alpha=0.7 soft/hard
split). Control: identical ResNet-10 architecture trained from
scratch, same 100-epoch schedule, no teacher -- isolates "KD helps"
from "smaller models are inherently easier to fit/calibrate."

| Arch | Seed | Test acc |
|---|---|---|
| ResNet-10 (KD) | 0 | 0.9433 |
| ResNet-10 (KD) | 1 | 0.9429 |
| ResNet-10 (KD) | 2 | 0.9421 |
| **ResNet-10 (KD) mean ± std** | — | **0.9428 ± 0.0006** |
| ResNet-10 (scratch) | 0 | 0.9401 |
| ResNet-10 (scratch) | 1 | 0.9368 |
| ResNet-10 (scratch) | 2 | 0.9393 |
| **ResNet-10 (scratch) mean ± std** | — | **0.9387 ± 0.0017** |

**Finding:** KD gives a small, consistent ~0.4 point accuracy edge
over the scratch-trained control across all 3 seeds (no overlap) --
directionally consistent with soft-target regularization effects
reported in the KD literature. Whether this accuracy edge corresponds
to a calibration edge, or whether KD's soft targets specifically
improve calibration beyond what the accuracy gain would predict, is a
Phase 7 question.

## Status: all compression arms trained
Dense baseline, fine-tuning control, unstructured pruning (4 ratios),
structured pruning (2 ratios), KD, and scratch-KD-control are all
complete for ResNet-18/ResNet-10 -- 27 real, verified checkpoints
total. Remaining before calibration analysis: Phase 4 (quantization,
INT8 PTQ + simulated low-bit) and Phase 6 (dump logits across all
domains for every checkpoint).

## Phase 4: INT8 static post-training quantization (ResNet-18)

One-shot conversion (no training/recovery). Calibrated on 1000 real
training images. Required patching BasicBlock's residual `+=` to use
`nn.quantized.FloatFunctional().add()` -- quantized tensors don't
support the plain += operator (PyTorch's own documented limitation,
same fix used in torchvision's official quantizable ResNet). The
patch has zero learnable parameters, so it's fully backward-compatible
with every checkpoint already trained.

| Seed | FP32 test acc | INT8 test acc | Delta | Checkpoint size |
|---|---|---|---|---|
| 0 | 0.9482 | 0.9478 | -0.0004 | 83MB -> 11.3MB |
| 1 | 0.9430 | 0.9433 | +0.0003 | 83MB -> 11.3MB |
| 2 | 0.9475 | 0.9476 | +0.0001 | 83MB -> 11.3MB |

**Finding:** as anticipated, INT8 PTQ is essentially a null result for
accuracy -- deltas are negligible in both directions across all 3
seeds, while the model shrinks ~7.3x on disk. This is a genuine,
useful finding, not a failure: it suggests INT8 is calibration-safe
purely from the accuracy side. Whether ECE tells the same story (or
whether accuracy hides a real calibration cost even here) is a Phase
7 question, same as every other compression arm. Lower bit-widths
(W4A8/W4A4, the Kuang & Wong regime where real degradation is
expected) are a follow-up, not yet run.

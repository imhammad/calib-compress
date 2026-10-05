# Calibration Under Compression

Code and results for "Calibration Under Compression: How Pruning, Quantization,
and Distillation Affect Uncertainty Under Domain Shift".

## Layout

- `splits/`: fixed, class-balanced 45k/5k CIFAR-10 train/validation split indices (seed 0).
  The 5k validation split is used only to fit the source temperature.
- `src/models.py`: CIFAR-style ResNet-18 and ResNet-10.
- `src/train.py`: dense training (100 epochs, SGD, cosine schedule).
- `src/finetune.py`: 20-epoch recovery schedule, also used for the fine-tuning control.
- `src/prune.py`: global unstructured L1 pruning (30/50/70/90%).
- `src/prune_structured.py`: channel pruning with torch-pruning (ratios 0.15 and 0.30).
- `src/quantize.py`: INT8 post-training static quantization.
- `src/distill.py`: ResNet-18 -> ResNet-10 distillation (T=2, KL weight 0.7).
- `src/dump_logits.py`: saves logits for 33 checkpoints x 78 domains
  (CIFAR-10 val and test, 75 CIFAR-10-C conditions, STL-10).
- `src/metrics.py`: ECE, adaptive ECE, classwise ECE, NLL, Brier, temperature scaling.
- `src/analyze_calibration.py`: raw, source-fit and oracle-fit metrics for every
  checkpoint and domain -> `results/calibration_results.csv`.
- `results/calibration_results.csv`: the single results file every table and figure is built from.

## Reproducing the paper's numbers

Every table and figure can be regenerated from `results/calibration_results.csv`
without retraining:

| Paper item | Command |
|---|---|
| Table 3, Table 4, all numbers in Section 5 | `python3 src/check_numbers.py` |
| Table 5 (five metrics) | `python3 src/summarize_other_metrics.py` |
| Figure 1 (gap vs severity) | `python3 src/review_numbers2.py` |
| Figure 2 (per corruption type) | `python3 src/plot_kd_corruption.py` |

To recompute the results file from logits:

    python3 src/analyze_calibration.py --logits-dir logits

Logits are produced by `src/dump_logits.py` from trained checkpoints; run
`python3 src/<script>.py --help` for each training and compression script's arguments.
Trained checkpoints and logits are too large for this repository and will be
released with the camera-ready version.

## Data

CIFAR-10 (via Hugging Face `uoft-cs/cifar10`), CIFAR-10-C (Hendrycks & Dietterich,
Zenodo record 2535967; set `CIFAR10C_ROOT` to its location), and STL-10 (torchvision).
For STL-10 we keep the nine classes shared with CIFAR-10 and mask the CIFAR-only
frog logit before the softmax.


Training and logit extraction ran on Kaggle (T4 GPU); the pinned versions in
requirements.txt are those used for the analysis.
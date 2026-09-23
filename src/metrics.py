"""
Calibration metrics for the calibration-under-compression project.

All functions take raw logits (N, K) and integer labels (N,) as numpy
arrays, EXCEPT where noted. Softmax is applied internally where needed
-- callers never need to pre-normalize.
"""
import numpy as np
import torch
import torch.nn.functional as F


def softmax_np(logits):
    logits = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(logits)
    return exp / exp.sum(axis=1, keepdims=True)


def accuracy(logits, labels):
    preds = logits.argmax(axis=1)
    return (preds == labels).mean()


def nll(logits, labels):
    """Mean negative log-likelihood."""
    probs = softmax_np(logits)
    n = len(labels)
    true_class_probs = probs[np.arange(n), labels]
    true_class_probs = np.clip(true_class_probs, 1e-12, 1.0)
    return -np.log(true_class_probs).mean()


def brier_score(logits, labels, n_classes=10):
    """Multi-class Brier score: mean squared error between predicted
    probabilities and one-hot labels."""
    probs = softmax_np(logits)
    n = len(labels)
    one_hot = np.zeros((n, n_classes))
    one_hot[np.arange(n), labels] = 1.0
    return ((probs - one_hot) ** 2).sum(axis=1).mean()


def ece(logits, labels, n_bins=15):
    """Standard Expected Calibration Error, equal-width bins on confidence."""
    probs = softmax_np(logits)
    confidences = probs.max(axis=1)
    predictions = probs.argmax(axis=1)
    accuracies = (predictions == labels).astype(float)

    bin_edges = np.linspace(0, 1, n_bins + 1)
    total_ece = 0.0
    n = len(labels)
    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        in_bin = (confidences > lo) & (confidences <= hi) if i > 0 else (confidences >= lo) & (confidences <= hi)
        bin_size = in_bin.sum()
        if bin_size == 0:
            continue
        bin_acc = accuracies[in_bin].mean()
        bin_conf = confidences[in_bin].mean()
        total_ece += (bin_size / n) * abs(bin_acc - bin_conf)
    return total_ece


def adaptive_ece(logits, labels, n_bins=15):
    """Adaptive-bin ECE: bins have equal COUNT rather than equal width,
    less sensitive to confidence distribution shape than standard ECE."""
    probs = softmax_np(logits)
    confidences = probs.max(axis=1)
    predictions = probs.argmax(axis=1)
    accuracies = (predictions == labels).astype(float)

    n = len(labels)
    order = np.argsort(confidences)
    conf_sorted = confidences[order]
    acc_sorted = accuracies[order]

    bin_boundaries = np.linspace(0, n, n_bins + 1).astype(int)
    total_ece = 0.0
    for i in range(n_bins):
        lo, hi = bin_boundaries[i], bin_boundaries[i + 1]
        if hi <= lo:
            continue
        bin_acc = acc_sorted[lo:hi].mean()
        bin_conf = conf_sorted[lo:hi].mean()
        bin_size = hi - lo
        total_ece += (bin_size / n) * abs(bin_acc - bin_conf)
    return total_ece


def classwise_ece(logits, labels, n_classes=10, n_bins=15):
    """Average ECE computed per-class (one-vs-rest), then averaged
    across classes -- catches miscalibration standard ECE can hide
    when it's concentrated in specific classes."""
    probs = softmax_np(logits)
    n = len(labels)
    total = 0.0
    for c in range(n_classes):
        class_probs = probs[:, c]
        class_labels = (labels == c).astype(float)
        bin_edges = np.linspace(0, 1, n_bins + 1)
        class_ece = 0.0
        for i in range(n_bins):
            lo, hi = bin_edges[i], bin_edges[i + 1]
            in_bin = (class_probs > lo) & (class_probs <= hi) if i > 0 else (class_probs >= lo) & (class_probs <= hi)
            bin_size = in_bin.sum()
            if bin_size == 0:
                continue
            bin_acc = class_labels[in_bin].mean()
            bin_conf = class_probs[in_bin].mean()
            class_ece += (bin_size / n) * abs(bin_acc - bin_conf)
        total += class_ece
    return total / n_classes


def fit_temperature(logits, labels, max_iter=200, lr=0.01):
    """
    Fits a single scalar temperature T > 0 to minimize NLL, via L-BFGS,
    following Guo et al. 2017. logits/labels are numpy; returns a
    Python float.
    """
    logits_t = torch.tensor(logits, dtype=torch.float32)
    labels_t = torch.tensor(labels, dtype=torch.int64)

    log_T = torch.zeros(1, requires_grad=True)  # optimize in log-space, guarantees T>0
    optimizer = torch.optim.LBFGS([log_T], lr=lr, max_iter=max_iter)

    def closure():
        optimizer.zero_grad()
        T = torch.exp(log_T)
        loss = F.cross_entropy(logits_t / T, labels_t)
        loss.backward()
        return loss

    optimizer.step(closure)
    return torch.exp(log_T).item()


def apply_temperature(logits, T):
    """Returns logits scaled by 1/T (does not renormalize -- callers
    should apply softmax afterward as usual)."""
    return logits / T


def compute_all_metrics(logits, labels, n_classes=10):
    """Convenience: returns a dict of every metric for one (logits, labels) pair."""
    return {
        "accuracy": accuracy(logits, labels),
        "nll": nll(logits, labels),
        "brier": brier_score(logits, labels, n_classes=n_classes),
        "ece": ece(logits, labels),
        "adaptive_ece": adaptive_ece(logits, labels),
        "classwise_ece": classwise_ece(logits, labels, n_classes=n_classes),
    }

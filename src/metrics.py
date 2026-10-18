import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)


def choose_threshold(labels, probs, target_recall=0.95):
    """Pick the decision threshold on the VALIDATION set.

    Among all thresholds that catch at least `target_recall` of pneumonia cases,
    return the one with the highest precision (fewest false alarms).
    A case is predicted PNEUMONIA when prob >= threshold, so a LOWER threshold
    means HIGHER recall.
    """
    precision, recall, thresholds = precision_recall_curve(labels, probs)
    # precision and recall have one extra final entry with no matching threshold; drop it.
    precision, recall = precision[:-1], recall[:-1]

    meets_target = recall >= target_recall
    best = np.argmax(np.where(meets_target, precision, -1.0))
    return float(thresholds[best])

def compute_metrics(labels, probs, threshold):
    """All metrics for one threshold, returned as a dictionary."""
    labels = np.asarray(labels)
    probs = np.asarray(probs)
    preds = (probs >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(labels, preds, labels=[0, 1]).ravel()
    return {
        "accuracy": accuracy_score(labels, preds),
        "precision": precision_score(labels, preds, zero_division=0),
        "recall": recall_score(labels, preds, zero_division=0),
        "specificity": float(tn / (tn + fp)),
        "f1": f1_score(labels, preds, zero_division=0),
        "roc_auc": roc_auc_score(labels, probs),
        "pr_auc": average_precision_score(labels, probs),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }

CI_METRICS = ("accuracy", "precision", "recall", "specificity", "f1", "roc_auc", "pr_auc")

def bootstrap_ci(labels, probs, threshold, n_boot=1000, seed=0):
    """95% confidence intervals: re-score many random resamples of the dataset."""
    labels = np.asarray(labels)
    probs = np.asarray(probs)
    rng = np.random.default_rng(seed)
    results = {m: [] for m in CI_METRICS}

    for _ in range(n_boot):
        idx = rng.integers(0, len(labels), len(labels))  # draw images WITH replacement
        if len(np.unique(labels[idx])) < 2:              # AUC needs both classes present
            continue
        m = compute_metrics(labels[idx], probs[idx], threshold)
        for key in CI_METRICS:
            results[key].append(m[key])

    return {key: (float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))) for key, v in results.items()}
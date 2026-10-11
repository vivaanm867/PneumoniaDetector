import numpy as np
from sklearn.metrics import precision_recall_curve


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
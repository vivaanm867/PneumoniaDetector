import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # save plots to files instead of opening windows
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from sklearn.metrics import precision_recall_curve, roc_curve
import csv

from data_loader import get_loaders
from metrics import bootstrap_ci, compute_metrics
from model import build_model, get_probabilities

# config
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "best_pneumonia_model.pth"
RESULTS_DIR = Path("results")

# load model
def load_model():
    model = build_model(pretrained=False)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model = model.to(DEVICE)
    model.eval()
    return model

def plot_confusion_matrix(m, path, name):
    cm = np.array([[m["tn"], m["fp"]], [m["fn"], m["tp"]]])
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Normal", "Pneumonia"], yticklabels=["Normal", "Pneumonia"])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(f"Confusion Matrix ({name})")
    plt.savefig(path, dpi=120, bbox_inches="tight")
    plt.close()


def plot_curves(labels, probs, m, threshold, path, name):
    """ROC and precision-recall curves, with the chosen threshold marked as a red dot."""
    fpr, tpr, _ = roc_curve(labels, probs)
    precision, recall, _ = precision_recall_curve(labels, probs)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))

    ax1.plot(fpr, tpr, label=f"AUC = {m['roc_auc']:.3f}")
    ax1.plot([0, 1], [0, 1], "--", color="gray")  # what random guessing would look like
    ax1.scatter([1 - m["specificity"]], [m["recall"]], color="red", zorder=3, label=f"threshold {threshold:.2f}")
    ax1.set_xlabel("False positive rate (1 - specificity)")
    ax1.set_ylabel("True positive rate (recall)")
    ax1.set_title("ROC curve")
    ax1.legend(loc="lower right")

    ax2.plot(recall, precision, label=f"AP = {m['pr_auc']:.3f}")
    ax2.scatter([m["recall"]], [m["precision"]], color="red", zorder=3, label=f"threshold {threshold:.2f}")
    ax2.set_xlabel("Recall")
    ax2.set_ylabel("Precision")
    ax2.set_title("Precision-recall curve")
    ax2.legend(loc="lower left")

    fig.suptitle(f"{name} set")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def evaluate(model, loader, threshold, name):
    labels, probs = get_probabilities(model, loader, DEVICE)
    m = compute_metrics(labels, probs, threshold)
    ci = bootstrap_ci(labels, probs, threshold)

    print(f"\n--- {name} set: {len(labels)} images, threshold {threshold:.3f} ---")
    for key in ("accuracy", "precision", "recall", "specificity", "f1", "roc_auc", "pr_auc"):
        low, high = ci[key]
        print(f"{key:<12} {m[key] * 100:6.2f}%   95% CI [{low * 100:.2f}, {high * 100:.2f}]")
    print(f"Confusion: TN={m['tn']} FP={m['fp']} FN={m['fn']} TP={m['tp']}")

    RESULTS_DIR.mkdir(exist_ok=True)
    plot_confusion_matrix(m, RESULTS_DIR / f"confusion_matrix_{name}.png", name)
    plot_curves(labels, probs, m, threshold, RESULTS_DIR / f"curves_{name}.png", name)
    with open(RESULTS_DIR / f"metrics_{name}.json", "w") as f:
        json.dump({"threshold": threshold, "n_images": len(labels), "metrics": m, "ci95": ci}, f, indent=2)

    # One row per image, so gradcam.py can find the false positives and false negatives
    with open(RESULTS_DIR / f"predictions_{name}.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["path", "label", "prob_pneumonia", "pred"])
        for (path, _), label, prob in zip(loader.dataset.samples, labels, probs):
            writer.writerow([path, int(label), f"{prob:.6f}", int(prob >= threshold)])
    
    return m

# main
if __name__ == "__main__":
    # Threshold chosen on the validation set by train.py
    with open("threshold.json") as f:
        threshold = json.load(f)["threshold"]
    print(f"Using threshold {threshold:.3f} (chosen on validation by train.py)")

    model = load_model()
    _, val_loader, test_loader = get_loaders()
    evaluate(model, val_loader, threshold, "val")
    evaluate(model, test_loader, threshold, "test")
    print(f"\nPlots and metrics saved to {RESULTS_DIR}/")
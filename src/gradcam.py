import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from data_loader import val_test_transforms
from model import build_model

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "best_pneumonia_model.pth"
RESULTS_DIR = Path("results")
PER_CATEGORY = 6  # images per grid

"""Grad-CAM heatmaps: where in the X-ray is the model looking?

Run evaluate.py first; this reads results/predictions_test.csv and draws heatmaps for
the most confident false positives, false negatives, true positives and true negatives.
"""

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.activations = None
        self.gradients = None
        # A "hook" is a function PyTorch calls automatically when the layer runs
        target_layer.register_forward_hook(self.save_activations)

    def save_activations(self, module, inputs, output):
        self.activations = output                    # the feature maps (question 1)
        output.register_hook(self.save_gradients)    # ask to be called again on the backward pass

    def save_gradients(self, grad):
        self.gradients = grad                        # how much each map matters (question 2)

    def __call__(self, x):
        self.model.zero_grad()
        logits = self.model(x)
        logits[0, 1].backward()  # backpropagate from the PNEUMONIA score

        weights = self.gradients.mean(dim=(2, 3), keepdim=True)              # one importance value per map
        cam = F.relu((weights * self.activations).sum(dim=1, keepdim=True))  # weighted sum, keep positives
        cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)  # 7x7 -> 224x224
        cam = cam[0, 0]
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)             # scale to 0..1 for display

        prob = torch.softmax(logits, dim=1)[0, 1].item()
        return cam.detach().cpu().numpy(), prob


def explain(cam, path):
    """Return (grayscale image for display, heatmap, pneumonia probability) for one X-ray."""
    with Image.open(path) as img:
        img = img.convert("RGB")
    x = val_test_transforms(img).unsqueeze(0).to(DEVICE)  # same preprocessing as evaluation
    heatmap, prob = cam(x)
    display = np.array(img.convert("L").resize((224, 224)))
    return display, heatmap, prob


# name: (true label, predicted label, sort highest probability first?)
CATEGORIES = {
    "false_positives": (0, 1, True),   # normal called pneumonia, most confident mistakes first
    "false_negatives": (1, 0, False),  # pneumonia missed, most confident mistakes first
    "true_positives": (1, 1, True),
    "true_negatives": (0, 0, False),
}


def save_grid(cam, rows, title, path):
    fig, axes = plt.subplots(2, len(rows), figsize=(3 * len(rows), 6.5), squeeze=False)
    for i, row in enumerate(rows):
        display, heatmap, prob = explain(cam, row["path"])
        axes[0, i].imshow(display, cmap="gray")
        axes[0, i].set_title(f"{Path(row['path']).name}\nP(pneumonia)={prob:.2f}", fontsize=8)
        axes[1, i].imshow(display, cmap="gray")
        axes[1, i].imshow(heatmap, cmap="jet", alpha=0.4)
        axes[0, i].axis("off")
        axes[1, i].axis("off")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main():
    model = build_model(pretrained=False)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model = model.to(DEVICE).eval()
    cam = GradCAM(model, model.layer4[-1])  # last block of layer4 = last layer with spatial maps

    with open(RESULTS_DIR / "predictions_test.csv") as f:
        rows = list(csv.DictReader(f))

    out_dir = RESULTS_DIR / "gradcam"
    out_dir.mkdir(exist_ok=True)
    for name, (label, pred, highest_first) in CATEGORIES.items():
        matches = [r for r in rows if int(r["label"]) == label and int(r["pred"]) == pred]
        matches.sort(key=lambda r: float(r["prob_pneumonia"]), reverse=highest_first)
        chosen = matches[:PER_CATEGORY]
        if not chosen:
            continue
        save_grid(cam, chosen, f"{name.replace('_', ' ')} (test set)", out_dir / f"{name}.png")
        print(f"{name}: {len(matches)} total, saved {len(chosen)} examples")
    print(f"Heatmaps saved to {out_dir}/")


if __name__ == "__main__":
    main()
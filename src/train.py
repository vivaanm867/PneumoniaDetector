import torch
import torch.nn as nn
import torch.optim as optim
from model import build_model, get_probabilities
from data_loader import get_loaders, class_weights
import json
from sklearn.metrics import roc_auc_score
from metrics import choose_threshold

import random

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def set_seed(seed):
    """Make training repeatable: same seed -> same starting weights, shuffling and augmentation."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def plot_history(history, path):
    epochs = [h["epoch"] for h in history]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    ax1.plot(epochs, [h["train_loss"] for h in history])
    ax1.set_xlabel("Epoch"); ax1.set_ylabel("Training loss"); ax1.set_title("Training loss")
    ax2.plot(epochs, [h["val_auc"] for h in history], color="tab:green")
    ax2.set_xlabel("Epoch"); ax2.set_ylabel("ROC-AUC"); ax2.set_title("Validation ROC-AUC")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)

def main():
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device {device}")
    train_loader, val_loader, _ = get_loaders() # get loaders returns 3 loaders, traning only needs 2 of them

    # move model to device
    model = build_model()
    model = model.to(device)

    weights = class_weights(train_loader.dataset.samples) # weights computed from training samples only
    print(f"Class weights (NORMAL, PNEUMONIA): {weights.tolist()}") # record which weights were used
    criterion = nn.CrossEntropyLoss(weight=weights.to(device))

    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=0.0001
    )
    
    # Halve the learning rate when validation AUC hasn't improved for 2 epochs
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=2)

    # training loop
    num_epochs = 30
    best_val_auc = 0.0
    best_epoch = 0
    patience = 5  # stop after this many epochs without improvement
    epochs_without_improvement = 0
    history = []

    for epoch in range(num_epochs):
        model.train() #set model to training mode
        running_loss = 0.0

        for batch_idx, (images, labels) in enumerate(train_loader):
            if batch_idx % 10 == 0:
                print(f"Batch {batch_idx}/{len(train_loader)}")

            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        avg_loss = running_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}")

        # Validation: score every validation image, measure ranking quality with ROC-AUC
        val_labels, val_probs = get_probabilities(model, val_loader, device)
        val_auc = roc_auc_score(val_labels, val_probs)
        print(f"Validation ROC-AUC: {val_auc:.4f}")

        scheduler.step(val_auc)
        current_lr = optimizer.param_groups[0]["lr"]
        history.append({"epoch": epoch + 1, "train_loss": avg_loss, "val_auc": float(val_auc), "lr": current_lr})
        print(f"Learning rate: {current_lr:.6f}")

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_epoch = epoch + 1
            epochs_without_improvement = 0
            torch.save(model.state_dict(), "best_pneumonia_model.pth")
            print("Saved new best model")
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                print(f"Early stopping: no improvement for {patience} epochs")
                break

    with open("history.json", "w") as f:
        json.dump(history, f, indent=2)
    plot_history(history, "training_curves.png")
    print(f"\nBest epoch: {best_epoch}. Saved history.json and training_curves.png")

    # Choose the decision threshold on the validation set, using the BEST model (not the last one)
    model.load_state_dict(torch.load("best_pneumonia_model.pth", map_location=device))
    val_labels, val_probs = get_probabilities(model, val_loader, device)
    threshold = choose_threshold(val_labels, val_probs, target_recall=0.95)

    print(f"\nBest validation ROC-AUC: {best_val_auc:.4f}")
    print(f"Threshold that keeps validation recall >= 95%: {threshold:.3f}")

    with open("threshold.json", "w") as f:
        json.dump({"threshold": threshold, "target_recall": 0.95, "best_val_auc": float(best_val_auc), "best_epoch": best_epoch}, f, indent=2)
    print("Saved threshold to threshold.json")

if __name__ == "__main__":
    main()

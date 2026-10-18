import torch
import torch.nn as nn
import torch.optim as optim
from model import build_model, get_probabilities
from data_loader import get_loaders, class_weights
import json
from sklearn.metrics import roc_auc_score
from metrics import choose_threshold


def main():
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

    # training loop
    num_epochs = 2
    best_val_auc = 0.0

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

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            torch.save(model.state_dict(), "best_pneumonia_model.pth")
            print("Saved new best model")

    # Choose the decision threshold on the validation set, using the BEST model (not the last one)
    model.load_state_dict(torch.load("best_pneumonia_model.pth", map_location=device))
    val_labels, val_probs = get_probabilities(model, val_loader, device)
    threshold = choose_threshold(val_labels, val_probs, target_recall=0.95)

    print(f"\nBest validation ROC-AUC: {best_val_auc:.4f}")
    print(f"Threshold that keeps validation recall >= 95%: {threshold:.3f}")

    with open("threshold.json", "w") as f:
        json.dump({"threshold": threshold, "target_recall": 0.95, "best_val_auc": float(best_val_auc)}, f, indent=2)
    print("Saved threshold to threshold.json")

if __name__ == "__main__":
    main()

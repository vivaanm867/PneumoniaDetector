import torch
import torch.nn as nn
from torchvision import models

NUM_CLASSES = 2


def build_model(pretrained=True):
    """ResNet-18 with only layer4 and a new classifier head trainable.

    pretrained=True downloads ImageNet weights (used for training).
    pretrained=False skips the download (used when loading a saved checkpoint,
    since the checkpoint's weights replace them anyway).
    """
    weights = models.ResNet18_Weights.DEFAULT if pretrained else None
    model = models.resnet18(weights=weights)

    # Freeze everything, then unfreeze the last block
    for param in model.parameters():
        param.requires_grad = False
    for param in model.layer4.parameters():
        param.requires_grad = True

    # Replace the final layer: 512 features -> 256 -> 2 classes (normal vs pneumonia)
    num_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(num_features, 256),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(256, NUM_CLASSES),
    )
    return model


def get_probabilities(model, loader, device):
    """Run the model on every image in a loader. Return the true labels and pneumonia probabilities."""
    model.eval()
    all_labels, all_probs = [], []
    with torch.no_grad():
        for images, labels in loader:
            outputs = model(images.to(device))
            probs = torch.softmax(outputs, dim=1)[:, 1]  # probability of class 1 = PNEUMONIA
            all_labels.extend(labels.numpy())
            all_probs.extend(probs.cpu().numpy())
    return all_labels, all_probs
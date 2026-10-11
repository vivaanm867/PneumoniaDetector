import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from pathlib import Path

import re # for reading patient ID's out of file names with regex
from PIL import Image
from sklearn.model_selection import StratifiedGroupKFold
from torch.utils.data import Dataset
import numpy as np

# <repo>/src/data_loader.py -> <repo>/data
DEFAULT_DATA_DIR = Path(__file__).resolve().parent.parent / "data"

# image transformations / normalization
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(num_output_channels=3),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.RandomAffine(degrees=0, translate=(0.1,0.1), scale=(0.95,1.05)),
    transforms.ColorJitter(brightness=0.05, contrast=0.05),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])
val_test_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

# Kaggle file names encode the patient:
#   PNEUMONIA: person1_bacteria_1.jpeg, person1_virus_6.jpeg  -> "person1"
#   NORMAL:    IM-0115-0001.jpeg, NORMAL2-IM-0927-0001.jpeg    -> "IM-0115" / "NORMAL2-IM-0927"
PATIENT_PATTERNS = (
    re.compile(r"^(person\d+)_", re.IGNORECASE),
    re.compile(r"^((?:NORMAL2-)?IM-\d+)-", re.IGNORECASE),
)
# ^(person\d+)_ means "start with person followed by digits, then an underscore" --> capures patients like person1
# (?:NORMAL2-)? means an optional prefix followed by IM- and digits, captures images of the patient

def patient_id(path):
    name = Path(path).name
    for pattern in PATIENT_PATTERNS:
        match = pattern.match(name)
        if match:
            return match.group(1).lower()
        
    return name.lower()  # unknown format: treat the image as its own patient

def patient_split(samples, val_fraction=0.15, seed=42):
    """Split [(path, label), ...] so no patient is in both train and val."""

    labels = [label for _, label in samples]
    groups = [patient_id(path) for path, _ in samples]
    n_splits = round(1 / val_fraction)  # 0.15 -> 7 folds -> one fold is ~14% of the data

    # stratifiedgroupkfold does two jobs at once
    # group: all images w same patient ID in the same fold
    # stratified: each fold keeps roughly same ratio of norm/pneu
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    train_idx, val_idx = next(splitter.split(np.zeros(len(samples)), labels, groups))
    
    return [samples[i] for i in train_idx], [samples[i] for i in val_idx]

# idea is to compute the weights from the actual class counts
# weight for a class = total images / (number of classes × images in that class

def class_weights(samples, num_classes=2):
    """Inverse-frequency weights: total / (num_classes * count) for each class."""

    counts = np.bincount([label for _, label in samples], minlength=num_classes)
    weights = len(samples) / (num_classes * counts)

    return torch.tensor(weights, dtype=torch.float32)

def get_loaders(data_dir=DEFAULT_DATA_DIR, batch_size=32, val_fraction=0.15, seed=42):
    data_dir = Path(data_dir)

    # Pool the official train folder and the 16-image val folder, then re-split by patient.
    # ImageFolder without a transform only lists the files; it doesn't load any images.
    pool = datasets.ImageFolder(data_dir / "train").samples + datasets.ImageFolder(data_dir / "val").samples
    train_samples, val_samples = patient_split(pool, val_fraction, seed)

    train_dataset = XrayDataset(train_samples, train_transform)
    val_dataset = XrayDataset(val_samples, val_test_transforms)
    test_dataset = datasets.ImageFolder(data_dir / "test", transform=val_test_transforms)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader

class XrayDataset(Dataset):
    def __init__(self, samples, transform):
        self.samples = samples      # list of (image_path, label)
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        with Image.open(path) as img:
            img = img.convert("RGB")
        return self.transform(img), label

# data loader test
if __name__ == "__main__":
    train_loader, val_loader, test_loader = get_loaders()

    split_patients = {}
    for name, loader in (("train", train_loader), ("val", val_loader)):
        samples = loader.dataset.samples
        labels = [label for _, label in samples]
        split_patients[name] = {patient_id(path) for path, _ in samples}
        print(f"{name}: {len(samples)} images, {len(split_patients[name])} patients "
              f"(NORMAL={labels.count(0)}, PNEUMONIA={labels.count(1)})")

    overlap = split_patients["train"] & split_patients["val"]
    print(f"Patients in both train and val: {len(overlap)}")
    print(f"Class weights (NORMAL, PNEUMONIA): {class_weights(train_loader.dataset.samples).tolist()}")

    print(f"Train batches: {len(train_loader)}, Validation batches: {len(val_loader)}, Test batches: {len(test_loader)}")
    images, labels = next(iter(train_loader))
    print(f"Batch images shape: {images.shape}, Batch labels shape: {labels.shape}")
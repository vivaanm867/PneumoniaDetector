import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from pathlib import Path

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

def get_loaders(data_dir=DEFAULT_DATA_DIR, batch_size=32):
    data_dir = Path(data_dir)

    train_dataset = datasets.ImageFolder(data_dir / "train", transform=train_transform)
    val_dataset = datasets.ImageFolder(data_dir / "val", transform=val_test_transforms)
    test_dataset = datasets.ImageFolder(data_dir / "test", transform=val_test_transforms)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader

# data loader test
if __name__ == "__main__":
    train_loader, val_loader, test_loader = get_loaders()
    print(f"Train batches: {len(train_loader)}, Validation batches: {len(val_loader)}, Test batches: {len(test_loader)}")

    images, labels = next(iter(train_loader))
    print(f"Batch images shape: {images.shape}, Batch labels shape: {labels.shape}")
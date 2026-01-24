import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

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

# load data
train_dataset = datasets.ImageFolder(root="C:/Users/vivaa/OneDrive/Documents/GitHub/PneumoniaDetector/data/train", transform=train_transform)
val_dataset = datasets.ImageFolder(root="C:/Users/vivaa/OneDrive/Documents/GitHub/PneumoniaDetector/data/val", transform=val_test_transforms)
test_dataset = datasets.ImageFolder(root="C:/Users/vivaa/OneDrive/Documents/GitHub/PneumoniaDetector/data/test", transform=val_test_transforms)

# create data loaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# data loader test
if __name__ == "__main__":
    print(f"Train batches: {len(train_loader)}, Validation batches: {len(val_loader)}, Test batches: {len(test_loader)}")
    
    images, labels = next(iter(train_loader))
    print(f"Batch images shape: {images.shape}, Batch labels shape: {labels.shape}")
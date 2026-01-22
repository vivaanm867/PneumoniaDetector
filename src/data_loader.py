import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# image transformations / normalization
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

# load data
train_dataset = datasets.ImageFolder(root="C:/Users/vivaa/OneDrive/Documents/GitHub/PneumoniaDetector/data/train", transform=transform)
val_dataset = datasets.ImageFolder(root="C:/Users/vivaa/OneDrive/Documents/GitHub/PneumoniaDetector/data/val", transform=transform)
test_dataset = datasets.ImageFolder(root="C:/Users/vivaa/OneDrive/Documents/GitHub/PneumoniaDetector/data/test", transform=transform)

# create data loaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# data loader test
if __name__ == "__main__":
    print(f"Train batches: {len(train_loader)}, Validation batches: {len(val_loader)}, Test batches: {len(test_loader)}")
    
    images, labels = next(iter(train_loader))
    print(f"Batch images shape: {images.shape}, Batch labels shape: {labels.shape}")
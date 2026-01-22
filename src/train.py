import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from data_loader import train_loader, val_loader

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device {device}")

    # load pre trained model
    model = models.resnet18(weights=True)

    # replace last layer to output 2 classes (normal vs pneumonia)
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, 2)

    # move model to device
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # training loop
    num_epochs = 5

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

        # Validation
        model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.to(device)

                output = model(images)
                _, predicted = torch.max(outputs, 1)

                batch_size = labels.size(0)
                correct += (predicted[:batch_size] == labels).sum().item()
                total += batch_size
        
        val_accuracy = 100 * correct / total
        print(f"Validation accuracy: {val_accuracy:.2f}%")

    torch.save(model.state_dict(), "pneumonia_model.pth")
    print("\nModel saved as pneumonia_model.pth")

if __name__ == "__main__":
    main()

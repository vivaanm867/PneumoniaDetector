import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from data_loader import train_loader, val_loader
from sklearn.metrics import confusion_matrix

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device {device}")

    # load pre trained model
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

    # freeze pretrained layers
    for param in model.parameters():
        param.requires_grad = False

    for name, param in model.named_parameters():
        if "layer4" in name or "fc" in name:
            param.requires_grad = True

    # replace last layer to output 2 classes (normal vs pneumonia)
    num_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(num_features, 256),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(256, 2))

    # move model to device
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=0.0001
    )

    # training loop
    num_epochs = 30
    best_val_acc = 0.0

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

        all_labels = []
        all_predictions = []

        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.to(device)

                output = model(images)
                _, predicted = torch.max(output, 1)

                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                all_labels.extend(labels.cpu().numpy())
                all_predictions.extend(predicted.cpu().numpy())
        
        val_accuracy = 100 * correct / total
        print(f"Validation accuracy: {val_accuracy:.2f}%")

        if val_accuracy > best_val_acc:
            best_val_acc = val_accuracy
            torch.save(model.state_dict(), "best_pneumonia_model.pth")
            print("Saved new best model")

    # confusion matrix
    cm = confusion_matrix(all_labels, all_predictions)
    print("Confusion Matrix:\n", cm)

    torch.save(model.state_dict(), "pneumonia_model.pth")
    print("\nModel saved as pneumonia_model.pth")

if __name__ == "__main__":
    main()

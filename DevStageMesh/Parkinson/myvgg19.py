import os
import time
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
from torchvision.models import vgg19
from torch.utils.data import DataLoader, random_split
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score

def get_transforms(train=True):
    """Get different transforms for training and validation datasets."""
    if train:
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomCrop((220, 220)),
            # transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            # transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.CenterCrop(220),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

if __name__ == '__main__':
    print("cuda.is_available " + str(torch.cuda.is_available()))
    # Define data transforms
    transform = get_transforms(False)

    # Define data directory
    data_dir = "train_data"
    dataset = torchvision.datasets.ImageFolder(root=data_dir, transform=transform)

    # Determine the number of classes in your dataset
    num_classes = len(os.listdir(data_dir))

    # Define hyperparameters
    num_epochs = 15
    learning_rate = 0.001
    batch_size = 16

    # Define cross-validation strategy (e.g., 5-fold)
    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    for fold, (train_indices, val_indices) in enumerate(kf.split(range(len(dataset)), dataset.targets)):
        print(f"Fold {fold + 1}:")

        # Split the dataset into training and validation sets for this fold
        train_dataset = torch.utils.data.Subset(dataset, train_indices)
        val_dataset = torch.utils.data.Subset(dataset, val_indices)

        # Create data loaders for training and validation
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

        # Define the model
        # weights=VGG19_Weights.IMAGENET1K_V1
        model = vgg19(pretrained=True)
        model.classifier[6] = nn.Linear(4096, num_classes)

        # Set the device
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model.to(device)

        # Define the loss function and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)

        # Training loop
        for epoch in range(num_epochs):
            epoch_start_time = time.time()  # Capture the start time of the epoch
            iterations_start_time = time.time()  # Capture the start time of the iteration

            model.train()
            running_loss = 0.0

            for i, data in enumerate(train_loader, 0):
                inputs, labels = data
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()

                if i % 10 == 9:
                    iterations_delta_time = time.time() - iterations_start_time  # Calculate the time difference
                    iterations_start_time = time.time()

                    print(f"[{epoch + 1}, {i + 1}], {iterations_delta_time}, loss: {running_loss / 10:.3f}")

                    running_loss = 0.0

            # Validation loop
            model.eval()
            y_true = []
            y_pred = []

            with torch.no_grad():
                for data in val_loader:
                    images, labels = data
                    images, labels = images.to(device), labels.to(device)
                    outputs = model(images)
                    _, predicted = torch.max(outputs, 1)

                    y_true.extend(labels.cpu().numpy())
                    y_pred.extend(predicted.cpu().numpy())

            f1 = f1_score(y_true, y_pred, average='weighted')
            epoch_delta_time = time.time() - epoch_start_time  # Calculate the time difference

            print(f"F1 Score (Fold {fold + 1}, Epoch {epoch + 1}, sec {epoch_delta_time}): {f1:.4f}")

        # Save the model for this fold if needed
        torch.save(model, f"model_vgg19_fold{fold + 1}.pth")
import os
import time

import torchvision
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torchvision.models import vgg16, VGG16_Weights
from torchvision import transforms

import matplotlib.pyplot as plt

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.metrics import roc_curve, auc


class CustomImageDataset(Dataset):
    def __init__(self, main_dir, transform=None):
        self.main_dir = main_dir
        self.transform = transform
        # list containing the names of the subfolders in main directory
        self.classes = os.listdir(main_dir)
        # alphabetical ascending order
        self.classes.sort()
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}

        # list of tuples if form ("fullpath/img", class_index), sequence of items that can't be changed (immutable)
        self.imgs = []
        for cls in self.classes:
            cls_folder = os.path.join(self.main_dir, cls)
            for img in os.listdir(cls_folder):
                self.imgs.append((os.path.join(cls_folder, img), self.class_to_idx[cls]))

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        img_path, label = self.imgs[idx]
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, label




# allwidth, allheight = 0, 0

# for idx in range(len(dataset)):
#     # Get image and label
#     image, label = dataset[idx]
#     # peculiarity image.shape of is that it gives the opposite values, usually it is considered width, height, depth
#     depth, height, width = image.shape[0], image.shape[1], image.shape[2]

#     allwidth+=width
#     allheight+=height

#     print(f"Image {idx}:")
#     print(f"Depth: {depth}, Width: {width}, Height: {height}")  # Print image dimensions
#     print(f"Label: {label}")  # Print image label
#     print("-" * 20)

# print("Average width=", allwidth/len(dataset), ", height=", allheight/len(dataset))

def conv_output_size(input_size, filter_size, stride, padding):
    output_size = ((input_size - filter_size + 2 * padding) // stride) + 1
    return output_size


def afterConvs(input_size):
    input_size = conv_output_size(input_size, 3, 1, 0)
    input_size = conv_output_size(input_size, 2, 2, 0)
    input_size = conv_output_size(input_size, 3, 3, 0)
    input_size = conv_output_size(input_size, 2, 2, 0)
    return input_size


class MyCNN(nn.Module):
    def __init__(self):
        super().__init__()

        # Convolutional layers
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=0)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=3, padding=0)

        # Batch normalization layers
        self.batchnorm1 = nn.BatchNorm2d(32)
        self.batchnorm2 = nn.BatchNorm2d(64)

        # Pooling layer
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Fully connected layers
        self.fc1 = nn.Linear(afterConvs(INPUT_SIZE) * afterConvs(INPUT_SIZE) * 64, 1024)
        self.fc2 = nn.Linear(1024, 30)

        # Dropout
        self.dropout = nn.Dropout(0.5)

        # Initialize weights
        nn.init.xavier_uniform_(self.conv1.weight)
        nn.init.xavier_uniform_(self.conv2.weight)
        nn.init.normal_(self.fc1.weight, std=0.01)
        nn.init.normal_(self.fc2.weight, std=0.01)

    def forward(self, x):
        # Apply convolutional layers, batch normalization, and ReLU activation
        x = self.pool(F.relu(self.batchnorm1(self.conv1(x))))
        x = self.pool(F.relu(self.batchnorm2(self.conv2(x))))

        # Flatten the tensor for the fully connected layer
        x = x.flatten(start_dim=1)
        # Apply fully connected layers with Sigmoid and dropout
        x = F.sigmoid(self.fc1(x))
        x = self.dropout(x)
        x = F.sigmoid(self.fc2(x))

        return x


if __name__ == "__main__":

    data_dir = "train"

    INPUT_SIZE = 224

    # Usage Example
    transform = transforms.Compose([
        transforms.Resize((INPUT_SIZE, INPUT_SIZE)),
        transforms.RandomRotation(3),
        transforms.ToTensor(),
    ])

    # dataset = CustomImageDataset(main_dir=data_dir, transform=transform)
    dataset = torchvision.datasets.ImageFolder(root=data_dir, transform=transform)
    data_loader = DataLoader(dataset, batch_size=4, shuffle=True)

    # Determine the number of classes in your dataset
    num_classes = len(os.listdir(data_dir))

    # Define hyperparameters
    num_epochs = 10
    learning_rate = 0.001
    batch_size = 24

    # Define cross-validation strategy (e.g., 5-fold)
    kf = StratifiedKFold(n_splits=2, shuffle=True, random_state=42)

    for fold, (train_indices, val_indices) in enumerate(kf.split(range(len(dataset)), dataset.targets)):
        print(f"Fold {fold + 1}:")

        # Split the dataset into training and validation sets for this fold
        train_dataset = torch.utils.data.Subset(dataset, train_indices)
        val_dataset = torch.utils.data.Subset(dataset, val_indices)

        # Create data loaders for training and validation
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

        # Define the model
        model = vgg16(weights=VGG16_Weights.IMAGENET1K_V1)
        # model = MyCNN
        model.classifier[6] = nn.Linear(4096, num_classes)

        # Set the device
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(device)
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

        # После обучения модели и вычисления метрик для текущего fold
        precision = precision_score(y_true, y_pred, average='weighted')
        recall = recall_score(y_true, y_pred, average='weighted')
        f1 = f1_score(y_true, y_pred, average='weighted')

        # Визуализация метрик для текущего fold
        metrics = [precision, recall, f1]
        metric_names = ['Precision', 'Recall', 'F1 Score']
        plt.figure(figsize=(7, 4))
        plt.bar(metric_names, metrics, color=['blue', 'orange', 'green'])
        plt.title(f'Metrics for Fold {fold + 1}')
        plt.ylim([0, 1])
        for i, v in enumerate(metrics):
            plt.text(i, v + 0.02, f"{v:.2f}", ha='center', va='bottom')
        plt.show(block=False)

        # Save the model for this fold if needed
        torch.save(model.state_dict(), f"model_fold{fold + 1}.pth")

    input("Press Enter to exit...")

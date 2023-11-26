import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, datasets
from PIL import Image
from sklearn.metrics import f1_score


# 1. Load and preprocess the dataset
class CustomDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.images, self.labels = self.load_data()

    def load_data(self):
        img_paths = []
        labels = []
        self.classes = os.listdir(self.root_dir)
        for label, class_name in enumerate(self.classes):
            class_dir = os.path.join(self.root_dir, class_name)
            for img_name in os.listdir(class_dir):
                img_paths.append(os.path.join(class_dir, img_name))
                labels.append(label)
        return img_paths, labels

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]  # получение пути к изображению
        image = Image.open(img_path).convert('RGB')  # открытие и конвертация изображения в RGB
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, label

    def __str__(self):
        class_str = ", ".join(self.classes)
        return f"CustomDataset with classes: {class_str}"


class VGGLike(nn.Module):
    def __init__(self):
        super(VGGLike, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.classifier = nn.Sequential(
            nn.Linear(128 * 32 * 32, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 1),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return torch.sigmoid(x)


# 2. Define the CNN model
class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.fc1 = nn.Linear(16 * 64 * 64, 1)

    def forward(self, x):
        out = self.conv1(x)
        out = self.relu(out)
        out = self.maxpool(out)
        out = out.view(out.size(0), -1)
        out = self.fc1(out)
        out = torch.sigmoid(out)  # Применяем сигмоиду
        return out


if __name__ == "__main__":

    print(torch.cuda.is_available())
    device = torch.device('cuda')
    # if torch.cuda.is_available() else 'cpu')

    transform = transforms.Compose([
        transforms.Resize((130, 130)),  # resizing to a bit larger size before random crop
        transforms.RandomCrop((128, 128)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
    ])

    train_dataset = CustomDataset("train_data", transform)
    print(train_dataset)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    model = VGGLike()
    model.to(device)

    # 3. Define the loss function and optimizer
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 4. Train the model
    for epoch in range(10):  # loop over the dataset multiple times

        running_loss = 0.0
        all_preds = []  # Store all predictions
        all_labels = []  # Store all true labels

        for i, (inputs, labels) in enumerate(train_loader, 0):
            inputs, labels = inputs.to(device), labels.to(device)

        # Изменяем форму меток и конвертируем их в float
        labels = labels.view(-1, 1).float()

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        # Store predictions and true labels
        preds = outputs.detach().cpu().numpy() > 0.5  # Convert to binary predictions
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().numpy())

    # Calculate F1 Score
    f1 = f1_score(all_labels, all_preds, average='binary')  # you can choose other averaging methods
    print(f'Epoch {epoch + 1}, Loss: {running_loss / len(train_loader)}, F1 Score: {f1:.4f}')

    print('Finished Training')

# 5. Evaluate the model
# You may want to use a separate validation dataset to evaluate your model's performance
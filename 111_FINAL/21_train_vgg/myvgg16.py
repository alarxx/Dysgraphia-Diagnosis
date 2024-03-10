import os
import time
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
from torchvision.models import vgg16, VGG16_Weights
from torch.utils.data import DataLoader, random_split
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import numpy as np
import cv2
import random
from PIL import Image
import csv


# Определение класса RandomContoursRemovalTransform
class RandomContoursRemovalTransform(object):
    def __init__(self, removal_probability=0.4):
        self.removal_probability = removal_probability

    def __call__(self, img):
        # Convert PIL image to numpy array
        img_np = np.array(img)

        # Convert RGB to Grayscale
        gray_img = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)

        # Apply threshold using Otsu's method
        _, binary_img = cv2.threshold(gray_img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        # Find contours, remove some, and draw them back onto the RGB image
        contours, _ = cv2.findContours(binary_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        num_contours_to_remove = int(len(contours) * self.removal_probability)
        contours_to_remove = random.sample(contours, num_contours_to_remove)
        cv2.drawContours(img_np, contours_to_remove, -1, (255, 255, 255), -1)

        return Image.fromarray(img_np)


# Функция для записи результатов в CSV файл
def write_to_csv(file_path, header, rows):
    with open(file_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(header)  # Заголовок
        writer.writerows(rows)  # Данные

# Функция для создания директории, если она не существует
def create_dir(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

def myValidate(model, val_loader, device, epoch_dir, epoch_start_time):
    model.eval()
    y_true = []
    probabilities_list = []

    # write_to_csv(os.path.join(epoch_dir, "val_test.csv"), ["correct", "probability 1", "probability 2"], val_test_data)

    with torch.no_grad():
        for data in val_loader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            
            outputs = model(images)

            _, predicted = torch.max(outputs, 1)
            # print("outputs:", outputs, "\nlol:", lol, "\npredicted:", predicted, "\nlabels:", labels)                    

            probabilities = torch.nn.functional.softmax(outputs, dim=1).cpu().numpy()  # Вычисление вероятностей

            for label, probability in zip(labels.cpu().numpy(), probabilities):
                y_true.append(label)
                probabilities_list.append(probability)  # Добавление вероятностей для каждого образца


    # print("\nprobabilities_list:", probabilities_list)

    rows = []
    for label, probs in zip(y_true, probabilities_list):
        row = [label] + list(probs)  # Преобразование label и probs в один список
        rows.append(row)
    # print(rows)
    
    val_test_csv_path = os.path.join(epoch_dir, "val_test.csv")
    write_to_csv(val_test_csv_path, ["correct", "probability 1", "probability 2"], rows)


    y_pred = np.argmax(probabilities_list, axis=1)  # Получение предсказаний из вероятностей
    # print("y_true:", y_true, "\ny_pred:", y_pred)
    f1 = f1_score(y_true, y_pred, average='weighted')

    epoch_delta_time = time.time() - epoch_start_time  # Calculate the time difference
    print(f"F1 Score (Fold {fold + 1}, Epoch {epoch + 1}, sec {epoch_delta_time}): {f1:.4f}")


def myTestOnTrain(model, train_loader, device, epoch_dir):
    model.eval()
    y_true = []
    probabilities_list = []

    # write_to_csv(os.path.join(epoch_dir, "val_test.csv"), ["correct", "probability 1", "probability 2"], val_test_data)

    with torch.no_grad():
        for data in train_loader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            
            outputs = model(images)

            _, predicted = torch.max(outputs, 1)
            # print("outputs:", outputs, "\nlol:", lol, "\npredicted:", predicted, "\nlabels:", labels)                    

            probabilities = torch.nn.functional.softmax(outputs, dim=1).cpu().numpy()  # Вычисление вероятностей

            for label, probability in zip(labels.cpu().numpy(), probabilities):
                y_true.append(label)
                probabilities_list.append(probability)  # Добавление вероятностей для каждого образца


    # print("\nprobabilities_list:", probabilities_list)

    rows = []
    for label, probs in zip(y_true, probabilities_list):
        row = [label] + list(probs)  # Преобразование label и probs в один список
        rows.append(row)
    # print(rows)
    
    val_test_csv_path = os.path.join(epoch_dir, "train_test.csv")
    write_to_csv(val_test_csv_path, ["correct", "probability 1", "probability 2"], rows)


if __name__ == '__main__':
    print("cuda.is_available " + str(torch.cuda.is_available()))
    # Define data transforms
    transform = transforms.Compose([
        transforms.Resize((400, 400)),
        RandomContoursRemovalTransform(removal_probability=0.4),
        transforms.RandomCrop((224, 224)),
        # transforms.RandomHorizontalFlip(),
        # transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(3),
        # transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        # transforms.RandomAffine(degrees=10, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        # transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Define data directory
    data_dir = "train"
    dataset = torchvision.datasets.ImageFolder(root=data_dir, transform=transform)

    # Determine the number of classes in your dataset
    num_classes = len(os.listdir(data_dir))

    # Define hyperparameters
    num_epochs = 100
    learning_rate = 0.001
    batch_size = 8

    # Define cross-validation strategy (e.g., 5-fold)
    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    for fold, (train_indices, val_indices) in enumerate(kf.split(range(len(dataset)), dataset.targets)):
        fold_dir = f"folds/fold_{fold + 1}"  # Имя папки для фолда
        print(fold_dir)
        create_dir(fold_dir)  # Создание папки для фолда

        # Split the dataset into training and validation sets for this fold
        train_dataset = torch.utils.data.Subset(dataset, train_indices)
        val_dataset = torch.utils.data.Subset(dataset, val_indices)

        # Create data loaders for training and validation
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

        # Define the model
        model = vgg16(weights=VGG16_Weights.IMAGENET1K_V1)
        # model = vgg16()
        model.classifier[3] = nn.Linear(4096, 4096)
        model.classifier[6] = nn.Linear(4096, num_classes)
        
        print(model)

        # Set the device
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model.to(device)

        # Define the loss function and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.6)

        # Training loop
        for epoch in range(num_epochs):
            epoch_dir = os.path.join(fold_dir, f"epoch_{epoch + 1}")  # Имя папки для эпохи
            print(epoch_dir)
            create_dir(epoch_dir)  # Создание папки для эпохи

            epoch_start_time = time.time()  # Capture the start time of the epoch

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

            # Validation loop
            myValidate(model, val_loader, device, epoch_dir, epoch_start_time)

            myTestOnTrain(model, train_loader, device, epoch_dir)

            # Save the model for this fold if needed
            # if epoch % 2 == 0:
            #     torch.save(model.state_dict(), f"model_fold{fold + 1}_{epoch + 1}.pth")

        # Save the model for this fold if needed
        torch.save(model.state_dict(), f"model_fold{fold + 1}.pth")

    input("Press Enter to exit...")
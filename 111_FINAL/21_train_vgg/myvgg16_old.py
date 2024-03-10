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

# Определение класса RandomContoursRemovalTransform
class RandomContoursRemovalTransform(object)
    def __init__(self, removal_probability=0.4)
        self.removal_probability = removal_probability

    def __call__(self, img)
        # Convert PIL image to numpy array
        img_np = np.array(img)

        # Convert RGB to Grayscale
        gray_img = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)

        # Apply threshold using Otsu's method
        _, binary_img = cv2.threshold(gray_img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        # Find contours, remove some, and draw them back onto the RGB image
        contours, _ = cv2.findContours(binary_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        num_contours_to_remove = int(len(contours)  self.removal_probability)
        contours_to_remove = random.sample(contours, num_contours_to_remove)
        cv2.drawContours(img_np, contours_to_remove, -1, (255, 255, 255), -1)

        return Image.fromarray(img_np)


if __name__ == '__main__'
    print(cuda.is_available  + str(torch.cuda.is_available()))
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
    data_dir = train
    dataset = torchvision.datasets.ImageFolder(root=data_dir, transform=transform)

    # # How to show the transformed image
    # # Путь к изображению
    # image_path = trainDYSGR00008.png
    # # Загрузка изображения с помощью PIL
    # image = Image.open(image_path)
    # # Применение transform к изображению
    # transformed_image = transform(image)
    # # Преобразование изображения к типу numpy для визуализации (если нужно)
    # transformed_image_np = transformed_image.numpy()
    # # Визуализация преобразованного изображения
    # plt.imshow(np.transpose(transformed_image_np, (1, 2, 0)))
    # plt.show()


    # Determine the number of classes in your dataset
    num_classes = len(os.listdir(data_dir))

    # Define hyperparameters
    num_epochs = 2
    learning_rate = 0.001
    batch_size = 8

    # Define cross-validation strategy (e.g., 5-fold)
    kf = StratifiedKFold(n_splits=2, shuffle=True, random_state=42)

    for fold, (train_indices, val_indices) in enumerate(kf.split(range(len(dataset)), dataset.targets))
        print(fFold {fold + 1})

        # Split the dataset into training and validation sets for this fold
        train_dataset = torch.utils.data.Subset(dataset, train_indices)
        val_dataset = torch.utils.data.Subset(dataset, val_indices)

        # Create data loaders for training and validation
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

        # Define the model
        model = vgg16(weights=VGG16_Weights.IMAGENET1K_V1)
        # model = vgg16()
        model.classifier[6] = nn.Linear(4096, num_classes)

        # Set the device
        device = torch.device(cuda0 if torch.cuda.is_available() else cpu)
        model.to(device)

        # Define the loss function and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.6)

        # Training loop
        for epoch in range(num_epochs)
            epoch_start_time = time.time()  # Capture the start time of the epoch
            iterations_start_time = time.time()  # Capture the start time of the iteration

            model.train()
            running_loss = 0.0

            for i, data in enumerate(train_loader, 0)
                inputs, labels = data
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()

                if i % 10 == 9
                    iterations_delta_time = time.time() - iterations_start_time  # Calculate the time difference
                    iterations_start_time = time.time()

                    print(f[{epoch + 1}, {i + 1}], {iterations_delta_time}, loss {running_loss  10.3f})

                    running_loss = 0.0

            # Validation loop
            model.eval()
            y_true = []
            y_pred = []

            with torch.no_grad()
                for data in val_loader
                    images, labels = data
                    images, labels = images.to(device), labels.to(device)
                    outputs = model(images)
                    lol, predicted = torch.max(outputs, 1)
                    print(outputs, outputs, nlol, lol, npredicted, predicted)                    

                    y_true.extend(labels.cpu().numpy())
                    y_pred.extend(predicted.cpu().numpy())

            f1 = f1_score(y_true, y_pred, average='weighted')
            epoch_delta_time = time.time() - epoch_start_time  # Calculate the time difference

            print(fF1 Score (Fold {fold + 1}, Epoch {epoch + 1}, sec {epoch_delta_time}) {f1.4f})

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
        for i, v in enumerate(metrics)
            plt.text(i, v + 0.02, f{v.2f}, ha='center', va='bottom')
        plt.show(block=False)

        # После обучения модели и вычисления метрик для текущего fold
        model.eval()
        y_true = []
        y_scores = []  # Список для хранения вероятностей классов

        with torch.no_grad()
            for data in val_loader
                images, labels = data
                images = images.to(device)
                outputs = model(images)
                probabilities = torch.nn.functional.softmax(outputs, dim=1)  # Вычисление вероятностей
                y_true.extend(labels.cpu().numpy())
                y_scores.extend(probabilities[, 1].cpu().numpy())  # Вероятности класса 1

        # Вычисление ROC-кривой и AUC
        fpr, tpr, thresholds = roc_curve(y_true, y_scores)
        roc_auc = auc(fpr, tpr)

        # Построение ROC-кривой
        plt.figure()
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'Receiver Operating Characteristic (Fold {fold + 1})')
        plt.legend(loc='lower right')
        plt.show(block=False)

        Save the model for this fold if needed
        torch.save(model.state_dict(), fmodel_fold{fold + 1}.pth)

    input(Press Enter to exit...)
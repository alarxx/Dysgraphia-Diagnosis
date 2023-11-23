import os
import time
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from sklearn.metrics import precision_score, recall_score, f1_score
from torchvision.models import vgg16, VGG16_Weights
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

# Предположим, что ваш тестовый датасет находится в папке 'dataset/test'
test_data_dir = "test"

# Трансформации для тестового датасета (без аугментаций)
test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Загрузка тестового датасета
test_dataset = datasets.ImageFolder(root=test_data_dir, transform=test_transform)
test_loader = DataLoader(test_dataset, batch_size=24, shuffle=False)

num_classes = len(os.listdir(test_data_dir))

# Загрузка обученной модели
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = vgg16(weights='IMAGENET1K_V1')  # Или ваша кастомная архитектура
model.classifier[6] = nn.Linear(4096, num_classes)  # Корректировка под ваше количество классов
model.load_state_dict(torch.load("w1/model_fold1.pth"))  # Замените 'model.pth' на путь к вашей модели
model.to(device)
model.eval()

# Сбор предсказаний и истинных меток
y_true = []
y_pred = []

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        y_true.extend(labels.cpu().numpy())
        y_pred.extend(predicted.cpu().numpy())

# Преобразование в numpy массивы
y_true = np.array(y_true)
y_pred = np.array(y_pred)

# Расчет метрик
precision = precision_score(y_true, y_pred, average='weighted')
recall = recall_score(y_true, y_pred, average='weighted')
f1 = f1_score(y_true, y_pred, average='weighted')

# Вывод метрик
print(f'Precision: {precision:.4f}')
print(f'Recall: {recall:.4f}')
print(f'F1-Score: {f1:.4f}')



# Сохраняем неверные индексы и пути к файлам
incorrect_files = []
incorrect_indices = [i for i, (pred, true) in enumerate(zip(y_pred, y_true)) if pred != true]

# Отображаем пути к неверно классифицированным изображениям
for incorrect_index in incorrect_indices:
    # Получаем путь к файлу из тестового датасета
    file_path = test_dataset.samples[incorrect_index][0]
    incorrect_files.append(file_path)
    print(f"Incorrect file: {file_path} - Predicted: {y_pred[incorrect_index]}, True: {y_true[incorrect_index]}")




# Определение softmax функции для преобразования логитов в вероятности
def softmax(x):
    return np.exp(x) / np.sum(np.exp(x), axis=1, keepdims=True)

# Сбор вероятностей для ROC-кривой, если задача бинарной классификации
probs = []

# Переопределяем цикл для сбора вероятностей
with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        outputs = model(images)
        sm_outputs = softmax(outputs.cpu().numpy())  # Применяем softmax
        probs.extend(sm_outputs)

probs = np.array(probs)
y_score = probs[:, 1]  # вероятности для класса 1, если у вас бинарная классификация

# Если у вас более двух классов, вы должны вычислить ROC-кривую для каждого класса отдельно

# Расчет precision, recall, f1-score для каждого класса
precision_class = precision_score(y_true, y_pred, average=None)
recall_class = recall_score(y_true, y_pred, average=None)
f1_class = f1_score(y_true, y_pred, average=None)

# Построение столбчатой диаграммы
plt.figure(figsize=(12, 7))
bar_width = 0.25
indices = np.arange(len(precision_class))
plt.bar(indices, precision_class, bar_width, label='Precision')
plt.bar(indices + bar_width, recall_class, bar_width, label='Recall')
plt.bar(indices + 2 * bar_width, f1_class, bar_width, label='F1-Score')

plt.xlabel('Classes')
plt.ylabel('Scores')
plt.xticks(indices + bar_width, [f'Class {i}' for i in range(num_classes)])
plt.legend()
plt.title('Precision, Recall, and F1-Score for Each Class')
plt.show()

# Расчет и построение ROC-кривой
if num_classes == 2:
    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.show()

import os
import shutil
from sklearn.model_selection import train_test_split

def split_dataset(base_dir, train_size=0.8):
    """
    Разделение датасета на обучающую и тестовую выборки.

    :param base_dir: Путь к базовой директории с данными.
    :param train_size: Процент данных, который будет использован для обучения (остальное - для теста).
    """
    # Создание директорий для обучающего и тестового наборов данных
    train_dir = os.path.join(os.path.dirname(base_dir), 'train')
    test_dir = os.path.join(os.path.dirname(base_dir), 'test')
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    classes = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
    for cls in classes:
        os.makedirs(os.path.join(train_dir, cls), exist_ok=True)
        os.makedirs(os.path.join(test_dir, cls), exist_ok=True)

        # Получение списка файлов в каждой папке-классе
        files = [os.path.join(base_dir, cls, f) for f in os.listdir(os.path.join(base_dir, cls))
                 if os.path.isfile(os.path.join(base_dir, cls, f))]

        # Разделение на обучающую и тестовую выборки
        train_files, test_files = train_test_split(files, train_size=train_size)

        # Копирование файлов в соответствующие папки
        for f in train_files:
            shutil.copy(f, os.path.join(train_dir, cls))

        for f in test_files:
            shutil.copy(f, os.path.join(test_dir, cls))

# Использование функции
split_dataset('dataset', train_size=0.8)  # Укажите свой путь и желаемый процент для обучающего набора

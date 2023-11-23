import os
import random
import shutil

# Путь к вашему датасету
dataset_path = 'word_dataset'

# Пропорция разделения (например, 80% для train и 20% для test)
proportion = 0.8

# Папки для train и test
train_path = 'word_train_dataset'
test_path = 'word_test_dataset'

# Создаем папки train и test
os.makedirs(train_path, exist_ok=True)
os.makedirs(test_path, exist_ok=True)

# Перебираем папки внутри датасета
for folder in os.listdir(dataset_path):
    folder_path = os.path.join(dataset_path, folder)
    if os.path.isdir(folder_path):
        files = os.listdir(folder_path)
        random.shuffle(files)
        split_index = int(len(files) * proportion)
        
        # Разделяем файлы в соответствии с пропорцией
        train_files = files[:split_index]
        test_files = files[split_index:]
        
        # Копируем файлы в папки train и test
        for file in train_files:
            src_path = os.path.join(folder_path, file)
            dst_path = os.path.join(train_path, folder, file)
            os.makedirs(os.path.dirname(dst_path), exist_ok=True)
            shutil.copy(src_path, dst_path)
        
        for file in test_files:
            src_path = os.path.join(folder_path, file)
            dst_path = os.path.join(test_path, folder, file)
            os.makedirs(os.path.dirname(dst_path), exist_ok=True)
            shutil.copy(src_path, dst_path)

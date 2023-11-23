import os
import shutil
import random


def remove_folder_if_exists(folder_path):
    if os.path.exists(folder_path):
        shutil.rmtree(folder_path)


def split_dataset_into_train_and_test(base_path_classes, base_output_path, train_size=0.8):
    # Create master folders for the datasets
    validation_folder = os.path.join(base_output_path, 'train')
    test_folder = os.path.join(base_output_path, 'test')

    # Delete existing folders, so that no examples accumulate in the output split
    remove_folder_if_exists(validation_folder)
    remove_folder_if_exists(test_folder)

    os.makedirs(validation_folder, exist_ok=True)
    os.makedirs(test_folder, exist_ok=True)

    # Create subfolders for each class
    for class_name in os.listdir(base_path_classes):
        os.makedirs(os.path.join(validation_folder, class_name), exist_ok=True)
        os.makedirs(os.path.join(test_folder, class_name), exist_ok=True)

    # For each class, randomly split the files and move them around
    for class_name in os.listdir(base_path_classes):
        class_folder = os.path.join(base_path_classes, class_name)
        files = [f for f in os.listdir(class_folder) if f.endswith('.png')]
        random.shuffle(files) # shuffle the files

        # Determine the number of files for the training sample
        train_count = int(len(files) * train_size)

        # Separating files
        train_files = files[:train_count]
        test_files = files[train_count:]

        # Moving files
        for f in train_files:
            shutil.move(os.path.join(class_folder, f), os.path.join(validation_folder, class_name, f))
        for f in test_files:
            shutil.move(os.path.join(class_folder, f), os.path.join(test_folder, class_name, f))


if __name__ == "__main__":
    # Path to dataset folder
    base_path_classes = 'dataset'
    # Path to the folder where the test and validation folders will be located
    base_output_path = ''

    # Call the function to split the dataset
    split_dataset_into_train_and_test(base_path_classes, base_output_path)
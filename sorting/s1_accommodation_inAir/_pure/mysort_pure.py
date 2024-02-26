import os
import shutil
import pandas as pd

"""
    It doesn't translate to images but just divides correctly, 
    I don't know why it might be needed, I'll leave it here.
"""
if __name__ == '__main__':
    excel_path = 'data2_SciRep_pub.xlsx'
    df = pd.read_excel(excel_path)
    id_class_mapping = df.set_index('ID')['diag'].to_dict()

    # Создаем папки для каждого класса
    base_path_classes = 'dataset_pure'
    class_names = set(str(value).strip() for value in id_class_mapping.values())

    for class_name in class_names:
        os.makedirs(os.path.join(base_path_classes, class_name), exist_ok=True)

    # Папка, где находятся исходные файлы .svc
    data_folder_path = 'all_in_one'

    # Проходим по всем файлам .svc и копируем их в соответствующие папки классов
    for file_name in os.listdir(data_folder_path):
        if file_name.endswith('.svc'):
            file_id = file_name.split('.')[0].lstrip('0')  # Удаляем ведущие нули для соответствия с ID в Excel
            class_name = str(id_class_mapping.get(int(file_id), 'Unknown'))  # Преобразуем ID в int для поиска
            if class_name != 'Unknown':
                # Полный путь к исходному файлу
                src_file_path = os.path.join(data_folder_path, file_name)
                # Полный путь к файлу назначения
                dest_file_path = os.path.join(base_path_classes, class_name, file_name)
                # Копируем файл в папку соответствующего класса
                shutil.copy(src_file_path, dest_file_path)
            else:
                print(f"File {file_name} has an Unknown class.")

import os
import pandas as pd
import svc_to_image

if __name__ == '__main__':
    excel_path = 'data2_SciRep_pub.xlsx'
    df = pd.read_excel(excel_path)
    id_class_mapping = df.set_index('ID')['diag'].to_dict()
    print(id_class_mapping)

    base_path_classes = 'dataset'
    class_names = set(str(value).strip() for value in id_class_mapping.values())
    print(class_names)

    for class_name in class_names:
        os.makedirs(os.path.join(base_path_classes, class_name), exist_ok=True)

    data_folder_path = 'all_in_one'

    for file_name in os.listdir(data_folder_path):
       
        if file_name.endswith('.svc'):
            # Assume ID is the first part of the file name before the dot (00006.svc)
            file_id = int(file_name.split('.')[0])  
            # 'Unknown' - if the ID is not found in Excel (should never be)
            class_name = str(id_class_mapping.get(file_id, 'Unknown'))  

            if class_name != 'Unknown':
                input_file_path = os.path.join(data_folder_path, file_name)
                output_file_path = os.path.join(base_path_classes, class_name, file_name.replace('.svc', '.png'))

                svc_to_image.save_pressed_points_as_image(svc_file_path=input_file_path, output_filename=output_file_path)

            else:
                print(file_name + "Unknown")

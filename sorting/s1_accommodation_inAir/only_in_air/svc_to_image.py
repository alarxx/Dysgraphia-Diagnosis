import numpy as np
import matplotlib.pyplot as plt
import cv2

def save_air_movements_as_image(svc_file_path, image_size=400, padding=20, output_filename=""):
    # Загрузка содержимого файла .svc
    with open(svc_file_path, 'r') as file:
        lines = file.readlines()

    # Извлечение данных и сохранение в списках
    x_coords, y_coords, flags = [], [], []
    for line in lines:
        if line[0] != '#':  # Игнорирование строк с комментариями
            data = line.split()
            if len(data) >= 4:  # Убедиться, что строка содержит достаточно данных
                x_coords.append(float(data[0]))
                y_coords.append(float(data[1]))
                flags.append(int(data[3]))

    # Фильтрация координат для сохранения только точек в воздухе
    air_x_coords = [x for x, flag in zip(x_coords, flags) if flag == 0]
    air_y_coords = [y for y, flag in zip(y_coords, flags) if flag == 0]

    # Нормализация координат для их помещения в заданный размер холста с отступами
    if air_x_coords and air_y_coords:  # Проверка наличия точек в воздухе
        max_x, min_x = max(air_x_coords), min(air_x_coords)
        max_y, min_y = max(air_y_coords), min(air_y_coords)

        scale_factor = (image_size - 2 * padding) / max(max_x - min_x, max_y - min_y)
        x_coords = [int(padding + scale_factor * (x - min_x)) for x in air_x_coords]
        y_coords = [int(padding + scale_factor * (y - min_y)) for y in air_y_coords]
        y_coords = [image_size - 1 - y for y in y_coords]  # Зеркальное отражение y-координат

        # Создание пустого белого холста
        canvas = 255 * np.ones((image_size, image_size, 3), dtype=np.uint8)

        # Рисование точек в воздухе на холсте красным цветом
        for x, y in zip(x_coords, y_coords):
            cv2.circle(canvas, (x, y), 1, (0, 0, 255), -1)  # Использование красного цвета

        # Сохранение холста в виде изображения, если указано имя выходного файла
        if output_filename:
            cv2.imwrite(output_filename, canvas)



svc_file_path = "all_in_one/00006.svc"
output_filename = "output_image.png"  # Имя файла для сохранения

save_air_movements_as_image(svc_file_path=svc_file_path, output_filename=output_filename)
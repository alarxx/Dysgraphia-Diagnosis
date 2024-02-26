import numpy as np
import matplotlib.pyplot as plt
import cv2


def save_pressed_points_as_image(svc_file_path, image_size=400, padding=20, output_filename=""):
    # Loading the .svc file content
    with open(svc_file_path, 'r') as file:
        lines = file.readlines()

    # Extracting data and storing in lists
    x_coords, y_coords, flags = [], [], []
    for line in lines:
        if line[0] != '#':  # Ignoring comment lines
            data = line.split()
            if len(data) >= 6:  # Making sure the line contains enough data
                x_coords.append(float(data[0]))
                y_coords.append(float(data[1]))
                flags.append(int(data[3]))  # on-surface or in-air flag

    # Normalizing the coordinates to fit within the specified canvas size with padding
    max_x, min_x = max(x_coords), min(x_coords)
    max_y, min_y = max(y_coords), min(y_coords)

    scale_factor = (image_size - 2 * padding) / max(max_x - min_x, max_y - min_y)
    x_coords = [int(padding + scale_factor * (x - min_x)) for x in x_coords]
    y_coords = [int(padding + scale_factor * (y - min_y)) for y in y_coords]
    y_coords = [image_size - 1 - y for y in y_coords]  # Mirroring the y-coordinates

    # Creating an empty canvas with 3 channels for RGB
    canvas = 255 * np.ones((image_size, image_size, 3), dtype=np.uint8)

    # Plotting the points on the canvas
    for x, y, flag in zip(x_coords, y_coords, flags):
        if flag == 1:  # on-surface points in black
            canvas[y, x] = [0, 0, 0]
        else:  # in-air points in red
            canvas[y, x] = [0, 0, 255]  # Red color in BGR

    # Saving the canvas as an image file if an output filename is provided
    if output_filename:
        cv2.imwrite(output_filename, canvas)


# svc_file_path = "all_in_one/00158.svc"
# output_filename = "output_image.png"  # Имя файла для сохранения

# save_pressed_points_as_image(svc_file_path=svc_file_path, output_filename=output_filename)
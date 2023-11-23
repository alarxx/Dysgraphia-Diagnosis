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
            if len(data) >= 4:  # Making sure the line contains enough data
                x_coords.append(float(data[0]))
                y_coords.append(float(data[1]))
                flags.append(int(data[3]))

    # Filtering coordinates to keep only the pressed points
    pressed_x_coords = [x for x, flag in zip(x_coords, flags) if flag == 1]
    pressed_y_coords = [y for y, flag in zip(y_coords, flags) if flag == 1]

    # Normalizing the coordinates to fit within the specified canvas size with padding
    max_x, min_x = max(pressed_x_coords), min(pressed_x_coords)
    max_y, min_y = max(pressed_y_coords), min(pressed_y_coords)

    scale_factor = (image_size - 2 * padding) / max(max_x - min_x, max_y - min_y)
    x_coords = [int(padding + scale_factor * (x - min_x)) for x in pressed_x_coords]
    y_coords = [int(padding + scale_factor * (y - min_y)) for y in pressed_y_coords]
    y_coords = [image_size - 1 - y for y in y_coords]  # Mirroring the y-coordinates

    # Creating an empty white canvas
    canvas = 255 * np.ones((image_size, image_size), dtype=np.uint8)

    # Plotting the on-surface (pressed) points on the canvas in black
    for x, y in zip(x_coords, y_coords):
        canvas[y, x] = 0  # Coloring the point black

    # Displaying the canvas as an image
    # plt.imshow(canvas, cmap='gray')
    # plt.axis('off')  # Hiding axes ticks and labels
    # plt.show()

    # Saving the canvas as an image file if an output filename is provided
    if output_filename:
        cv2.imwrite(output_filename, canvas)

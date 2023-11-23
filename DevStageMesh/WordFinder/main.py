import cv2
import numpy as np
import os
from matplotlib import pyplot as plt

def place_contour_widthwise_in_center(binary_image, contour, output_size=(400, 400)):
    # Get the bounding rectangle for the contour
    x, y, w, h = cv2.boundingRect(contour)
    # Crop the contour out of the binary image
    cropped_image = binary_image[y:y+h, x:x+w]

    # Calculate the ratio by which the image needs to be resized
    ratio = output_size[0] / w
    resized_w = output_size[0]
    resized_h = int(h * ratio)

    # If the resized height is greater than the output height, adjust the ratio
    if resized_h > output_size[1]:
        ratio = output_size[1] / h
        resized_h = output_size[1]
        resized_w = int(w * ratio)

    # Resize the image
    cropped_image = cv2.resize(cropped_image, (resized_w, resized_h), interpolation=cv2.INTER_AREA)

    # Create a new black image of the specified size
    centered_image = np.zeros(output_size, dtype=np.uint8)

    # Calculate the position to place the cropped image so it's centered heightwise
    y_center = (output_size[1] - resized_h) // 2

    # Place the cropped image in the center of the new image heightwise
    centered_image[y_center:y_center+resized_h, (output_size[0] - resized_w) // 2:(output_size[0] - resized_w) // 2 + resized_w] = cropped_image

    return centered_image


def process_image(image_path, output_folder):
    # Read the image in grayscale
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    # Apply binary thresholding with Otsu's method
    _, binary_image = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    # Apply morphological closing to fill small holes in the text
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    closed_image = cv2.morphologyEx(binary_image, cv2.MORPH_CLOSE, kernel)
    # Find contours
    contours, _ = cv2.findContours(closed_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # Sort contours by area, in descending order
    largest_contours = sorted(contours, key=cv2.contourArea, reverse=True)

    # Process each contour
    for i, contour in enumerate(largest_contours):
        # Check if the contour area is large enough
        if cv2.contourArea(contour) > 100:
            # Place the contour widthwise in the center
            centered_image = place_contour_widthwise_in_center(binary_image, contour)
            # Save the image to the output folder
            output_image_path = os.path.join(output_folder, f'word_{i}_{os.path.basename(image_path)}')
            cv2.imwrite(output_image_path, centered_image)

if __name__ == "__main__":
    # Replace with the path to your folder with input images
    # input_folder = 'dataset/DYSGR'  # For example: 'dataset/0'
    input_folder = 'dataset/0'  # For example: 'dataset/0'
    # Replace with the path to the folder where you want to save the results
    # output_folder = 'word_dataset/DYSGR'  # For example: 'words_dataset/0'
    output_folder = 'word_dataset/0'  # For example: 'words_dataset/0'

    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Process each image in the input folder
    for filename in os.listdir(input_folder):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(input_folder, filename)
            process_image(image_path, output_folder)

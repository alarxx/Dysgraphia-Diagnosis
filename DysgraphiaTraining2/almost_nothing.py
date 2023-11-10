import os
import numpy as np
from PIL import Image
import cv2


def is_image_almost_empty(image, area_threshold):
    """
    Determine if an image is 'almost empty' based on the area of the largest
    bounded rectangle around the black regions.
    """
    # Convert PIL Image to numpy array
    image_np = np.array(image)

    # If image is grayscale, convert it to RGB
    if len(image_np.shape) == 2:
        image_np = cv2.cvtColor(image_np, cv2.COLOR_GRAY2RGB)

    # Convert image to grayscale and then to binary
    gray_image = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
    _, binary_image = cv2.threshold(gray_image, 30, 255, cv2.THRESH_BINARY_INV)
    
    # Find contours of the black regions
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Find the largest bounding rectangle area
    largest_area = 0
    for contour in contours:
        _, _, w, h = cv2.boundingRect(contour)
        largest_area = max(largest_area, w * h)
    
    # Determine if the image is almost empty
    return largest_area < area_threshold


def delete_almost_empty_images(directory_path, area_threshold):
    """
    Recursively delete 'almost empty' images in the directory.
    """
    for root, _, files in os.walk(directory_path):
        for file in files:
            if file.lower().endswith('.png'):
                full_path = os.path.join(root, file)
                with Image.open(full_path) as img:
                    if is_image_almost_empty(img, area_threshold):
                        # Delete the image if it is almost empty
                        os.remove(full_path)
                        print(f"Deleted 'almost empty' image: {full_path}")

directory_path = "dataset"
area_threshold = 300
delete_almost_empty_images(directory_path, area_threshold)

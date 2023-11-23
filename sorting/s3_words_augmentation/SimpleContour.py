import random
import cv2
import numpy as np
from matplotlib import pyplot as plt

# Load the image
image_path = '00032.png'  # Замените на путь к вашему изображению
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# Apply threshold to get binary image
_, binary_image = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

# Find contours
contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Draw rectangles around each contour on the original image
contoured_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)  # Convert to BGR for colored drawing
filled_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)  # Convert to BGR for colored drawing

# Выберем случайное подмножество контуров для отрисовки
percentage_to_delete = 0.1  # Процент контуров для отрисовки, например, 50%
selected_indices = random.sample(range(len(contours)), int(len(contours) * percentage_to_delete))
selected_contours = [contours[i] for i in selected_indices]

for contour in selected_contours:
    # Get bounding box for each contour
    x, y, w, h = cv2.boundingRect(contour)
    # Draw the rectangle
    cv2.rectangle(contoured_image, (x, y), (x+w, y+h), (0, 255, 0), 2)

cv2.drawContours(filled_image, selected_contours, -1, (255, 255, 255), -1)

# Display the original image and the image with contours
plt.figure(figsize=(15, 5))
plt.subplot(1, 3, 1)
plt.title('Original Image')
plt.imshow(image, cmap='gray')
plt.axis('off')

plt.subplot(1, 3, 2)
plt.title('Image with Contours (random:' + str(int(percentage_to_delete*100)) + '%)')
plt.imshow(contoured_image)
plt.axis('off')

plt.subplot(1, 3, 3)
plt.title('Filled Image')
plt.imshow(filled_image)
plt.axis('off')

plt.show()

import cv2
import numpy as np
from matplotlib import pyplot as plt

# Function to place the contour in the center of a new image
def place_contour_in_center_binary(binary_image, contour, output_size=(224, 224)):
    """
    This function takes a binary image and a contour, crops the contour from the image,
    and places it in the center of a new black image of the specified output size.
    The binary image should have white text (255) on a black background (0).
    """
    x, y, w, h = cv2.boundingRect(contour)  # Get the bounding rectangle for the contour
    cropped_image = binary_image[y:y+h, x:x+w]  # Crop the contour out of the binary image

    # Create a new black image of the specified size
    centered_image = np.zeros(output_size, dtype=np.uint8)

    # Resize cropped_image if it's larger than output_size, maintaining the aspect ratio
    if w > output_size[0] or h > output_size[1]:
        # Find the ratio by which the image needs to be resized
        ratio = min(output_size[0] / w, output_size[1] / h)
        resized_w = int(w * ratio)
        resized_h = int(h * ratio)
        # Resize the image
        cropped_image = cv2.resize(cropped_image, (resized_w, resized_h), interpolation=cv2.INTER_AREA)

    # Update the dimensions after resizing
    h, w = cropped_image.shape

    # Create a new black image of the specified size
    centered_image = np.zeros(output_size, dtype=np.uint8)

    # Calculate the position to place the cropped image so it's centered
    x_center = (output_size[0] - w) // 2
    y_center = (output_size[1] - h) // 2

    # Place the cropped image in the center of the new image
    centered_image[y_center:y_center+h, x_center:x_center+w] = cropped_image

    return centered_image


# if __name__ == "__main__":
#     # Load the image and create a binary version
#     image_path = '00032.png'  # Replace with your image path
#     image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
#     _, binary_image = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

#     # Define the structuring element for morphological operations
#     kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

#     # Apply morphological closing
#     closed_image = cv2.morphologyEx(binary_image, cv2.MORPH_CLOSE, kernel)

#     plt.imshow(closed_image, cmap='gray')
#     plt.show()


#     # Find contours on the closed image
#     contours, _ = cv2.findContours(closed_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

#     # Sort the contours by area and display them one by one
#     largest_contours = sorted(contours, key=cv2.contourArea, reverse=True)

#     # Display all contours one by one
#     for i, contour in enumerate(largest_contours):
#         if cv2.contourArea(contour) > 4:  # Consider contours with area larger than an arbitrary threshold
#             centered_image = place_contour_in_center_binary(binary_image, contour)
            
#             # Display the centered contour image
#             plt.figure(figsize=(5, 5))
#             plt.imshow(centered_image, cmap='gray')
#             plt.title(f'Centered Contour #{i+1}')
#             plt.axis('off')
#             plt.show()

if __name__ == "__main__":
    # Load the image and create a binary version
    image_path = '00032.png'
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    _, binary_image = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Define the structuring element for morphological operations
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

    # Apply morphological closing
    closed_image = cv2.morphologyEx(binary_image, cv2.MORPH_CLOSE, kernel)

    # Find contours on the closed image
    contours, _ = cv2.findContours(closed_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Sort the contours by area
    largest_contours = sorted(contours, key=cv2.contourArea, reverse=True)

    for i, contour in enumerate(largest_contours):
        if cv2.contourArea(contour) > 4:  # Consider contours with area larger than an arbitrary threshold
            # Convert the original image to BGR for color drawing
            image_with_rectangle = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

            # Get the bounding rectangle for the contour
            x, y, w, h = cv2.boundingRect(contour)

            # Draw the bounding rectangle on the original image with a green color
            cv2.rectangle(image_with_rectangle, (x, y), (x+w, y+h), (0, 255, 0), 2)

            # Place the contour in the center of a new 224x224 image
            centered_image = place_contour_in_center_binary(binary_image, contour)

            # Plot the results
            plt.figure(figsize=(15, 5))

            # After morphEx
            plt.subplot(1, 3, 1)
            plt.imshow(closed_image, cmap='gray')
            plt.title('After Morphological Closing')
            plt.axis('off')

            # Original image with bounding rectangle
            plt.subplot(1, 3, 2)
            plt.imshow(cv2.cvtColor(image_with_rectangle, cv2.COLOR_BGR2RGB))
            plt.title(f'Original Image with Boundary of Word #{i+1}')
            plt.axis('off')

            # Centered word in 224x224 image
            plt.subplot(1, 3, 3)
            plt.imshow(centered_image, cmap='gray')
            plt.title(f'224x224 Image with Word #{i+1}')
            plt.axis('off')

            # Display the plot
            plt.show()
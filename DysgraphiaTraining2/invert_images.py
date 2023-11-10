import os
from PIL import Image, ImageOps

def invert_images_in_directory(directory_path):
    """
    Recursively inverts all PNG images in the directory that start with 'word'.
    """
    for root, dirs, files in os.walk(directory_path):
        for file in files:
            if file.lower().endswith('.png') and file.startswith('word'):
                full_path = os.path.join(root, file)
                with Image.open(full_path) as img:
                    # Assuming the images are in mode 'L' for black and white
                    if img.mode == 'L':
                        inverted_image = ImageOps.invert(img)
                        inverted_image.save(full_path)
                        print(f"Inverted: {full_path}")
                    else:
                        print(f"Image {full_path} is not in mode 'L' (black and white).")

# Replace 'your_directory_path' with the path to the directory containing your dataset
directory_path = "dataset"
invert_images_in_directory(directory_path)

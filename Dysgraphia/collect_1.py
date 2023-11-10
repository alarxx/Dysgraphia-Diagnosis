import os
import shutil

def collect_files(src_dir, dest_dir):
    """
    This function collects all files from subdirectories in src_dir and copies them to dest_dir
    
    :param src_dir: str, source directory path
    :param dest_dir: str, destination directory path
    """
    # If destination directory doesn't exist, create it
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)
        
    # Walk through source directory
    for root, dirs, files in os.walk(src_dir):
        for file in files:
            file_path = os.path.join(root, file)
            # Copy each file to the destination directory
            shutil.copy2(file_path, dest_dir)

# Example usage:
src_dir = "data"  # Update this path
dest_dir = "all_in_one"  # Update this path

collect_files(src_dir, dest_dir)
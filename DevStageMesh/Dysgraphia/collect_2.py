import os
import shutil
import re

def rename_and_collect_files(src_dir, dest_dir):
    """
    This function renames files by extracting the user number, excludes specific files,
    and copies the remaining files to dest_dir
    
    :param src_dir: str, source directory path
    :param dest_dir: str, destination directory path
    """
    # If destination directory doesn't exist, create it
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)
        
    # Regular expression to match filenames like u00006s00001_hw00001.svc
    pattern = re.compile(r"u\d+s\d+_hw\d+.svc$")
        
    # Walk through source directory
    for root, dirs, files in os.walk(src_dir):
        for file in files:
            # Check if the file matches the pattern
            if pattern.match(file):
                # Extract user number from the filename
                user_number = file.split('u')[1].split('s')[0]
                
                # Create new filename
                new_filename = f"{user_number}.svc"
                
                # Path to copy the file to
                dest_file_path = os.path.join(dest_dir, new_filename)
                
                # Copy and rename each file to the destination directory
                shutil.copy2(os.path.join(root, file), dest_file_path)

# Example usage:
src_dir = "data"  # Update this path
dest_dir = "all_in_one"  # Update this path

rename_and_collect_files(src_dir, dest_dir)

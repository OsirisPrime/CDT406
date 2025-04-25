import os
import re
import shutil

def copy_and_sort_files(flat_folder, dst_root):
    pattern = re.compile(r'^P([1-9]|1[0-9]|2[0-8])C([1-3])S[1-4]M6F1O1')

    for file in os.listdir(flat_folder):
        name, ext = os.path.splitext(file)
        match = pattern.fullmatch(name)
        if match:
            p = int(match.group(1))
            c = int(match.group(2))

            # Create subject/cycle folder structure
            dst_folder = os.path.join(dst_root, f"subject #{p}", f"cycle #{c}")
            os.makedirs(dst_folder, exist_ok=True)

            src_path = os.path.join(flat_folder, file)
            dst_path = os.path.join(dst_folder, file)

            shutil.copy2(src_path, dst_path)
            print(f"Copied: {src_path} → {dst_path}")

# Example usage:
source_folder = r"C:\Users\kimsv\OneDrive - Mälardalens universitet\Desktop\WyoFlex_Dataset\VOLTAGE DATA"
destination_folder = r"C:\Users\kimsv\OneDrive - Mälardalens universitet\Desktop\Dataset Training"

copy_and_sort_files(source_folder, destination_folder)

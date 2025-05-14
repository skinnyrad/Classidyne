import os
from PIL import Image

def verify_and_remove_images(folder_path):
    corrupted_images = []
    
    for root, _, files in os.walk(folder_path):  # Recursively walk through the directory
        for file in files:
            file_path = os.path.join(root, file)
            try:
                with Image.open(file_path) as img:
                    img.verify()  # Verify the image file without fully opening it
            except (IOError, SyntaxError):
                corrupted_images.append(file_path)
                print(f"Corrupted image: {file_path}")
                
                try:
                    os.remove(file_path)  # Remove the corrupted file
                    print(f"Deleted corrupted image: {file_path}")
                except Exception as removal_error:
                    print(f"Failed to delete {file_path}: {removal_error}")
    
    if corrupted_images:
        print("\nSummary of removed corrupted images:")
        for img in corrupted_images:
            print(img)
    else:
        print("No corrupted images found!")

# Specify the folder to check
folder_to_check = './datasets'
verify_and_remove_images(folder_to_check)


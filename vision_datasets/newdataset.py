import json
import shutil
import os

# Specify the source and destination paths
json_file_path = 'functioncalls.json'
destination_folder = 'G:/images/fnimages/'

# Create the destination folder if it doesn't exist
os.makedirs(destination_folder, exist_ok=True)

# Function to copy the image files and JSON file
def copy_files(json_path, dest_folder):
    with open(json_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    
    for item in data:
        image_path = item.get("image_path")
        if image_path:
            # Copy each image to the new folder
            shutil.copy(image_path, dest_folder)
    
    # Copy the JSON file itself to the new folder
    shutil.copy(json_path, dest_folder)

# Execute the function
copy_files(json_file_path, destination_folder)

print("Files have been copied successfully.")
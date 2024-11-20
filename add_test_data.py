import os
import shutil
import random

# Define the source dataset (where the images are originally stored)
source_dataset_dir = 'PBC_dataset/train'

# Define the target test directory (where the images should be moved to)
test_data_dir = 'test'

# List of class names (same as your training classes)
class_names = ['basophil', 'eosinophil', 'erythroblast', 'ig', 'lymphocyte', 'monocyte', 'neutrophil', 'platelet']

# Number of images to pick per class for the test set
num_images_per_class = 10

# Create the test directory if it doesn't exist
if not os.path.exists(test_data_dir):
    os.makedirs(test_data_dir)

# Iterate through each class and pick images
for class_name in class_names:
    # Define source folder for the class
    class_folder = os.path.join(source_dataset_dir, class_name)
    
    # If class folder exists in source dataset
    if os.path.exists(class_folder):
        # Get a list of all image files in the class folder
        image_files = [f for f in os.listdir(class_folder) if f.endswith(('jpg', 'png', 'jpeg'))]
        
        # Pick a subset of images for the test set
        selected_images = random.sample(image_files, min(num_images_per_class, len(image_files)))
        
        # Create the subfolder for the class in the test directory
        class_test_folder = os.path.join(test_data_dir, class_name)
        if not os.path.exists(class_test_folder):
            os.makedirs(class_test_folder)
        
        # Move selected images to the test directory
        for image in selected_images:
            source_path = os.path.join(class_folder, image)
            target_path = os.path.join(class_test_folder, image)
            shutil.move(source_path, target_path)
            print(f'Moved {image} to {class_test_folder}')
    else:
        print(f'Class folder {class_name} not found in source dataset.')

print('Image selection and movement to test directory complete.')

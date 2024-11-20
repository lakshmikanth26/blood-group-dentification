import os
import shutil
import random

def create_validation_split(train_dir, val_dir, split_ratio=0.2):
    for class_name in os.listdir(train_dir):
        class_path = os.path.join(train_dir, class_name)
        if not os.path.isdir(class_path):
            continue

        images = os.listdir(class_path)
        random.shuffle(images)
        val_count = int(len(images) * split_ratio)

        # Create validation class folder
        val_class_path = os.path.join(val_dir, class_name)
        os.makedirs(val_class_path, exist_ok=True)

        # Move images to validation folder
        for img in images[:val_count]:
            shutil.move(os.path.join(class_path, img), os.path.join(val_class_path, img))

# Define paths
train_dir = "PBC_dataset/train"
val_dir = "PBC_dataset/val"

# Create validation set
create_validation_split(train_dir, val_dir)

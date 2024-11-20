from PIL import Image
import os

def check_images(directory):
    for class_name in os.listdir(directory):
        class_dir = os.path.join(directory, class_name)
        if os.path.isdir(class_dir):
            for img_name in os.listdir(class_dir):
                img_path = os.path.join(class_dir, img_name)
                try:
                    img = Image.open(img_path)
                    img.verify()  # Verify if image is corrupted
                except (IOError, SyntaxError):
                    print(f"Corrupted image: {img_path}")

check_images('PBC_dataset/train')
check_images('PBC_dataset/val')

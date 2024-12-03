import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
# from tensorflow import keras
from keras._tf_keras.keras.preprocessing.image import ImageDataGenerator

IMG_SIZE = (224, 224) 
BASE_DIR = "./dataset/images" 
OUTPUT_DIR = "processed_data"

def load_images_from_folders(base_dir, img_size):
    """
    Loads images and assigns labels based on folder names.
    """
    print("Loading dataset from folders...")
    images, labels = [], []

    folder_label_map = {
        "eligible": 1,
        "not_eligible": 0
    }

    for folder_name, label in folder_label_map.items():
        folder_path = os.path.join(base_dir, folder_name)
        if not os.path.exists(folder_path):
            print(f"Warning: Folder '{folder_name}' not found in '{base_dir}'. Skipping...")
            continue

        for file_name in os.listdir(folder_path):
            file_path = os.path.join(folder_path, file_name)
            if file_name.endswith((".jpg", ".jpeg", ".png")):
                img = cv2.imread(file_path)
                img = cv2.resize(img, img_size)
                images.append(img)
                labels.append(label)

    images = np.array(images, dtype="float32") / 255.0 
    labels = np.array(labels)

    print(f"Loaded {len(images)} images with corresponding labels.")
    return images, labels

def augment_data(images, labels):
    """
    Augments data using ImageDataGenerator for better model performance.
    """
    print("Augmenting data...")
    datagen = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode="nearest",
        brightness_range=[0.8, 1.2]
    )

    return datagen.flow(images, labels, batch_size=32)

if __name__ == "__main__":
    images, labels = load_images_from_folders(BASE_DIR, IMG_SIZE)

    train_x, temp_x, train_y, temp_y = train_test_split(images, labels, test_size=0.3, random_state=42)
    val_x, test_x, val_y, test_y = train_test_split(temp_x, temp_y, test_size=0.5, random_state=42)

    print("Data split completed:")
    print(f"Train set: {len(train_x)} samples")
    print(f"Validation set: {len(val_x)} samples")
    print(f"Test set: {len(test_x)} samples")

    train_generator = augment_data(train_x, train_y)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    np.save(os.path.join(OUTPUT_DIR, "train_x.npy"), train_x)
    np.save(os.path.join(OUTPUT_DIR, "train_y.npy"), train_y)
    np.save(os.path.join(OUTPUT_DIR, "val_x.npy"), val_x)
    np.save(os.path.join(OUTPUT_DIR, "val_y.npy"), val_y)
    np.save(os.path.join(OUTPUT_DIR, "test_x.npy"), test_x)
    np.save(os.path.join(OUTPUT_DIR, "test_y.npy"), test_y)

    print("Data preprocessing completed. Processed data saved.")

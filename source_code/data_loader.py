import os
import cv2
import imghdr
import logging
import pandas as pd
from sklearn.model_selection import train_test_split
from concurrent.futures import ThreadPoolExecutor
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Setup logging for image validation
logging.basicConfig(filename='logs/image_validation.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def check_image(image_path, valid_exts):
    """
    Validates the image format and content. Removes invalid images.
    """
    try:
        img = cv2.imread(image_path)
        tip = imghdr.what(image_path)
        if tip not in valid_exts or img is None:
            logging.warning(f"Removing invalid image: {image_path}")
            os.remove(image_path)
    except Exception as e:
        logging.error(f"Error processing image {image_path}: {str(e)}")


def process_directory(data_dir, valid_exts=['jpeg', 'jpg', 'bmp', 'png', 'gif']):
    """
    Validates and removes invalid images in a directory.
    """
    with ThreadPoolExecutor() as executor:
        for class_dir in os.listdir(data_dir):
            class_path = os.path.join(data_dir, class_dir)
            if os.path.isdir(class_path):
                images = [os.path.join(class_path, img) for img in os.listdir(class_path)]
                executor.map(lambda img: check_image(img, valid_exts), images)


def load_data(data_dir):
    """
    Loads images and labels from a directory and creates a DataFrame.
    """
    image_paths, labels = [], []
    for class_name in os.listdir(data_dir):
        class_dir = os.path.join(data_dir, class_name)
        if os.path.isdir(class_dir):
            for img_name in os.listdir(class_dir):
                img_path = os.path.join(class_dir, img_name)
                image_paths.append(img_path)
                labels.append(class_name)
    df = pd.DataFrame({'filename': image_paths, 'label': labels})
    return df


def prepare_data(data_dir):
    """
    Prepares the dataset by splitting into train, validation, and test sets.
    """
    df = load_data(data_dir)
    train_df, test_df = train_test_split(df, test_size=0.2, stratify=df['label'], random_state=42)
    train_df, val_df = train_test_split(train_df, test_size=0.15, stratify=train_df['label'], random_state=42)
    return train_df, val_df, test_df


def create_image_generators(train_df, val_df, test_df):
    """
    Creates ImageDataGenerators for training, validation, and test sets.
    """
    datagen = ImageDataGenerator(rescale=1./255)

    train_data = datagen.flow_from_dataframe(train_df, x_col='filename', y_col='label', target_size=(256, 256), class_mode='binary', batch_size=32)
    val_data = datagen.flow_from_dataframe(val_df, x_col='filename', y_col='label', target_size=(256, 256), class_mode='binary', batch_size=32)
    test_data = datagen.flow_from_dataframe(test_df, x_col='filename', y_col='label', target_size=(256, 256), class_mode='binary', batch_size=32, shuffle=False)

    return train_data, val_data, test_data

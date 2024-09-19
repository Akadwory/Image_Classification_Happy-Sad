import tensorflow as tf
import os
import cv2
import imghdr
from matplotlib import pyplot as plt
import numpy as np
import math
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard
from tensorflow.keras.metrics import Precision, Recall, BinaryAccuracy
import logging
from concurrent.futures import ThreadPoolExecutor

class EmotionClassifier:

    def __init__(self, data_dir, img_size=(256, 256), batch_size=32, epochs=20, log_dir="logs"):
        """
        Initialize the classifier with paths and hyperparameters.
        """
        self.data_dir = data_dir
        self.img_size = img_size
        self.batch_size = batch_size
        self.epochs = epochs
        self.log_dir = log_dir
        self.model = None

        # Prepare for TensorBoard logging
        self.tensorboard_callback = TensorBoard(log_dir=self.log_dir)

        # Logging setup
        logging.basicConfig(filename='image_validation.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    def check_image(self, image_path):
        """
        Function to check if an image is valid. If invalid, it removes the image.
        Logs any issues encountered.
        """
        image_exts = ['jpeg', 'jpg', 'bmp', 'png', 'gif']
        try:
            img = cv2.imread(image_path)
            tip = imghdr.what(image_path)

            if tip not in image_exts:
                logging.warning(f'Invalid image format for {image_path}. Removing file.')
                os.remove(image_path)

            if img is None:
                logging.warning(f'Image could not be read by OpenCV {image_path}. Removing file.')
                os.remove(image_path)

        except FileNotFoundError:
            logging.error(f'File not found: {image_path}')
        except OSError as e:
            logging.error(f'Error reading file {image_path}: {e}')
        except Exception as e:
            logging.error(f'Unexpected error with {image_path}: {e}')

    def process_directory(self):
        """
        Process the dataset directory to remove invalid images.
        """
        class_dirs = [os.path.join(self.data_dir, class_dir) for class_dir in os.listdir(self.data_dir)]
        images_to_check = []

        for class_dir in class_dirs:
            if os.path.isdir(class_dir):  # Only process directories (class folders)
                images_to_check.extend([os.path.join(class_dir, image) for image in os.listdir(class_dir)])

        with ThreadPoolExecutor() as executor:
            executor.map(self.check_image, images_to_check)

    def load_data(self):
        """
        Load the dataset and return train, validation, and test sets using ImageDataGenerator.
        """
        # Gather all image file paths and labels
        image_paths, labels = [], []
        for class_name in os.listdir(self.data_dir):
            class_dir = os.path.join(self.data_dir, class_name)
            if os.path.isdir(class_dir):
                for img_name in os.listdir(class_dir):
                    img_path = os.path.join(class_dir, img_name)
                    image_paths.append(img_path)
                    labels.append(class_name)

        # Create a pandas DataFrame
        df = pd.DataFrame({'filename': image_paths, 'label': labels})

        # Split the data into train, validation, and test sets
        train_df, test_df = train_test_split(df, test_size=0.2, stratify=df['label'], random_state=42)
        train_df, val_df = train_test_split(train_df, test_size=0.15, stratify=train_df['label'], random_state=42)

        # Create ImageDataGenerators
        datagen = ImageDataGenerator(rescale=1./255)
        
        train_data = datagen.flow_from_dataframe(
            train_df,
            x_col='filename',
            y_col='label',
            target_size=self.img_size,
            batch_size=self.batch_size,
            class_mode='binary',
            shuffle=True
        )
        
        val_data = ImageDataGenerator(rescale=1./255).flow_from_dataframe(
            val_df,
            x_col='filename',
            y_col='label',
            target_size=self.img_size,
            batch_size=self.batch_size,
            class_mode='binary',
            shuffle=True
        )
        
        test_data = ImageDataGenerator(rescale=1./255).flow_from_dataframe(
            test_df,
            x_col='filename',
            y_col='label',
            target_size=self.img_size,
            batch_size=self.batch_size,
            class_mode='binary',
            shuffle=False
        )

        return train_data, val_data, test_data

    def build_model(self):
        """
        Define the CNN model architecture.
        """
        model = Sequential([
            Conv2D(32, (3, 3), activation='relu', input_shape=(*self.img_size, 3)),
            MaxPooling2D(),
            Conv2D(64, (3, 3), activation='relu'),
            MaxPooling2D(),
            Conv2D(128, (3, 3), activation='relu'),
            MaxPooling2D(),
            Flatten(),
            Dense(512, activation='relu'),
            Dropout(0.5),
            Dense(1, activation='sigmoid')
        ])

        model.compile(optimizer='adam', loss=tf.losses.BinaryCrossentropy(), metrics=['accuracy'])
        self.model = model
        return model

    def train(self, train_data, val_data):
        """
        Train the CNN model.
        """
        early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

        history = self.model.fit(
            train_data,
            validation_data=val_data,
            epochs=self.epochs,
            callbacks=[self.tensorboard_callback, early_stopping]
        )
        return history

    def evaluate(self, test_data):
        """
        Evaluate the model on test data and calculate precision, recall, and accuracy.
        """
        pre = Precision()
        re = Recall()
        acc = BinaryAccuracy()

        for batch in test_data:
            X, y = batch
            yhat = self.model.predict(X)
            pre.update_state(y, yhat)
            re.update_state(y, yhat)
            acc.update_state(y, yhat)

        print(f'Precision: {pre.result().numpy()}, Recall: {re.result().numpy()}, Accuracy: {acc.result().numpy()}')

    def predict(self, img_path):
        """
        Predict the class of a given image.
        """
        img = cv2.imread(img_path)
        resize = tf.image.resize(img, self.img_size)
        yhat = self.model.predict(np.expand_dims(resize/255, 0))

        if yhat > 0.5:
            print("Predicted class is Sad")
        else:
            print("Predicted class is Happy")

    def plot_metrics(self, history):
        """
        Plot accuracy and loss graphs.
        """
        fig = plt.figure()
        plt.plot(history.history['loss'], color='teal', label='loss')
        plt.plot(history.history['val_loss'], color='orange', label='val_loss')
        fig.suptitle('Loss', fontsize=20)
        plt.legend(loc="upper left")
        plt.show()

        fig = plt.figure()
        plt.plot(history.history['accuracy'], color='teal', label='accuracy')
        plt.plot(history.history['val_accuracy'], color='orange', label='val_accuracy')
        fig.suptitle('Accuracy', fontsize=20)
        plt.legend(loc="upper left")
        plt.show()

# Example usage:
# emotion_classifier = EmotionClassifier(data_dir='/path/to/data')
# emotion_classifier.process_directory()
# train_data, val_data, test_data = emotion_classifier.load_data()
# model = emotion_classifier.build_model()
# history = emotion_classifier.train(train_data, val_data)
# emotion_classifier.evaluate(test_data)
# emotion_classifier.plot_metrics(history)
# emotion_classifier.predict('/path/to/image.png')

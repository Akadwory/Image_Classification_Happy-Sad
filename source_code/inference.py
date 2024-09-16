import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt

def load_and_preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(256, 256))
    img_array = image.img_to_array(img) / 255.0
    return np.expand_dims(img_array, axis=0)

def predict_image(model_path, img_path):
    model = tf.keras.models.load_model(model_path)
    img = load_and_preprocess_image(img_path)
    
    prediction = model.predict(img)[0][0]
    if prediction > 0.5:
        print(f"Predicted class is 'Sad'")
    else:
        print(f"Predicted class is 'Happy'")

    plt.imshow(image.load_img(img_path))
    plt.title(f'Prediction: {"Sad" if prediction > 0.5 else "Happy"}')
    plt.show()

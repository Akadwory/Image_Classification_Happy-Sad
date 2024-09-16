import tensorflow as tf
from src.model import build_model
from src.data_loader import create_image_generators, prepare_data

def train_model(data_dir, logdir='logs', model_path='models/cnn_model.h5'):
    """
    Trains the CNN model using the prepared dataset.
    """
    train_df, val_df, test_df = prepare_data(data_dir)
    train_data, val_data, test_data = create_image_generators(train_df, val_df, test_df)

    model = build_model()
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)

    model.fit(train_data, validation_data=val_data, epochs=20, callbacks=[tensorboard_callback])

    model.save(model_path)
    print("Model saved to:", model_path)

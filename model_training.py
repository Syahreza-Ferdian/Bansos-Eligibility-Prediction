import os
import numpy as np
from keras._tf_keras.keras.models import Sequential
from keras._tf_keras.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras._tf_keras.keras.callbacks import EarlyStopping, ModelCheckpoint
from keras._tf_keras.keras.optimizers import Adam

PROCESSED_DATA_DIR = "processed_data"
MODEL_SAVE_PATH = "models/bansos_model_ai.keras"
IMG_SIZE = (224, 224, 3) 

def load_processed_data(data_dir):
    """
    Load processed data saved as .npy files.
    """
    print("Loading processed data...")
    train_x = np.load(os.path.join(data_dir, "train_x.npy"))
    train_y = np.load(os.path.join(data_dir, "train_y.npy"))
    val_x = np.load(os.path.join(data_dir, "val_x.npy"))
    val_y = np.load(os.path.join(data_dir, "val_y.npy"))
    test_x = np.load(os.path.join(data_dir, "test_x.npy"))
    test_y = np.load(os.path.join(data_dir, "test_y.npy"))

    print("Data successfully loaded.")
    return train_x, train_y, val_x, val_y, test_x, test_y

def build_model(img_size):
    """
    Build a Convolutional Neural Network (CNN) model for classification.
    """
    print("Building the CNN model...")
    model = Sequential()

    model.add(Conv2D(32, (3, 3), activation="relu", input_shape=img_size))
    model.add(MaxPooling2D((2, 2)))

    model.add(Conv2D(64, (3, 3), activation="relu"))
    model.add(MaxPooling2D((2, 2)))

    model.add(Conv2D(128, (3, 3), activation="relu"))
    model.add(MaxPooling2D((2, 2)))

    model.add(Flatten())
    model.add(Dense(128, activation="relu"))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation="sigmoid"))  

    model.compile(optimizer=Adam(learning_rate=0.001), 
                  loss="binary_crossentropy", 
                  metrics=["accuracy"])
    print("Model built and compiled successfully.")
    return model

def train_model(model, train_x, train_y, val_x, val_y):
    """
    Train the CNN model with the given data.
    """
    print("Starting model training...")
    os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)

    early_stopping = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)
    model_checkpoint = ModelCheckpoint(MODEL_SAVE_PATH, save_best_only=True, monitor="val_loss")

    class_weight = {0: 1., 1: 10.}  
    history = model.fit(
        train_x, train_y,
        validation_data=(val_x, val_y),
        epochs=20,
        batch_size=32,
        class_weight=class_weight,
        callbacks=[early_stopping, model_checkpoint]
    )

    print("Model training completed. Best model saved to:", MODEL_SAVE_PATH)
    return history

def evaluate_model(model, test_x, test_y):
    """
    Evaluate the trained model on the test set.
    """
    print("Evaluating the model on the test set...")
    loss, accuracy = model.evaluate(test_x, test_y)
    print(f"Test Accuracy: {accuracy:.2f}")
    return loss, accuracy

if __name__ == "__main__":
    train_x, train_y, val_x, val_y, test_x, test_y = load_processed_data(PROCESSED_DATA_DIR)

    model = build_model(IMG_SIZE)

    history = train_model(model, train_x, train_y, val_x, val_y)

    evaluate_model(model, test_x, test_y)

import os
import numpy as np
from sklearn.utils import class_weight
from keras._tf_keras.keras.models import Sequential
from keras._tf_keras.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from keras._tf_keras.keras.callbacks import EarlyStopping, ModelCheckpoint
from keras._tf_keras.keras.optimizers import Adam
import matplotlib.pyplot as plt

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
    print("Building the CNN model...")
    model = Sequential()

    # Lapisan konvolusi pertama
    model.add(Conv2D(32, (3, 3), activation="relu", input_shape=img_size))
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.25))  # Dropout untuk mencegah overfitting

    # Lapisan konvolusi kedua
    model.add(Conv2D(64, (3, 3), activation="relu"))
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.25))

    # Lapisan konvolusi ketiga
    model.add(Conv2D(128, (3, 3), activation="relu"))
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.25))

    # BatchNormalization untuk mempercepat pelatihan
    model.add(BatchNormalization())

    # Lapisan Dense
    model.add(Flatten())
    model.add(Dense(128, activation="relu"))
    model.add(Dropout(0.5))  # Dropout pada layer fully connected
    model.add(Dense(1, activation="sigmoid"))  # Sigmoid untuk klasifikasi binary

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
    model_checkpoint = ModelCheckpoint(MODEL_SAVE_PATH, save_best_only=False, monitor="val_loss")

    class_weight = {0: 1., 1: 8.} 

    # class_weights = class_weight.compute_class_weight('balanced', classes=np.unique(train_y), y=train_y)
    # class_weights_dict = {i: class_weights[i] for i in range(len(class_weights))}

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

def plot_training_history(history):
    plt.figure(figsize=(12, 6))
    # Loss plot
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Loss')
    plt.legend()

    # Accuracy plot
    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Accuracy')
    plt.legend()

    plt.show()

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

    plot_training_history(history)

    evaluate_model(model, test_x, test_y)

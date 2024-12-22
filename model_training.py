import os
import numpy as np
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix
from keras._tf_keras.keras.models import Sequential
from keras._tf_keras.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization, GlobalAveragePooling2D
from keras._tf_keras.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, LearningRateScheduler
from keras._tf_keras.keras.optimizers import Adam
from keras._tf_keras.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
from keras._tf_keras.keras.regularizers import l2

PROCESSED_DATA_DIR = "processed_data"
MODEL_SAVE_PATH = "models/bansos_model_ai.keras"
IMG_SIZE = (224, 224, 3)

def load_processed_data(data_dir):
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

    # First Convolutional Layer
    model.add(Conv2D(32, (3, 3), activation="relu", input_shape=img_size))
    model.add(MaxPooling2D((2, 2)))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))

    # Second Convolutional Layer
    model.add(Conv2D(64, (3, 3), activation="relu", kernel_regularizer=l2(0.01)))
    model.add(MaxPooling2D((2, 2)))
    model.add(BatchNormalization())
    model.add(Dropout(0.25))

    # Third Convolutional Layer
    model.add(Conv2D(128, (3, 3), activation="relu", kernel_regularizer=l2(0.01)))
    model.add(MaxPooling2D((2, 2)))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))

    # Fourth Convolutional Layer
    model.add(Conv2D(256, (3, 3), activation="relu", kernel_regularizer=l2(0.01)))
    model.add(MaxPooling2D((2, 2)))
    model.add(BatchNormalization())
    model.add(Dropout(0.4))

    # Use Global Average Pooling instead of Flatten for better generalization
    model.add(GlobalAveragePooling2D())

    # Dense Layers
    model.add(Dense(512, activation="relu", kernel_regularizer=l2(0.01)))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation="sigmoid"))

    model.compile(optimizer=Adam(learning_rate=0.001), 
                  loss="binary_crossentropy", 
                  metrics=["accuracy"])

    print("Model built and compiled successfully.")
    return model

def lr_schedule(epoch, lr):
    """
    Learning Rate Scheduler: decrease LR after each epoch.
    """
    if epoch < 10:
        return lr
    else:
        return lr * 0.1  # Reduce LR after 10 epochs

def train_model(model, train_x, train_y, val_x, val_y):
    print("Starting model training...")
    os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)

    early_stopping = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True, min_delta=0.001)
    model_checkpoint = ModelCheckpoint(MODEL_SAVE_PATH, save_best_only=False, monitor="val_loss")
    reduce_lr = ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, verbose=1)
    lr_scheduler = LearningRateScheduler(lr_schedule)

    # Compute class weights to handle class imbalance
    class_weights = compute_class_weight('balanced', classes=np.unique(train_y), y=train_y)
    class_weight_dict = {i: class_weights[i] for i in range(len(class_weights))}

    datagen = ImageDataGenerator(
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.3,
        horizontal_flip=True,
        fill_mode="nearest",
        brightness_range=[0.8, 1.2]
    )
    datagen.fit(train_x)

    history = model.fit(
        train_x, train_y,
        validation_data=(val_x, val_y),
        epochs=30,
        batch_size=32,
        class_weight=class_weight_dict,
        callbacks=[early_stopping, model_checkpoint, reduce_lr, lr_scheduler]
    )

    print("Model training completed. Best model saved to:", MODEL_SAVE_PATH)
    return history

def plot_training_history(history):
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Accuracy')
    plt.legend()

    plt.show()

if __name__ == "__main__":
    train_x, train_y, val_x, val_y, test_x, test_y = load_processed_data(PROCESSED_DATA_DIR)

    model = build_model(IMG_SIZE)

    history = train_model(model, train_x, train_y, val_x, val_y)

    plot_training_history(history)

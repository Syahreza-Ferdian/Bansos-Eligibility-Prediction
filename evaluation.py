import os
import numpy as np
from keras._tf_keras.keras.models import load_model
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

MODEL_PATH = "models/bansos_model_ai.keras"
PROCESSED_DATA_DIR = "processed_data"
IMG_SIZE = (224, 224, 3)  

def load_processed_data(data_dir):
    """
    Load processed data saved as .npy files for evaluation.
    """
    print("Loading processed data...")
    test_x = np.load(os.path.join(data_dir, "test_x.npy"))
    test_y = np.load(os.path.join(data_dir, "test_y.npy"))

    print("Data successfully loaded.")
    return test_x, test_y

def load_trained_model(model_path):
    """
    Load the trained model from the specified path.
    """
    print("Loading the trained model from:", model_path)
    model = load_model(model_path)
    print("Model loaded successfully.")
    return model

def evaluate_model(model, test_x, test_y):
    """
    Evaluate the model on the test data and print classification report and confusion matrix.
    """
    print("Evaluating the model on the test set...")
    test_loss, test_accuracy = model.evaluate(test_x, test_y)
    print(f"Test Accuracy: {test_accuracy:.2f}")
    print(f"Test Loss: {test_loss:.2f}")

    predictions = model.predict(test_x)
    predicted_classes = (predictions > 0.3).astype("int32") 

    print("\nClassification Report:")
    print(classification_report(test_y, predicted_classes))

    print("\nConfusion Matrix:")
    cm = confusion_matrix(test_y, predicted_classes)
    print(cm)

    plt.figure(figsize=(6, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=["Not Eligible", "Eligible"], yticklabels=["Not Eligible", "Eligible"])
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()

if __name__ == "__main__":
    test_x, test_y = load_processed_data(PROCESSED_DATA_DIR)

    model = load_trained_model(MODEL_PATH)

    evaluate_model(model, test_x, test_y)

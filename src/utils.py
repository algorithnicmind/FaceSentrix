"""
FaceSentrix - Utility Functions
"""

import os
import numpy as np
import matplotlib.pyplot as plt

# The 7 universal emotion categories for FaceSentrix
EMOTION_LABELS = {
    0: "Angry",
    1: "Disgust",
    2: "Fear",
    3: "Happy",
    4: "Sad",
    5: "Surprise",
    6: "Neutral"
}

def get_class_labels():
    """
    Returns the mapping of integer class IDs to emotion string labels.
    """
    return EMOTION_LABELS

def load_processed_data(data_dir="data/processed"):
    """
    Loads the preprocessed numpy arrays for training, validation, and testing.
    
    Returns:
        tuple: (X_train, y_train, X_val, y_val, X_test, y_test)
    """
    try:
        print(f"Loading preprocessed dataset from {data_dir}...")
        X_train = np.load(os.path.join(data_dir, "X_train.npy"))
        y_train = np.load(os.path.join(data_dir, "y_train.npy"))
        X_val = np.load(os.path.join(data_dir, "X_val.npy"))
        y_val = np.load(os.path.join(data_dir, "y_val.npy"))
        X_test = np.load(os.path.join(data_dir, "X_test.npy"))
        y_test = np.load(os.path.join(data_dir, "y_test.npy"))
        print("Data successfully loaded!")
        return X_train, y_train, X_val, y_val, X_test, y_test
    except FileNotFoundError as e:
        print(f"Error loading data: {e}")
        print("Please ensure you have run the preprocessing pipeline FIRST.")
        return None, None, None, None, None, None

def visualize_sample(image, label_id):
    """
    Visualizes a single 48x48 grayscale face image.
    """
    plt.imshow(image.squeeze(), cmap="gray")
    plt.title(f"Label: {EMOTION_LABELS.get(label_id, 'Unknown')}")
    plt.axis("off")
    plt.show()

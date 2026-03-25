"""
FaceSentrix - Data Augmentation Strategies
"""

from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np

def get_train_augmenter():
    """
    Returns an ImageDataGenerator configured with our chosen augmentation strategies
    for the FER-2013 training set. Assumes inputs are normalized [0,1].
    """
    datagen = ImageDataGenerator(
        rotation_range=15,          # Rotate up to 15 degrees
        width_shift_range=0.1,      # Shift width by 10%
        height_shift_range=0.1,     # Shift height by 10%
        zoom_range=0.1,             # Zoom up to 10%
        horizontal_flip=True,       # Mirror faces horizontally
        fill_mode='nearest'         # Fill pixels created by shifting
    )
    return datagen

def get_validation_augmenter():
    """
    Returns an ImageDataGenerator for validation/testing.
    No geometric augmentations should be applied here, only formatting.
    """
    return ImageDataGenerator() # No augmentation for validation data

def apply_augmentation(X_train, y_train, batch_size=64):
    """
    Creates an iterator that yields augmented batches of training data.
    """
    datagen = get_train_augmenter()
    # Provide the generator
    return datagen.flow(X_train, y_train, batch_size=batch_size)

if __name__ == "__main__":
    print("Augmentation configuration module loaded successfully.")

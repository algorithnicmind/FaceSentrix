"""
FaceSentrix - CNN Emotion Classifier Module
"""

import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
import numpy as np

class EmotionClassifier:
    """
    Emotion Classifier using a Convolutional Neural Network.
    
    Categories:
    0: Angry, 1: Disgust, 2: Fear, 3: Happy, 4: Sad, 5: Surprise, 6: Neutral
    """
    
    def __init__(self, model_path=None):
        """
        Initializes the classifier. If model_path is provided, loads a pre-trained model.
        Otherwise, a new model needs to be built utilizing build_model().
        """
        self.model = None
        if model_path:
            self.load_active_model(model_path)
            
    def load_active_model(self, model_path):
        """Loads a pre-trained Keras model from disk."""
        try:
            self.model = load_model(model_path)
            print(f"Model successfully loaded from {model_path}")
        except Exception as e:
            print(f"Failed to load model: {e}")
            
    def build_model(self, input_shape=(48, 48, 1), num_classes=7):
        """
        Builds the Convolutional Neural Network architecture based on our specs.
        Returns the uncompiled Sequential model.
        """
        model = Sequential([
            # Block 1: Feature Extraction
            Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=input_shape),
            BatchNormalization(),
            Conv2D(32, (3, 3), padding='same', activation='relu'),
            BatchNormalization(),
            MaxPooling2D(pool_size=(2, 2)),
            Dropout(0.25),
            
            # Block 2
            Conv2D(64, (3, 3), padding='same', activation='relu'),
            BatchNormalization(),
            Conv2D(64, (3, 3), padding='same', activation='relu'),
            BatchNormalization(),
            MaxPooling2D(pool_size=(2, 2)),
            Dropout(0.25),
            
            # Block 3
            Conv2D(128, (3, 3), padding='same', activation='relu'),
            BatchNormalization(),
            Conv2D(128, (3, 3), padding='same', activation='relu'),
            BatchNormalization(),
            MaxPooling2D(pool_size=(2, 2)),
            Dropout(0.25),
            
            # Block 4
            Conv2D(256, (3, 3), padding='same', activation='relu'),
            BatchNormalization(),
            Conv2D(256, (3, 3), padding='same', activation='relu'),
            BatchNormalization(),
            MaxPooling2D(pool_size=(2, 2)),
            Dropout(0.25),
            
            # Classification Head
            Flatten(),
            Dense(512, activation='relu'),
            BatchNormalization(),
            Dropout(0.5),
            Dense(256, activation='relu'),
            BatchNormalization(),
            Dropout(0.5),
            Dense(num_classes, activation='softmax')
        ])
        
        self.model = model
        return model

    def preprocess_face(self, face_img, target_size=(48, 48)):
        """
        Resizes, normalizes, and reshapes a raw grayscale face crop for model input.
        Includes CLAHE for lighting robustness.
        """
        import cv2
        # Apply CLAHE to enhance contrast and normalize lighting
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        face_enhanced = clahe.apply(face_img)
        
        # Resize to model input shape
        face_resized = cv2.resize(face_enhanced, target_size)
        
        # Normalize pixel values
        face_norm = face_resized.astype('float32') / 255.0
        
        # Reshape to (48, 48, 1)
        face_final = np.expand_dims(face_norm, axis=-1)
        return face_final

    def predict(self, face_image):
        """
        Predicts the emotion for a single preprocessed face image.
        Expected input: normalized numpy array of shape (48, 48, 1).
        """
        if self.model is None:
            raise ValueError("Model is not loaded or built. Cannot predict.")
            
        # Ensure correct shape (batch_size, height, width, channels)
        if len(face_image.shape) == 3:
            face_image = np.expand_dims(face_image, axis=0)
            
        predictions = self.model.predict(face_image, verbose=0)
        class_id = np.argmax(predictions[0])
        confidence = predictions[0][class_id]
        
        return class_id, confidence, predictions[0]

if __name__ == "__main__":
    classifier = EmotionClassifier()
    model = classifier.build_model()
    model.summary()

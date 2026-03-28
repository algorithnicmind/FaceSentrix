import os
import sys
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score

# Ensure root directory is in sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils import EMOTION_LABELS

def evaluate_model():
    print("Loading test data...")
    X_test = np.load('data/processed/X_test.npy')
    y_test = np.load('data/processed/y_test.npy')
    
    # Scale if not scaled (assuming preprocessing scaled to 0-1)
    if X_test.max() > 1.0:
        X_test = X_test.astype('float32') / 255.0
        
    print(f"Test shape: {X_test.shape}")
    
    print("Loading model...")
    model = tf.keras.models.load_model('models/emotion_model.h5')
    
    print("Predicting on test set...")
    y_pred_probs = model.predict(X_test, batch_size=64)
    y_pred = np.argmax(y_pred_probs, axis=1)
    if len(y_test.shape) > 1 and y_test.shape[1] > 1:
        y_true = np.argmax(y_test, axis=1)
    else:
        y_true = y_test
    
    # Accuracy and F1
    acc = accuracy_score(y_true, y_pred)
    weighted_f1 = f1_score(y_true, y_pred, average='weighted')
    
    print("-" * 50)
    print(f"Overall Accuracy:  {acc:.4f}")
    print(f"Weighted F1 Score: {weighted_f1:.4f}")
    print("-" * 50)
    
    # Classification Report
    emotion_labels = [EMOTION_LABELS[i] for i in range(7)]
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=emotion_labels))
    
    # Confusion Matrix Heatmap
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=emotion_labels, yticklabels=emotion_labels)
    plt.title('Emotion Classification Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    
    # Save the confusion matrix plot
    os.makedirs('assets', exist_ok=True)
    plt.savefig('assets/confusion_matrix.png', dpi=300)
    print("Saved confusion matrix visualization to assets/confusion_matrix.png")
    
if __name__ == '__main__':
    evaluate_model()

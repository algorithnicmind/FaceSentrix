"""
FaceSentrix - Model Training Script
"""

import os
import argparse
import json
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, CSVLogger

# Fix imports since script is in a subdirectory
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.utils import load_processed_data, get_balanced_class_weights
from training.augment import apply_augmentation
from src.emotion_classifier import EmotionClassifier

def train_model(epochs=50, batch_size=64):
    print("--- Starting FaceSentrix Model Training ---")
    
    # 1. Load Data
    X_train, y_train, X_val, y_val, _, _ = load_processed_data()
    if X_train is None:
        return
        
    # Convert labels to one-hot categorical format
    y_train_cat = tf.keras.utils.to_categorical(y_train, num_classes=7)
    y_val_cat = tf.keras.utils.to_categorical(y_val, num_classes=7)
    
    # 2. Get Class Weights to combat class imbalance
    class_weights = get_balanced_class_weights(y_train)
    print(f"Computed Class Weights: {class_weights}")
    
    # 3. Create Augmentation Generator
    train_generator = apply_augmentation(X_train, y_train_cat, batch_size=batch_size)
    steps_per_epoch = len(X_train) // batch_size
    
    # 4. Build the CNN Model
    classifier = EmotionClassifier()
    model = classifier.build_model()
    
    # 5. Compile the Model
    optimizer = Adam(learning_rate=0.0001)
    model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # 6. Configure Callbacks
    os.makedirs('models', exist_ok=True)
    os.makedirs('logs', exist_ok=True)
    
    checkpoint = ModelCheckpoint(
        filepath='models/emotion_model.h5',
        monitor='val_accuracy',
        save_best_only=True,
        mode='max',
        verbose=1
    )
    
    early_stop = EarlyStopping(
        monitor='val_accuracy',
        patience=15, # Increased patience for deeper model
        restore_best_weights=True,
        verbose=1
    )
    
    reduce_lr = ReduceLROnPlateau(
        monitor='val_accuracy',
        factor=0.2,
        patience=5,
        min_lr=1e-7,
        verbose=1
    )
    
    csv_logger = CSVLogger('models/training_log.csv')
    
    callbacks = [checkpoint, early_stop, reduce_lr, csv_logger]
    
    # 7. Start Training
    print("Beginning Training Loop...")
    history = model.fit(
        train_generator,
        steps_per_epoch=steps_per_epoch,
        epochs=epochs,
        validation_data=(X_val, y_val_cat),
        class_weight=class_weights,
        callbacks=callbacks
    )
    
    # 8. Save Training History
    with open('models/training_history.json', 'w') as f:
        # Convert NumPy types to Python scalars for JSON serialization
        hist_dict = {k: [float(val) for val in v] for k, v in history.history.items()}
        json.dump(hist_dict, f)
        
    print("--- Training Completed Successfully ---")
    print("Best model automatically saved to models/emotion_model.h5")

def main():
    parser = argparse.ArgumentParser(description="FaceSentrix CNN Trainer")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size for training")
    args = parser.parse_args()
    
    train_model(epochs=args.epochs, batch_size=args.batch_size)

if __name__ == "__main__":
    main()

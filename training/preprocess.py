"""
FaceSentrix - Dataset Preprocessing & Download Script
"""

import os
import argparse
import numpy as np
import cv2
from tqdm import tqdm
from sklearn.model_selection import train_test_split

EMOTIONS = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]

def download_dataset(output_dir="data/raw"):
    """
    Instructions or placeholder script to download the FER-2013 dataset.
    Since Kaggle requires API keys, we log instructions for manual download.
    """
    os.makedirs(output_dir, exist_ok=True)
    print("--- FaceSentrix Dataset Downloader ---")
    print(f"Target Directory: {os.path.abspath(output_dir)}")
    print("Please download FER-2013 from Kaggle:")
    print("Link: https://www.kaggle.com/datasets/msambare/fer2013")
    print("After downloading, extract the contents so that you have:")
    print("  - data/raw/train")
    print("  - data/raw/test")
    print("--------------------------------------")

def process_image_folder(folder_path):
    images = []
    labels = []
    
    if not os.path.exists(folder_path):
        print(f"Warning: {folder_path} does not exist.")
        return np.array(images), np.array(labels)
        
    for label_id, emotion in enumerate(EMOTIONS):
        emotion_dir = os.path.join(folder_path, emotion)
        if not os.path.exists(emotion_dir):
            continue
            
        for img_name in tqdm(os.listdir(emotion_dir), desc=f"Processing {emotion}"):
            img_path = os.path.join(emotion_dir, img_name)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                img = cv2.resize(img, (48, 48))
                images.append(img)
                labels.append(label_id)
                
    return np.array(images), np.array(labels)

def preprocess_dataset(raw_dir="data/raw", processed_dir="data/processed"):
    print("--- FaceSentrix Data Preprocessing Pipeline ---")
    train_dir = os.path.join(raw_dir, "train")
    test_dir = os.path.join(raw_dir, "test")
    
    if not os.path.exists(train_dir) or not os.path.exists(test_dir):
        print("Dataset not found! Run with --download to see instructions or download manually.")
        return
        
    print("Processing Training Data...")
    X_train_full, y_train_full = process_image_folder(train_dir)
    
    print("Processing Testing Data...")
    X_test, y_test = process_image_folder(test_dir)
    
    # Normalize images
    X_train_full = X_train_full.astype('float32') / 255.0
    X_test = X_test.astype('float32') / 255.0
    
    # Reshape for CNN (samples, height, width, channels=1)
    X_train_full = np.expand_dims(X_train_full, axis=-1)
    X_test = np.expand_dims(X_test, axis=-1)
    
    # Split train into train and validation (90/10)
    print("Splitting training data into train/validation sets...")
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full, y_train_full, test_size=0.1, random_state=42, stratify=y_train_full
    )
    
    # Save processed arrays
    print(f"Saving preprocessed data to {processed_dir}...")
    os.makedirs(processed_dir, exist_ok=True)
    
    np.save(os.path.join(processed_dir, "X_train.npy"), X_train)
    np.save(os.path.join(processed_dir, "y_train.npy"), y_train)
    np.save(os.path.join(processed_dir, "X_val.npy"), X_val)
    np.save(os.path.join(processed_dir, "y_val.npy"), y_val)
    np.save(os.path.join(processed_dir, "X_test.npy"), X_test)
    np.save(os.path.join(processed_dir, "y_test.npy"), y_test)
    
    print("--- Preprocessing Complete! ---")
    print(f"Train samples: {X_train.shape[0]}")
    print(f"Val samples:   {X_val.shape[0]}")
    print(f"Test samples:  {X_test.shape[0]}")

def main():
    parser = argparse.ArgumentParser(description="FaceSentrix Dataset Tools")
    parser.add_argument("--download", action="store_true", help="Download the dataset")
    parser.add_argument("--process", action="store_true", help="Process the raw dataset")
    args = parser.parse_args()

    if args.download:
        download_dataset()
    elif args.process:
        preprocess_dataset()
    else:
        print("Please provide an argument: --download or --process")

if __name__ == "__main__":
    main()

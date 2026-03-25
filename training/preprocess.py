"""
FaceSentrix - Dataset Preprocessing & Download Script
"""

import os
import argparse

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

def main():
    parser = argparse.ArgumentParser(description="FaceSentrix Dataset Tools")
    parser.add_argument("--download", action="store_true", help="Download the dataset")
    args = parser.parse_args()

    if args.download:
        download_dataset()
    else:
        print("Run with --download to get dataset instructions.")
        print("Preprocessing steps to be implemented...")

if __name__ == "__main__":
    main()

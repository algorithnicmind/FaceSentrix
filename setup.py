from setuptools import setup, find_packages

setup(
    name="FaceSentrix",
    version="0.1.0",
    description="Real-Time Face Detection & Emotion Recognition System",
    author="AlgorithmicMind",
    author_email="contact@example.com",
    packages=find_packages(),
    install_requires=[
        "opencv-python>=4.8.0",
        "tensorflow>=2.13.0",
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "matplotlib>=3.7.0",
        "seaborn>=0.12.0",
        "scikit-learn>=1.3.0",
        "Pillow>=10.0.0",
        "tqdm>=4.65.0",
    ],
    classifiers=[
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "License :: CC0 1.0 Universal (CC0 1.0) Public Domain Dedication",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
)

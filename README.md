<p align="center">
  <img src="../assets/banner.png" alt="FaceSentrix Banner" width="800"/>
</p>

<h1 align="center">📘 FaceSentrix — Project Documentation</h1>

<p align="center">
  <b>Real-Time Face Detection & Emotion Recognition System</b><br/>
  <i>Powered by Python · OpenCV · CNN / Deep Learning</i>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Status-In%20Development-yellow?style=for-the-badge" alt="status"/>
  <img src="https://img.shields.io/badge/Python-3.8+-blue?style=for-the-badge&logo=python" alt="python"/>
  <img src="https://img.shields.io/badge/OpenCV-4.x-green?style=for-the-badge&logo=opencv" alt="opencv"/>
  <img src="https://img.shields.io/badge/TensorFlow-2.x-orange?style=for-the-badge&logo=tensorflow" alt="tensorflow"/>
</p>

---

## 🧠 Project Overview

**FaceSentrix** is a real-time emotion detection system that identifies human faces from a live camera feed and classifies their emotional state using deep learning (CNN). It is designed to be lightweight, extensible, and deployable across platforms.

### 🎯 Core Objectives

| # | Objective | Description |
|---|-----------|-------------|
| 1 | **Face Detection** | Detect one or multiple faces in real-time from a webcam or video stream using Haar Cascades or DNN-based detectors |
| 2 | **Emotion Classification** | Predict emotional states — **Happy, Sad, Angry, Surprise, Fear, Disgust, Neutral** — using a trained CNN model |
| 3 | **Real-Time Processing** | Achieve smooth, low-latency inference suitable for live camera feeds (target: ≥15 FPS) |
| 4 | **Visual Feedback** | Display bounding boxes around detected faces with emotion labels and confidence scores overlaid |
| 5 | **Model Training Pipeline** | Build an end-to-end training pipeline: data → preprocessing → augmentation → model → evaluation |

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        FaceSentrix Pipeline                     │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   ┌──────────┐    ┌──────────────┐    ┌─────────────────┐       │
│   │  Camera   │───▶│  Face Detect │───▶│ Emotion Classify│       │
│   │  Input    │    │  (OpenCV)    │    │  (CNN Model)    │       │
│   └──────────┘    └──────────────┘    └────────┬────────┘       │
│                                                │                │
│                                       ┌────────▼────────┐       │
│                                       │  Display Output │       │
│                                       │  (Labels + Box) │       │
│                                       └─────────────────┘       │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## 🧰 Technology Stack

| Layer | Technology | Purpose |
|-------|-----------|---------|
| **Language** | Python 3.8+ | Core programming language |
| **Computer Vision** | OpenCV 4.x | Camera access, face detection, image processing |
| **Deep Learning** | TensorFlow / Keras | CNN model building, training, and inference |
| **Data Handling** | NumPy, Pandas | Numerical operations & dataset management |
| **Visualization** | Matplotlib, Seaborn | Training metrics plots, confusion matrices |
| **Model Serving** | TensorFlow Lite *(optional)* | Lightweight model for edge deployment |
| **Dataset** | FER-2013 / AffectNet | Labeled facial expression images for training |

---

## 📁 Planned Project Structure

```
FaceSentrix/
├── 📄 README.md                  # Main project README
├── 📄 LICENSE                    # CC0 1.0 Universal License
├── 📄 .gitignore                 # Git ignore rules
├── 📄 requirements.txt           # Python dependencies
├── 📄 setup.py                   # Package setup (optional)
│
├── 📂 data/                      # Dataset storage
│   ├── raw/                      # Original FER-2013 data
│   ├── processed/                # Preprocessed & augmented images
│   └── README.md                 # Dataset documentation
│
├── 📂 models/                    # Trained model files
│   ├── emotion_model.h5          # Saved Keras model
│   ├── emotion_model.tflite      # TFLite converted model (optional)
│   └── training_history.json     # Training metrics log
│
├── 📂 src/                       # Source code
│   ├── __init__.py
│   ├── face_detector.py          # Face detection module
│   ├── emotion_classifier.py     # Emotion prediction module
│   ├── camera.py                 # Webcam capture module
│   ├── visualizer.py             # Overlay rendering (bounding boxes, labels)
│   └── utils.py                  # Utility functions
│
├── 📂 training/                  # Model training scripts
│   ├── train.py                  # Main training script
│   ├── evaluate.py               # Model evaluation & metrics
│   ├── preprocess.py             # Data preprocessing pipeline
│   └── augment.py                # Data augmentation strategies
│
├── 📂 notebooks/                 # Jupyter notebooks
│   ├── 01_data_exploration.ipynb
│   ├── 02_model_training.ipynb
│   └── 03_evaluation.ipynb
│
├── 📂 tests/                     # Unit & integration tests
│   ├── test_face_detector.py
│   ├── test_emotion_classifier.py
│   └── test_camera.py
│
├── 📂 assets/                    # Static assets (images, banners)
│   └── banner.png
│
├── 📂 docs/                      # Documentation (gitignored)
│   ├── README.md                 # This file
│   ├── TODO.md                   # Project roadmap & task list
│   ├── ARCHITECTURE.md           # Detailed architecture docs
│   ├── MODEL_DESIGN.md           # CNN model design decisions
│   ├── DATASET_GUIDE.md          # Dataset setup & preprocessing guide
│   └── DEPLOYMENT.md             # Deployment instructions
│
└── 📂 app/                       # Web/Desktop app (Phase 3)
    ├── app.py                    # Flask/Streamlit app entry
    ├── templates/
    └── static/
```

---

## 📊 Emotion Classes

The model will classify faces into **7 universal emotion categories**:

| # | Emotion | Description | Example Use Case |
|---|---------|-------------|-----------------|
| 0 | 😡 **Angry** | Frustration, irritation | Customer feedback analysis |
| 1 | 🤢 **Disgust** | Revulsion, distaste | Product reaction testing |
| 2 | 😨 **Fear** | Anxiety, apprehension | Safety & security systems |
| 3 | 😊 **Happy** | Joy, satisfaction | User experience monitoring |
| 4 | 😢 **Sad** | Sorrow, disappointment | Mental health screening |
| 5 | 😮 **Surprise** | Shock, amazement | Engagement detection |
| 6 | 😐 **Neutral** | Calm, expressionless | Baseline comparison |

---

## 🔗 Documentation Index

| Document | Description |
|----------|-------------|
| [📋 TODO.md](./TODO.md) | Complete project roadmap with step-by-step tasks |
| [🏗️ ARCHITECTURE.md](./ARCHITECTURE.md) | System architecture & design patterns |
| [🧠 MODEL_DESIGN.md](./MODEL_DESIGN.md) | CNN model architecture & training strategy |
| [📦 DATASET_GUIDE.md](./DATASET_GUIDE.md) | Dataset download, setup & preprocessing |
| [🚀 DEPLOYMENT.md](./DEPLOYMENT.md) | Deployment & production guide |

---

## 🚀 Quick Start (Preview)

```bash
# Clone the repository
git clone https://github.com/algorithnicmind/FaceSentrix.git
cd FaceSentrix

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Download the dataset
python training/preprocess.py --download

# Train the model
python training/train.py --epochs 50 --batch-size 64

# Run real-time detection
python src/camera.py
```

---

## 📈 Expected Performance Targets

| Metric | Target | Notes |
|--------|--------|-------|
| **Detection FPS** | ≥ 15 FPS | On standard webcam with CPU |
| **Model Accuracy** | ≥ 65% | On FER-2013 test set |
| **Inference Time** | < 50ms | Per frame (detection + classification) |
| **Model Size** | < 50MB | For easy distribution |
| **Face Detection Rate** | ≥ 95% | For frontal faces |

---

## 🤝 Contributing

This project follows a **commit-per-change** workflow. Every modification — no matter how small — gets its own commit. This ensures a clear, traceable development history.

---

<p align="center">
  <b>Built with ❤️ by <a href="https://github.com/algorithnicmind">AlgorithmicMind</a></b>
</p>

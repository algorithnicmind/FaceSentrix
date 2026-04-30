<h1 align="center">📦 FaceSentrix — Dataset Guide</h1>

<p align="center">
  <i>Complete guide for dataset selection, download, setup, and preprocessing</i>
</p>

---

## 📋 Dataset Overview

### Primary Dataset: FER-2013

| Property | Details |
|----------|---------|
| **Name** | Facial Expression Recognition 2013 (FER-2013) |
| **Source** | [Kaggle](https://www.kaggle.com/datasets/msambare/fer2013) |
| **Total Images** | 35,887 |
| **Image Size** | 48 × 48 pixels |
| **Color** | Grayscale |
| **Format** | CSV (pixel values) or PNG images |
| **Classes** | 7 emotions |
| **License** | Open for research use |

### Class Distribution

```
Emotion       Count     Percentage    Bar
──────────────────────────────────────────────────
Angry         4,953     13.8%         ██████████████
Disgust         547      1.5%         ██
Fear          5,121     14.3%         ██████████████
Happy         8,989     25.0%         █████████████████████████
Sad           6,077     16.9%         █████████████████
Surprise      4,002     11.2%         ███████████
Neutral       6,198     17.3%         █████████████████
──────────────────────────────────────────────────
Total        35,887    100.0%
```

### Data Split

| Split | Count | Percentage | Purpose |
|-------|-------|------------|---------|
| Training | 28,709 | 80% | Model training |
| Validation | 3,589 | 10% | Hyperparameter tuning |
| Test | 3,589 | 10% | Final evaluation |

> **⚠️ Note:** The "Disgust" class is severely under-represented (1.5%). Special handling is required (see Class Imbalance section).

---

## 📥 Download Instructions

### Method 1: Kaggle API (Recommended)

```bash
# Step 1: Install Kaggle CLI
pip install kaggle

# Step 2: Set up Kaggle API credentials
# Download kaggle.json from https://www.kaggle.com/settings
# Place it in:
#   Windows: C:\Users\<username>\.kaggle\kaggle.json
#   Linux/Mac: ~/.kaggle/kaggle.json

# Step 3: Download the dataset
kaggle datasets download -d msambare/fer2013 -p data/raw/

# Step 4: Unzip
cd data/raw
unzip fer2013.zip
```

### Method 2: Manual Download

1. Visit [FER-2013 on Kaggle](https://www.kaggle.com/datasets/msambare/fer2013)
2. Click "Download" (requires free Kaggle account)
3. Extract the ZIP to `data/raw/` directory

### Method 3: Using the Project Script

```bash
# Our automated download script (when implemented)
python training/preprocess.py --download

# This will:
# 1. Check for Kaggle API credentials
# 2. Download the dataset
# 3. Verify file integrity
# 4. Extract to correct directory
```

### Expected Directory Structure After Download

```
data/
├── raw/
│   └── fer2013/
│       ├── train/
│       │   ├── angry/       (3,995 images)
│       │   ├── disgust/     (436 images)
│       │   ├── fear/        (4,097 images)
│       │   ├── happy/       (7,215 images)
│       │   ├── sad/         (4,830 images)
│       │   ├── surprise/    (3,171 images)
│       │   └── neutral/     (4,965 images)
│       └── test/
│           ├── angry/       (958 images)
│           ├── disgust/     (111 images)
│           ├── fear/        (1,024 images)
│           ├── happy/       (1,774 images)
│           ├── sad/         (1,247 images)
│           ├── surprise/    (831 images)
│           └── neutral/     (1,233 images)
└── processed/
    ├── X_train.npy          (training images array)
    ├── y_train.npy          (training labels array)
    ├── X_val.npy            (validation images array)
    ├── y_val.npy            (validation labels array)
    ├── X_test.npy           (test images array)
    └── y_test.npy           (test labels array)
```

---

## 🔄 Preprocessing Pipeline

### Step-by-Step Process

```
Step 1: Load Images
├── Read images from directory structure
├── Convert to numpy arrays
└── Store with corresponding labels

Step 2: Validate Data
├── Check for corrupted images
├── Verify image dimensions (48×48)
├── Remove duplicates (if any)
└── Log statistics

Step 3: Normalize
├── Convert pixel values: [0, 255] → [0.0, 1.0]
├── Formula: pixel_normalized = pixel / 255.0
└── Data type: float32

Step 4: Reshape
├── Ensure shape: (num_samples, 48, 48, 1)
├── Add channel dimension if missing
└── Verify tensor shapes

Step 5: Encode Labels
├── Map emotion strings → integers (0-6)
├── One-hot encode: 3 → [0, 0, 0, 1, 0, 0, 0]
└── Verify encoding correctness

Step 6: Split Dataset
├── Training:   80% (28,709 samples)
├── Validation: 10% (3,589 samples)
├── Test:       10% (3,589 samples)
└── Stratified split (maintain class proportions)

Step 7: Save Processed Data
├── Save as .npy files for fast loading
└── Verify saved file integrity
```

### Label Encoding

| Label | Emotion | Integer | One-Hot |
|-------|---------|---------|---------|
| angry | Angry | 0 | [1,0,0,0,0,0,0] |
| disgust | Disgust | 1 | [0,1,0,0,0,0,0] |
| fear | Fear | 2 | [0,0,1,0,0,0,0] |
| happy | Happy | 3 | [0,0,0,1,0,0,0] |
| sad | Sad | 4 | [0,0,0,0,1,0,0] |
| surprise | Surprise | 5 | [0,0,0,0,0,1,0] |
| neutral | Neutral | 6 | [0,0,0,0,0,0,1] |

---

## ⚖️ Handling Class Imbalance

### The Problem

The "Disgust" class has only **547 samples** (1.5%) compared to "Happy" with **8,989 samples** (25%). This can cause the model to be biased toward majority classes.

### Solutions

#### Option 1: Class Weights (Recommended)

```python
from sklearn.utils.class_weight import compute_class_weight

class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(y_train),
    y=y_train
)
class_weight_dict = dict(enumerate(class_weights))

# Use in model.fit()
model.fit(X_train, y_train, class_weight=class_weight_dict)
```

#### Option 2: Oversampling Minority Classes

```python
# Random oversampling of minority classes
from imblearn.over_sampling import RandomOverSampler

ros = RandomOverSampler(random_state=42)
X_resampled, y_resampled = ros.fit_resample(
    X_train.reshape(len(X_train), -1), y_train
)
X_resampled = X_resampled.reshape(-1, 48, 48, 1)
```

#### Option 3: Data Augmentation on Minority Classes

Apply more aggressive augmentation specifically to under-represented classes (Disgust, Fear).

---

## 🔀 Data Augmentation Details

### Augmentation Pipeline

```python
from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    brightness_range=[0.8, 1.2],
    fill_mode='nearest'
)
```

### Visual Examples

```
Original     Rotated      Flipped      Zoomed       Shifted
┌────────┐   ┌────────┐   ┌────────┐   ┌────────┐   ┌────────┐
│  😊    │   │  😊↗  │   │  😊🔄 │   │ 😊🔍  │   │  😊→  │
│ (face) │   │ (tilted)│   │(mirror)│   │(closer)│   │(moved) │
└────────┘   └────────┘   └────────┘   └────────┘   └────────┘
```

---

## 🔍 Alternative Datasets

| Dataset | Size | Resolution | Classes | Pros | Cons |
|---------|------|-----------|---------|------|------|
| **FER-2013** | 35,887 | 48×48 | 7 | Free, widely used, easy to access | Low resolution, noisy labels |
| **AffectNet** | 450,000+ | Varies | 8 | Large, high quality | Requires application, large size |
| **RAF-DB** | 29,672 | 100×100 | 7 | Better labels, higher quality | Requires application |
| **CK+** | 981 | 640×490 | 8 | Very accurate labels | Very small, not diverse |
| **JAFFE** | 213 | 256×256 | 7 | High quality | Tiny, only Japanese females |

### Recommendation

Start with **FER-2013** for initial development and prototyping. Later, consider **AffectNet** for production-quality training if higher accuracy is needed.

---

## ⚠️ Dataset Limitations & Ethical Considerations

1. **Label Noise:** ~35% disagreement rate among human annotators on FER-2013
2. **Demographic Bias:** Dataset may not represent all ethnicities, ages, and genders equally
3. **Cultural Bias:** Emotions are expressed differently across cultures
4. **Consent:** Ensure dataset usage complies with its license terms
5. **Privacy:** Never store or process real-world face data without proper consent

---

<p align="center"><i>This document will be updated as the dataset pipeline is built and refined.</i></p>

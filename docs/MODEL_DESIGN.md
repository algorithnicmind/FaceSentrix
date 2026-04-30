<h1 align="center">🧠 FaceSentrix — CNN Model Design</h1>

<p align="center">
  <i>Model architecture decisions, training strategy, and optimization notes</i>
</p>

---

## 📝 Design Philosophy

The model is designed with these principles:
1. **Simplicity First** — Start with a proven custom CNN before exploring complex architectures
2. **Real-Time Capable** — Must run at ≥15 FPS on CPU with face detection overhead
3. **Lightweight** — Target model size < 50MB for easy distribution
4. **Interpretable** — Clear emotion probability output, not just a single label

---

## 🏛️ Model Architecture (v1 — Custom CNN)

### Layer-by-Layer Breakdown

```
Input: (48, 48, 1) — Grayscale face image, normalized [0, 1]

════════════════════════════════════════════════════════════════

Block 1: Feature Extraction (Low-Level)
├── Conv2D(32 filters, 3×3, padding='same')
├── BatchNormalization()
├── ReLU Activation
├── Conv2D(32 filters, 3×3, padding='same')
├── BatchNormalization()
├── ReLU Activation
├── MaxPooling2D(2×2)
└── Dropout(0.25)
    Output: (24, 24, 32)

════════════════════════════════════════════════════════════════

Block 2: Feature Extraction (Mid-Level)
├── Conv2D(64 filters, 3×3, padding='same')
├── BatchNormalization()
├── ReLU Activation
├── Conv2D(64 filters, 3×3, padding='same')
├── BatchNormalization()
├── ReLU Activation
├── MaxPooling2D(2×2)
└── Dropout(0.25)
    Output: (12, 12, 64)

════════════════════════════════════════════════════════════════

Block 3: Feature Extraction (High-Level)
├── Conv2D(128 filters, 3×3, padding='same')
├── BatchNormalization()
├── ReLU Activation
├── Conv2D(128 filters, 3×3, padding='same')
├── BatchNormalization()
├── ReLU Activation
├── MaxPooling2D(2×2)
└── Dropout(0.25)
    Output: (6, 6, 128)

════════════════════════════════════════════════════════════════

Block 4: Feature Extraction (Deep)
├── Conv2D(256 filters, 3×3, padding='same')
├── BatchNormalization()
├── ReLU Activation
├── Conv2D(256 filters, 3×3, padding='same')
├── BatchNormalization()
├── ReLU Activation
├── MaxPooling2D(2×2)
└── Dropout(0.25)
    Output: (3, 3, 256)

════════════════════════════════════════════════════════════════

Classification Head:
├── Flatten()                          → (2304,)
├── Dense(512)
├── BatchNormalization()
├── ReLU Activation
├── Dropout(0.5)
├── Dense(256)
├── BatchNormalization()
├── ReLU Activation
├── Dropout(0.5)
└── Dense(7, activation='softmax')     → (7,)

════════════════════════════════════════════════════════════════

Output: 7 class probabilities
[Angry, Disgust, Fear, Happy, Sad, Surprise, Neutral]
```

### Model Summary

| Property | Value |
|----------|-------|
| **Total Parameters** | ~2.5M (estimated) |
| **Trainable Parameters** | ~2.5M |
| **Input Shape** | (48, 48, 1) |
| **Output Shape** | (7,) |
| **Model Size (saved)** | ~30MB (.h5) |
| **Inference Time** | ~10-15ms (CPU) |

---

## 🏛️ Model Architecture (v2 — Transfer Learning)

For improved accuracy, a transfer learning approach can be used:

```
Base Model: MobileNetV2 (pretrained on ImageNet)
├── Input: (48, 48, 3) — Upsample to (96, 96, 3) or use (224, 224, 3)
├── MobileNetV2 layers (frozen initially)
├── GlobalAveragePooling2D()
├── Dense(256, ReLU)
├── Dropout(0.5)
├── Dense(128, ReLU)
├── Dropout(0.3)
└── Dense(7, softmax)
```

**Strategy:**
1. **Phase 1:** Freeze base model, train only classifier head (5 epochs)
2. **Phase 2:** Unfreeze last 30 layers, fine-tune with low LR (20 epochs)
3. **Phase 3:** Unfreeze all layers, train with very low LR (10 epochs)

---

## ⚙️ Training Configuration

### Hyperparameters

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| **Optimizer** | Adam | Adaptive learning rate, good default |
| **Learning Rate** | 0.0001 | Start low to avoid overshooting |
| **Batch Size** | 64 | Balance between memory and gradient stability |
| **Epochs** | 50-100 | With early stopping for convergence |
| **Loss Function** | Categorical Cross-Entropy | Standard for multi-class classification |
| **Weight Init** | He Normal | Best for ReLU activations |

### Learning Rate Schedule

```
Epoch  1-10:   lr = 0.0001   (warmup & initial learning)
Epoch 11-30:   lr = 0.0001   (main training)
Epoch 31+:     lr = reduced   (ReduceLROnPlateau, factor=0.5, patience=5)
```

### Callbacks

| Callback | Configuration | Purpose |
|----------|---------------|---------|
| `ModelCheckpoint` | `monitor='val_accuracy', save_best_only=True` | Save best model |
| `EarlyStopping` | `patience=10, restore_best_weights=True` | Prevent overfitting |
| `ReduceLROnPlateau` | `factor=0.5, patience=5, min_lr=1e-7` | Adaptive LR |
| `TensorBoard` | `log_dir='./logs'` | Training visualization |
| `CSVLogger` | `filename='training_log.csv'` | Training history |

---

## 📐 Data Preprocessing

### Input Pipeline

```
Raw Image (varies) 
    → Resize to 48×48
    → Convert to Grayscale
    → Normalize: pixel / 255.0
    → Reshape to (48, 48, 1)
    → Ready for model input
```

### Augmentation Strategy

| Augmentation | Range | Probability |
|-------------|-------|-------------|
| Horizontal Flip | — | 50% |
| Rotation | ±15° | Per-sample |
| Width Shift | ±10% | Per-sample |
| Height Shift | ±10% | Per-sample |
| Zoom | ±10% | Per-sample |
| Brightness | ±20% | Per-sample |

**Not Applied:**
- Vertical flip (faces don't appear upside down normally)
- Heavy rotation (>30° distorts expressions)
- Color jitter (grayscale input)

---

## 📊 Evaluation Metrics

### Primary Metrics

| Metric | Description | Target |
|--------|-------------|--------|
| **Accuracy** | Overall correct predictions / total | ≥ 65% |
| **Weighted F1** | F1 score weighted by class frequency | ≥ 0.63 |
| **Macro F1** | Average F1 across all classes (unweighted) | ≥ 0.55 |

### Per-Class Expected Performance

| Emotion | Expected Accuracy | Difficulty | Notes |
|---------|-------------------|------------|-------|
| Happy | ~85% | Easy | Most distinctive features (smile) |
| Surprise | ~75% | Easy | Open mouth, raised eyebrows |
| Angry | ~60% | Medium | Can overlap with disgust |
| Neutral | ~65% | Medium | Default state, subtle |
| Sad | ~55% | Hard | Subtle features, overlaps with neutral |
| Fear | ~50% | Hard | Similar to surprise |
| Disgust | ~45% | Very Hard | Rarest class, hardest to distinguish |

### Known Challenges

1. **Label Noise in FER-2013:** Human annotator agreement is only ~65%
2. **Class Imbalance:** "Disgust" is severely under-represented
3. **Ambiguous Expressions:** Some expressions naturally overlap (fear/surprise, sad/neutral)
4. **Demographic Bias:** Model performance may vary across demographics

---

## 🔧 Optimization Roadmap

### Stage 1: Baseline Model
- Custom CNN as described above
- Target: 60-65% accuracy

### Stage 2: Architecture Improvements
- Add residual connections (skip connections)
- Try Squeeze-and-Excitation blocks
- Experiment with attention mechanisms
- Target: 65-70% accuracy

### Stage 3: Transfer Learning
- Fine-tune MobileNetV2 or EfficientNetB0
- Use pretrained face recognition features
- Target: 68-72% accuracy

### Stage 4: Ensemble
- Combine multiple models (vote/average)
- Blend custom CNN + transfer learning model
- Target: 70-75% accuracy

---

## 📂 Model Files

| File | Format | Description |
|------|--------|-------------|
| `emotion_model.h5` | HDF5 | Full Keras model (weights + architecture) |
| `emotion_model.tflite` | TFLite | Optimized for edge/mobile deployment |
| `training_history.json` | JSON | Epoch-wise metrics (loss, accuracy) |
| `model_config.json` | JSON | Hyperparameters and configuration |

---

<p align="center"><i>This document will be updated as the model evolves through training iterations.</i></p>

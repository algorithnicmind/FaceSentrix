<h1 align="center">📋 FaceSentrix — Project Roadmap & TODO</h1>

<p align="center">
  <i>Step-by-step development plan from zero to deployment</i><br/>
  <b>Last Updated:</b> March 25, 2026
</p>

---

## 🗺️ Development Phases Overview

```
Phase 1 ──▶ Phase 2 ──▶ Phase 3 ──▶ Phase 4 ──▶ Phase 5 ──▶ Phase 6
Project     Data &       Model       Real-Time   Web App &   Polish &
Setup       Preprocessing Training   Detection   Deployment  Release
(Week 1)    (Week 2)     (Week 3-4)  (Week 5)    (Week 6-7)  (Week 8)
```

---

## 📌 Phase 1: Project Setup & Environment Configuration
> **Goal:** Set up the repository, development environment, and project skeleton  
> **Duration:** ~3-4 days  
> **Priority:** 🔴 Critical

### Tasks

- [x] **1.1** Initialize GitHub repository with README and LICENSE
  - Create repo on GitHub
  - Add CC0 1.0 License
  - Write initial README.md with project description
  - **Commit:** `init: initialize repository with README and LICENSE`

- [x] **1.2** Create `.gitignore` file
  - Add standard Python ignores (`__pycache__/`, `*.pyc`, `.env`)
  - Add data directory ignores (`data/raw/`, `data/processed/`)
  - Add model file ignores (`*.h5`, `*.tflite`, `*.pkl`)
  - Add IDE ignores (`.vscode/`, `.idea/`)
  - Add `docs/` directory to gitignore
  - **Commit:** `chore: add comprehensive .gitignore`

- [x] **1.3** Set up Python virtual environment
  - Create `venv` using `python -m venv venv`
  - Document activation steps for Windows/Linux/macOS
  - **Commit:** (no commit — venv is gitignored)

- [x] **1.4** Create `requirements.txt` with core dependencies
  ```
  opencv-python>=4.8.0
  tensorflow>=2.13.0
  keras>=2.13.0
  numpy>=1.24.0
  pandas>=2.0.0
  matplotlib>=3.7.0
  seaborn>=0.12.0
  scikit-learn>=1.3.0
  Pillow>=10.0.0
  tqdm>=4.65.0
  ```
  - **Commit:** `build: add requirements.txt with core dependencies`

- [x] **1.5** Create project directory structure
  - Create `src/` directory with `__init__.py`
  - Create `training/` directory
  - Create `data/raw/` and `data/processed/` directories
  - Create `models/` directory
  - Create `notebooks/` directory
  - Create `tests/` directory
  - Create `assets/` directory
  - Create `app/` directory
  - **Commit:** `chore: create project directory structure`

- [x] **1.6** Create `setup.py` or `pyproject.toml`
  - Define package metadata (name, version, author)
  - Define entry points
  - **Commit:** `build: add setup.py with package metadata`

- [x] **1.7** Add project banner/assets
  - Create or generate project banner image
  - Place in `assets/` directory
  - **Commit:** `chore: add project banner asset`

---

## 📌 Phase 2: Dataset Acquisition & Preprocessing
> **Goal:** Download, explore, clean, and preprocess the FER-2013 dataset  
> **Duration:** ~4-5 days  
> **Priority:** 🔴 Critical

### Tasks

- [x] **2.1** Research and select dataset
  - Primary: FER-2013 (35,887 grayscale 48×48 images)
  - Alternative: AffectNet, RAF-DB
  - Document dataset comparison in `docs/DATASET_GUIDE.md`
  - **Commit:** `docs: add dataset comparison and selection notes`

- [x] **2.2** Create dataset download script
  - Write `training/preprocess.py` with `--download` flag
  - Auto-download FER-2013 from Kaggle (using `kaggle` API or manual)
  - Verify file integrity (checksum validation)
  - **Commit:** `feat: add dataset download utility`

- [x] **2.3** Explore the dataset (Jupyter Notebook)
  - Create `notebooks/01_data_exploration.ipynb`
  - Visualize class distribution (bar chart)
  - Show sample images per emotion class
  - Identify class imbalance issues
  - Compute basic statistics (mean, std, pixel range)
  - **Commit:** `feat: add data exploration notebook`

- [x] **2.4** Implement data preprocessing pipeline
  - Write `training/preprocess.py` — main preprocessing logic
  - Convert CSV pixel data → image arrays
  - Normalize pixel values to [0, 1] range
  - Reshape images to (48, 48, 1)
  - Split into train/validation/test sets (80/10/10)
  - Save processed arrays as `.npy` files
  - **Commit:** `feat: implement data preprocessing pipeline`

- [x] **2.5** Implement data augmentation
  - Write `training/augment.py`
  - Apply augmentations:
    - Random rotation (±15°)
    - Horizontal flip
    - Random zoom (±10%)
    - Random brightness adjustment
    - Random shift (width & height)
  - Use `tf.keras.preprocessing.image.ImageDataGenerator` or `albumentations`
  - **Commit:** `feat: implement data augmentation strategies`

- [x] **2.6** Handle class imbalance
  - Implement oversampling (SMOTE or random oversampling)
  - Alternatively: compute class weights for loss function
  - Validate balanced distribution
  - **Commit:** `feat: add class imbalance handling`

- [x] **2.7** Create data loading utilities
  - Write helper functions in `src/utils.py`
  - `load_dataset()` — loads preprocessed data
  - `get_class_labels()` — returns emotion label mapping
  - `visualize_samples()` — displays sample images
  - **Commit:** `feat: add data loading utility functions`

---

## 📌 Phase 3: CNN Model Design & Training
> **Goal:** Design, build, train, and evaluate the CNN emotion classifier  
> **Duration:** ~7-10 days  
> **Priority:** 🔴 Critical

### Tasks

- [ ] **3.1** Research CNN architectures for emotion recognition
  - Study existing approaches (VGG-like, ResNet, MobileNet)
  - Document findings in `docs/MODEL_DESIGN.md`
  - Choose architecture approach (custom CNN vs transfer learning)
  - **Commit:** `docs: document CNN architecture research`

- [x] **3.2** Design custom CNN architecture
  - Write `src/emotion_classifier.py` — model definition
  - Architecture plan:
    ```
    Input (48×48×1)
    → Conv2D(32, 3×3) → BatchNorm → ReLU → MaxPool(2×2) → Dropout(0.25)
    → Conv2D(64, 3×3) → BatchNorm → ReLU → MaxPool(2×2) → Dropout(0.25)
    → Conv2D(128, 3×3) → BatchNorm → ReLU → MaxPool(2×2) → Dropout(0.25)
    → Conv2D(256, 3×3) → BatchNorm → ReLU → MaxPool(2×2) → Dropout(0.25)
    → Flatten
    → Dense(512) → BatchNorm → ReLU → Dropout(0.5)
    → Dense(256) → BatchNorm → ReLU → Dropout(0.5)
    → Dense(7, softmax)
    ```
  - Print model summary (total params, layer shapes)
  - **Commit:** `feat: design CNN model architecture`

- [x] **3.3** Implement training script
  - Write `training/train.py`
  - Configure:
    - Optimizer: Adam (lr=0.0001)
    - Loss: Categorical Cross-Entropy
    - Metrics: Accuracy, Precision, Recall, F1
  - Add callbacks:
    - `ModelCheckpoint` — save best model
    - `EarlyStopping` — patience=10
    - `ReduceLROnPlateau` — reduce LR on plateau
    - `TensorBoard` — training visualization
  - Log training history to `models/training_history.json`
  - **Commit:** `feat: implement model training script`

- [x] **3.4** Train the model
  - Run training with 50-100 epochs
  - Monitor training/validation loss & accuracy curves
  - Save best model as `models/emotion_model.h5`
  - **Commit:** `feat: train emotion classification model`

- [x] **3.5** Create training notebook
  - Write `notebooks/02_model_training.ipynb`
  - Visualize training progress (loss & accuracy curves)
  - Show learning rate schedule
  - Document hyperparameter choices
  - **Commit:** `feat: add model training notebook`

- [x] **3.6** Model evaluation
  - Write `training/evaluate.py`
  - Generate classification report (precision, recall, F1 per class)
  - Create confusion matrix (heatmap visualization)
  - Calculate overall accuracy, weighted F1
  - Test on unseen test set
  - **Commit:** `feat: implement model evaluation metrics`

- [x] **3.7** Create evaluation notebook
  - Write `notebooks/03_evaluation.ipynb`
  - Visualize confusion matrix
  - Show per-class performance
  - Identify worst-performing classes
  - Show misclassified examples
  - **Commit:** `feat: add evaluation analysis notebook`

- [x] **3.8** Experiment with transfer learning (optional)
  - Try MobileNetV2 / VGG16 as base model
  - Fine-tune on FER-2013
  - Compare accuracy with custom CNN
  - **Commit:** `feat: experiment with transfer learning approach`

- [x] **3.9** Model optimization
  - Prune unnecessary layers
  - Quantize model (float16 or int8)
  - Convert to TFLite format: `models/emotion_model.tflite`
  - Benchmark inference speed (CPU vs GPU)
  - **Commit:** `feat: optimize and convert model to TFLite`

---

## 📌 Phase 4: Face Detection & Real-Time Pipeline
> **Goal:** Build the face detection module and real-time camera pipeline  
> **Duration:** ~5-6 days  
> **Priority:** 🔴 Critical

### Tasks

- [x] **4.1** Implement face detection module
  - Write `src/face_detector.py`
  - Support multiple detection backends:
    - Haar Cascade (fast, less accurate)
    - DNN-based (OpenCV's `cv2.dnn`, more accurate)
  - Functions:
    - `detect_faces(frame)` → returns list of bounding boxes
    - `draw_faces(frame, faces)` → draws rectangles on frame
  - Add face region extraction (crop detected face)
  - **Commit:** `feat: implement face detection module`

- [x] **4.2** Implement webcam capture module
  - Write `src/camera.py`
  - Functions:
    - `start_camera()` — initialize webcam capture
    - `read_frame()` — read single frame
    - `release_camera()` — cleanup resources
  - Handle camera not found errors gracefully
  - Support multiple camera indices
  - **Commit:** `feat: implement webcam capture module`

- [x] **4.3** Implement emotion prediction module
  - Update `src/emotion_classifier.py` with inference methods
  - Functions:
    - `load_model(model_path)` — load trained model
    - `preprocess_face(face_img)` — resize, normalize, reshape
    - `predict_emotion(face_img)` — returns (label, confidence)
  - Batch prediction support for multiple faces
  - **Commit:** `feat: implement emotion prediction inference`

- [x] **4.4** Implement visualization overlay module
  - Write `src/visualizer.py`
  - Functions:
    - `draw_bounding_box(frame, box, label, confidence)` — fancy box with label
    - `draw_emotion_bar(frame, emotions)` — confidence bar chart overlay
    - `add_fps_counter(frame, fps)` — FPS display
  - Color-coded boxes per emotion
  - Smooth label transitions (avoid flickering)
  - **Commit:** `feat: implement visualization overlay module`

- [x] **4.5** Build the real-time detection pipeline
  - Create main entry point `main.py` or update `src/camera.py`
  - Pipeline:
    ```
    Camera Frame → Face Detection → Crop Faces → 
    Preprocess → Predict Emotion → Draw Overlay → Display
    ```
  - Handle multiple faces simultaneously
  - Add FPS counter
  - Keyboard controls: `q` to quit, `s` to screenshot
  - **Commit:** `feat: build real-time emotion detection pipeline`

- [x] **4.6** Performance optimization
  - Implement frame skipping (detect every N frames)
  - Add face tracking between detection frames
  - Multi-threading: separate capture and processing threads
  - Optimize image preprocessing pipeline
  - **Commit:** `perf: optimize real-time detection performance`

- [x] **4.7** Write unit tests for core modules
  - Write `tests/test_face_detector.py`
  - Write `tests/test_emotion_classifier.py`
  - Write `tests/test_camera.py`
  - Test edge cases (no face, multiple faces, partial face)
  - **Commit:** `test: add unit tests for core modules`

---

## 📌 Phase 5: Web Application & API (Optional)
> **Goal:** Build a web interface or API to serve the emotion detection system  
> **Duration:** ~5-7 days  
> **Priority:** 🟡 Medium

### Tasks

- [ ] **5.1** Choose web framework
  - Option A: **Streamlit** (quick demo, minimal code)
  - Option B: **Flask** (REST API + custom frontend)
  - Option C: **FastAPI** (async, high performance)
  - **Commit:** `docs: document web framework selection`

- [ ] **5.2** Build API endpoints (Flask/FastAPI)
  - Write `app/app.py`
  - Endpoints:
    - `POST /api/detect` — upload image, return detected emotions
    - `GET /api/stream` — real-time stream with WebSocket
    - `GET /api/health` — health check
  - **Commit:** `feat: implement emotion detection API`

- [ ] **5.3** Build web frontend
  - Create `app/templates/index.html`
  - Features:
    - Live camera feed display
    - Upload image for analysis
    - Emotion results dashboard
    - History/log of detections
  - **Commit:** `feat: build web frontend interface`

- [ ] **5.4** Implement image upload detection
  - Accept image upload via web form
  - Process uploaded image through pipeline
  - Return annotated image + emotion data (JSON)
  - **Commit:** `feat: implement image upload emotion detection`

- [ ] **5.5** Add WebSocket for live stream
  - Stream webcam feed through WebSocket
  - Real-time emotion updates on web dashboard
  - **Commit:** `feat: add WebSocket live streaming support`

- [ ] **5.6** Dockerize the application
  - Write `Dockerfile`
  - Write `docker-compose.yml`
  - Optimize image size (multi-stage build)
  - **Commit:** `build: add Docker configuration`

---

## 📌 Phase 6: Documentation, Testing & Release
> **Goal:** Finalize documentation, comprehensive testing, and project release  
> **Duration:** ~3-4 days  
> **Priority:** 🟡 Medium

### Tasks

- [ ] **6.1** Write comprehensive README.md
  - Project overview & motivation
  - Installation guide (step-by-step)
  - Usage examples (with screenshots/GIFs)
  - Model performance results
  - Contributing guidelines
  - **Commit:** `docs: write comprehensive project README`

- [ ] **6.2** Add demo GIF/video
  - Record real-time detection in action
  - Create GIF from recording
  - Embed in README.md
  - **Commit:** `docs: add demo GIF to README`

- [ ] **6.3** Write API documentation
  - Document all endpoints
  - Add request/response examples
  - Include Postman collection (optional)
  - **Commit:** `docs: add API documentation`

- [ ] **6.4** Add comprehensive tests
  - Achieve ≥80% code coverage
  - Add integration tests
  - Add edge case tests
  - Set up CI with GitHub Actions
  - **Commit:** `test: add comprehensive test suite`

- [ ] **6.5** Create GitHub Actions CI/CD
  - Write `.github/workflows/ci.yml`
  - Run tests on every push
  - Lint check (flake8/pylint)
  - Type check (mypy)
  - **Commit:** `ci: add GitHub Actions CI workflow`

- [ ] **6.6** Add code quality tools
  - Add `pyproject.toml` with:
    - `black` — code formatting
    - `flake8` — linting
    - `mypy` — type checking
    - `isort` — import sorting
  - Add pre-commit hooks
  - **Commit:** `chore: add code quality tools configuration`

- [ ] **6.7** Create release
  - Tag version `v1.0.0`
  - Write release notes
  - Upload trained model as release asset
  - **Commit:** `chore: prepare v1.0.0 release`

---

## 📊 Progress Tracker

| Phase | Status | Progress | Start Date | End Date |
|-------|--------|----------|------------|----------|
| Phase 1: Project Setup | 🟡 In Progress | ██░░░░░░░░ 20% | Mar 25, 2026 | — |
| Phase 2: Data & Preprocessing | ⬜ Not Started | ░░░░░░░░░░ 0% | — | — |
| Phase 3: Model Training | ⬜ Not Started | ░░░░░░░░░░ 0% | — | — |
| Phase 4: Real-Time Detection | ⬜ Not Started | ░░░░░░░░░░ 0% | — | — |
| Phase 5: Web Application | ⬜ Not Started | ░░░░░░░░░░ 0% | — | — |
| Phase 6: Docs & Release | ⬜ Not Started | ░░░░░░░░░░ 0% | — | — |

---

## 🏷️ Commit Convention

All commits follow the [Conventional Commits](https://www.conventionalcommits.org/) specification:

| Prefix | Usage |
|--------|-------|
| `init:` | Initial setup / repository creation |
| `feat:` | New feature or functionality |
| `fix:` | Bug fix |
| `docs:` | Documentation changes |
| `style:` | Code formatting (no logic change) |
| `refactor:` | Code restructuring (no feature change) |
| `test:` | Adding or updating tests |
| `perf:` | Performance improvements |
| `build:` | Build system / dependency changes |
| `ci:` | CI/CD configuration |
| `chore:` | Maintenance tasks |

**Rule:** Every single change — no matter how small — gets its own commit.

---

## 📝 Notes

- **Dataset:** FER-2013 is free and commonly used, but has known label noise (~65% human accuracy ceiling)
- **GPU:** Training is recommended on GPU. Google Colab (free) can be used if local GPU is unavailable
- **Python Version:** Target Python 3.8+ for broad compatibility
- **Model Architecture:** Start simple (custom CNN), then iterate with transfer learning if needed
- **Testing:** Write tests as you build, not after. TDD approach preferred
- **Commits:** One change = one commit. Keep the git history clean and traceable

---

<p align="center">
  <b>🎯 Total Tasks: 38 | ✅ Completed: 0 | 🔄 In Progress: 2 | ⬜ Remaining: 36</b>
</p>

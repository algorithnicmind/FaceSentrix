<h1 align="center">🏗️ FaceSentrix — System Architecture</h1>

<p align="center">
  <i>Detailed architecture documentation for the emotion detection pipeline</i>
</p>

---

## 📐 High-Level Architecture

The system follows a **modular pipeline architecture** where each component has a single responsibility and communicates through well-defined interfaces.

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        FaceSentrix System Architecture                  │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ┌────────────┐   ┌──────────────┐   ┌───────────────┐   ┌──────────┐  │
│  │   Input     │   │    Face      │   │   Emotion     │   │  Output  │  │
│  │   Layer     │──▶│  Detection   │──▶│  Classifier   │──▶│  Layer   │  │
│  │            │   │   Layer      │   │    Layer      │   │          │  │
│  └────────────┘   └──────────────┘   └───────────────┘   └──────────┘  │
│       │                 │                    │                  │        │
│  ┌────▼────┐      ┌────▼─────┐       ┌─────▼──────┐     ┌────▼────┐   │
│  │ Webcam  │      │ OpenCV   │       │ CNN Model  │     │ Display │   │
│  │ Image   │      │ Haar/DNN │       │ TensorFlow │     │ Web App │   │
│  │ Video   │      │          │       │            │     │ API     │   │
│  └─────────┘      └──────────┘       └────────────┘     └─────────┘   │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 🧩 Component Breakdown

### 1. Input Layer (`src/camera.py`)

Responsible for capturing visual input from various sources.

```
┌─────────────────────────────────────┐
│           Input Layer               │
├─────────────────────────────────────┤
│                                     │
│  Sources:                           │
│  ├── Webcam (real-time)             │
│  ├── Video file (pre-recorded)      │
│  ├── Image file (static)            │
│  └── API upload (HTTP POST)         │
│                                     │
│  Output: Raw BGR frame (numpy)      │
│  Format: (height, width, 3)         │
│                                     │
└─────────────────────────────────────┘
```

**Key Responsibilities:**
- Initialize and manage camera/video capture
- Handle frame buffering and timing
- Support multiple input sources (camera index, file path, URL)
- Graceful error handling (camera not found, permission denied)
- Resource cleanup on exit

**Interface:**
```python
class CameraCapture:
    def __init__(self, source=0, width=640, height=480):
        """Initialize capture from webcam (int) or file (str)"""
    
    def read_frame(self) -> Tuple[bool, np.ndarray]:
        """Read a single frame. Returns (success, frame)"""
    
    def release(self) -> None:
        """Release camera resources"""
    
    @property
    def fps(self) -> float:
        """Current frames per second"""
```

---

### 2. Face Detection Layer (`src/face_detector.py`)

Locates and extracts face regions from input frames.

```
┌─────────────────────────────────────────────────┐
│            Face Detection Layer                  │
├─────────────────────────────────────────────────┤
│                                                  │
│  Input: Raw BGR frame                            │
│                                                  │
│  Processing:                                     │
│  ├── Convert to grayscale                        │
│  ├── Apply face detector                         │
│  ├── Filter by confidence threshold              │
│  └── Extract face ROIs                           │
│                                                  │
│  Output: List of FaceRegion objects              │
│  ├── bounding_box: (x, y, w, h)                 │
│  ├── face_image: cropped face (numpy)            │
│  └── confidence: detection confidence            │
│                                                  │
└─────────────────────────────────────────────────┘
```

**Detection Backends:**

| Backend | Speed | Accuracy | Use Case |
|---------|-------|----------|----------|
| Haar Cascade | ⚡ Fast | 🟡 Medium | CPU-only, real-time |
| OpenCV DNN (SSD) | 🔄 Medium | 🟢 High | Balanced performance |
| MTCNN | 🐢 Slow | 🟢 Very High | Highest accuracy needed |

**Interface:**
```python
class FaceDetector:
    def __init__(self, backend="haar", confidence_threshold=0.5):
        """Initialize detector with chosen backend"""
    
    def detect(self, frame: np.ndarray) -> List[FaceRegion]:
        """Detect all faces in frame"""
    
    def extract_faces(self, frame, faces) -> List[np.ndarray]:
        """Extract and preprocess face ROIs"""
```

---

### 3. Emotion Classification Layer (`src/emotion_classifier.py`)

Classifies preprocessed face images into emotion categories using the trained CNN.

```
┌─────────────────────────────────────────────────┐
│         Emotion Classification Layer             │
├─────────────────────────────────────────────────┤
│                                                  │
│  Input: Cropped face image                       │
│                                                  │
│  Preprocessing:                                  │
│  ├── Resize to 48×48                             │
│  ├── Convert to grayscale                        │
│  ├── Normalize to [0, 1]                         │
│  └── Reshape to (1, 48, 48, 1)                   │
│                                                  │
│  Model Inference:                                │
│  ├── Forward pass through CNN                    │
│  └── Softmax output (7 probabilities)            │
│                                                  │
│  Output: EmotionResult                           │
│  ├── label: "Happy"                              │
│  ├── confidence: 0.87                            │
│  └── all_probabilities: {emotion: prob, ...}     │
│                                                  │
└─────────────────────────────────────────────────┘
```

**Interface:**
```python
class EmotionClassifier:
    EMOTIONS = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]
    
    def __init__(self, model_path: str):
        """Load trained model from .h5 or .tflite file"""
    
    def preprocess(self, face_image: np.ndarray) -> np.ndarray:
        """Preprocess face for model input"""
    
    def predict(self, face_image: np.ndarray) -> EmotionResult:
        """Predict emotion for single face"""
    
    def predict_batch(self, face_images: List[np.ndarray]) -> List[EmotionResult]:
        """Predict emotions for multiple faces"""
```

---

### 4. Visualization Layer (`src/visualizer.py`)

Renders detection results back onto the video frame.

```
┌─────────────────────────────────────────────────┐
│           Visualization Layer                    │
├─────────────────────────────────────────────────┤
│                                                  │
│  Overlay Elements:                               │
│  ├── Bounding boxes (color-coded by emotion)     │
│  ├── Emotion labels with confidence %            │
│  ├── Confidence bar chart (mini overlay)         │
│  ├── FPS counter (top-left)                      │
│  └── Status indicators                           │
│                                                  │
│  Emotion Colors:                                 │
│  ├── Angry   → 🔴 Red     (#FF4444)             │
│  ├── Disgust → 🟣 Purple  (#AA44FF)             │
│  ├── Fear    → 🟠 Orange  (#FF8844)             │
│  ├── Happy   → 🟢 Green   (#44FF44)             │
│  ├── Sad     → 🔵 Blue    (#4444FF)             │
│  ├── Surprise→ 🟡 Yellow  (#FFFF44)             │
│  └── Neutral → ⚪ Gray    (#CCCCCC)             │
│                                                  │
└─────────────────────────────────────────────────┘
```

---

### 5. Pipeline Orchestrator (`main.py`)

Coordinates all components in the real-time detection loop.

```python
# Pseudocode for main pipeline
def main():
    camera = CameraCapture(source=0)
    detector = FaceDetector(backend="haar")
    classifier = EmotionClassifier(model_path="models/emotion_model.h5")
    visualizer = Visualizer()
    
    while True:
        success, frame = camera.read_frame()
        if not success:
            break
        
        faces = detector.detect(frame)
        
        for face in faces:
            emotion = classifier.predict(face.face_image)
            visualizer.draw_result(frame, face, emotion)
        
        visualizer.draw_fps(frame, camera.fps)
        cv2.imshow("FaceSentrix", frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    camera.release()
```

---

## 📦 Data Flow Diagram

```
                    ┌──────────────┐
                    │   Raw Frame  │
                    │  (640×480×3) │
                    └──────┬───────┘
                           │
                    ┌──────▼───────┐
                    │  Grayscale   │
                    │  Conversion  │
                    └──────┬───────┘
                           │
                    ┌──────▼───────┐
                    │    Face      │
                    │  Detection   │
                    └──────┬───────┘
                           │
              ┌────────────┼────────────┐
              ▼            ▼            ▼
        ┌──────────┐ ┌──────────┐ ┌──────────┐
        │  Face 1  │ │  Face 2  │ │  Face N  │
        │ (x,y,w,h)│ │ (x,y,w,h)│ │ (x,y,w,h)│
        └────┬─────┘ └────┬─────┘ └────┬─────┘
             │             │             │
        ┌────▼─────┐ ┌────▼─────┐ ┌────▼─────┐
        │  Crop &  │ │  Crop &  │ │  Crop &  │
        │  Resize  │ │  Resize  │ │  Resize  │
        │ (48×48×1)│ │ (48×48×1)│ │ (48×48×1)│
        └────┬─────┘ └────┬─────┘ └────┬─────┘
             │             │             │
        ┌────▼─────┐ ┌────▼─────┐ ┌────▼─────┐
        │   CNN    │ │   CNN    │ │   CNN    │
        │ Predict  │ │ Predict  │ │ Predict  │
        └────┬─────┘ └────┬─────┘ └────┬─────┘
             │             │             │
              └────────────┼────────────┘
                           │
                    ┌──────▼───────┐
                    │   Annotated  │
                    │    Frame     │
                    └──────┬───────┘
                           │
                    ┌──────▼───────┐
                    │   Display /  │
                    │   Stream     │
                    └──────────────┘
```

---

## 🔒 Error Handling Strategy

| Component | Error | Handling |
|-----------|-------|----------|
| Camera | Device not found | Show error message, list available cameras |
| Camera | Permission denied | Display system permission instructions |
| Face Detection | No face detected | Skip classification, show "No face detected" |
| Face Detection | Partial face | Apply minimum size threshold, skip small faces |
| Model | Model file not found | Show download instructions, fallback to default |
| Model | Invalid input shape | Auto-resize and log warning |
| Pipeline | Low FPS (<5) | Reduce resolution, increase frame skip interval |

---

## 🧵 Threading Model

```
┌─────────────────────────────────────────────────┐
│                Threading Model                   │
├─────────────────────────────────────────────────┤
│                                                  │
│  Thread 1 (Capture):                             │
│  └── Continuously reads frames into buffer       │
│                                                  │
│  Thread 2 (Processing) — Main:                   │
│  ├── Takes latest frame from buffer              │
│  ├── Runs face detection                         │
│  ├── Runs emotion classification                 │
│  └── Updates display                             │
│                                                  │
│  Thread 3 (UI/API) — Optional:                   │
│  └── Serves web interface / API responses        │
│                                                  │
│  Synchronization: Queue + Lock                   │
│                                                  │
└─────────────────────────────────────────────────┘
```

---

## 📊 Performance Considerations

1. **Frame Skipping:** Run face detection every 3-5 frames, interpolate positions between
2. **ROI Caching:** Cache face positions and only re-detect when significant movement occurs
3. **Model Quantization:** Use TFLite with int8 quantization for 2-4x speedup
4. **Resolution Scaling:** Process at lower resolution, display at full resolution
5. **Batch Processing:** Batch multiple face crops for single model inference call

---

<p align="center"><i>This document is a living reference. It will be updated as the architecture evolves.</i></p>

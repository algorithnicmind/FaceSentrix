<h1 align="center">🚀 FaceSentrix — Deployment Guide</h1>

<p align="center">
  <i>Instructions for deploying the emotion detection system in various environments</i>
</p>

---

## 📋 Deployment Options

| Option | Complexity | Best For | Requirements |
|--------|-----------|----------|--------------|
| **Local Desktop** | ⭐ Easy | Development, personal use | Python, webcam |
| **Streamlit App** | ⭐⭐ Medium | Demo, presentation | Python, Streamlit |
| **Flask/FastAPI** | ⭐⭐⭐ Medium | Web service, API | Python, web server |
| **Docker** | ⭐⭐⭐ Medium | Reproducible deployment | Docker Engine |
| **Cloud (AWS/GCP)** | ⭐⭐⭐⭐ Advanced | Production, scalable | Cloud account |
| **Edge (Raspberry Pi)** | ⭐⭐⭐⭐ Advanced | IoT, embedded | Raspberry Pi, TFLite |

---

## 🖥️ Option 1: Local Desktop Deployment

### Prerequisites

- Python 3.8+
- Webcam (built-in or USB)
- Minimum 4GB RAM

### Steps

```bash
# 1. Clone the repository
git clone https://github.com/algorithnicmind/FaceSentrix.git
cd FaceSentrix

# 2. Create virtual environment
python -m venv venv

# 3. Activate virtual environment
# Windows:
venv\Scripts\activate
# Linux/macOS:
source venv/bin/activate

# 4. Install dependencies
pip install -r requirements.txt

# 5. Download the trained model (if pre-trained model available)
# python scripts/download_model.py

# 6. Run the application
python main.py
# OR
python src/camera.py
```

### Command-Line Options

```bash
python main.py --help

Options:
  --source      Camera index or video file path (default: 0)
  --model       Path to trained model file (default: models/emotion_model.h5)
  --detector    Face detector backend: haar, dnn (default: haar)
  --confidence  Detection confidence threshold (default: 0.5)
  --no-display  Run without GUI (headless mode)
  --save-video  Save output to video file
  --fps-limit   Limit processing FPS (default: 30)
```

---

## 🌐 Option 2: Streamlit Web App

### Setup

```bash
# Install Streamlit
pip install streamlit

# Run the app
streamlit run app/streamlit_app.py
```

### Expected `streamlit_app.py` Structure

```python
import streamlit as st
import cv2
import numpy as np
from src.face_detector import FaceDetector
from src.emotion_classifier import EmotionClassifier

st.title("😊 FaceSentrix — Emotion Detection")
st.sidebar.title("Settings")

# Model loading
model = EmotionClassifier("models/emotion_model.h5")
detector = FaceDetector(backend="haar")

# Input options
input_mode = st.sidebar.radio("Input Mode", ["Upload Image", "Webcam"])

if input_mode == "Upload Image":
    uploaded_file = st.file_uploader("Upload a face image", type=["jpg", "png", "jpeg"])
    if uploaded_file is not None:
        # Process and display results
        pass

elif input_mode == "Webcam":
    st.warning("Webcam mode requires camera access")
    # WebRTC streaming implementation
    pass
```

---

## 🐳 Option 3: Docker Deployment

### Dockerfile

```dockerfile
# Stage 1: Build
FROM python:3.10-slim AS builder

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Stage 2: Runtime
FROM python:3.10-slim

WORKDIR /app

# Install OpenCV system dependencies
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /usr/local/lib/python3.10/site-packages /usr/local/lib/python3.10/site-packages
COPY . .

EXPOSE 8000

CMD ["python", "app/app.py"]
```

### docker-compose.yml

```yaml
version: '3.8'

services:
  facesentrix:
    build: .
    ports:
      - "8000:8000"
    devices:
      - /dev/video0:/dev/video0   # Camera access (Linux)
    environment:
      - MODEL_PATH=models/emotion_model.h5
      - DETECTOR_BACKEND=haar
    volumes:
      - ./models:/app/models
    restart: unless-stopped
```

### Docker Commands

```bash
# Build the image
docker build -t facesentrix .

# Run the container
docker run -p 8000:8000 --device=/dev/video0 facesentrix

# Run with docker-compose
docker-compose up -d

# View logs
docker-compose logs -f

# Stop
docker-compose down
```

---

## 🔌 Option 4: Flask/FastAPI REST API

### API Endpoints

| Method | Endpoint | Description | Input | Output |
|--------|----------|-------------|-------|--------|
| `GET` | `/api/health` | Health check | — | `{"status": "ok"}` |
| `POST` | `/api/detect` | Detect emotions in image | Image file (multipart) | JSON with emotions |
| `POST` | `/api/detect/base64` | Detect from base64 image | `{"image": "base64..."}` | JSON with emotions |
| `GET` | `/api/model/info` | Model information | — | Model metadata |

### Response Format

```json
{
  "success": true,
  "processing_time_ms": 45.2,
  "faces_detected": 2,
  "results": [
    {
      "face_id": 1,
      "bounding_box": {"x": 120, "y": 80, "w": 100, "h": 100},
      "emotion": "Happy",
      "confidence": 0.87,
      "all_emotions": {
        "Angry": 0.02,
        "Disgust": 0.01,
        "Fear": 0.03,
        "Happy": 0.87,
        "Sad": 0.02,
        "Surprise": 0.03,
        "Neutral": 0.02
      }
    },
    {
      "face_id": 2,
      "bounding_box": {"x": 340, "y": 90, "w": 95, "h": 95},
      "emotion": "Neutral",
      "confidence": 0.72,
      "all_emotions": { "..." : "..." }
    }
  ]
}
```

### Example API Call

```bash
# Upload an image for emotion detection
curl -X POST \
  -F "image=@test_face.jpg" \
  http://localhost:8000/api/detect

# Base64 encoded image
curl -X POST \
  -H "Content-Type: application/json" \
  -d '{"image": "iVBORw0KGgoAAAANSUhEUgAA..."}' \
  http://localhost:8000/api/detect/base64
```

---

## ☁️ Option 5: Cloud Deployment

### AWS (EC2 + Docker)

```bash
# 1. Launch EC2 instance (t3.medium recommended)
# 2. SSH into instance
ssh -i key.pem ec2-user@<ip-address>

# 3. Install Docker
sudo yum update -y
sudo yum install docker -y
sudo systemctl start docker

# 4. Pull and run
docker pull ghcr.io/algorithnicmind/facesentrix:latest
docker run -d -p 80:8000 facesentrix
```

### Google Cloud Run (Serverless)

```bash
# 1. Build and push to GCR
gcloud builds submit --tag gcr.io/PROJECT_ID/facesentrix

# 2. Deploy to Cloud Run
gcloud run deploy facesentrix \
  --image gcr.io/PROJECT_ID/facesentrix \
  --platform managed \
  --port 8000 \
  --memory 2Gi
```

---

## 🔧 Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `MODEL_PATH` | `models/emotion_model.h5` | Path to trained model |
| `DETECTOR_BACKEND` | `haar` | Face detection backend (haar/dnn) |
| `CONFIDENCE_THRESHOLD` | `0.5` | Minimum detection confidence |
| `MAX_FACES` | `10` | Maximum faces to process per frame |
| `PORT` | `8000` | Application port |
| `DEBUG` | `false` | Enable debug mode |
| `LOG_LEVEL` | `INFO` | Logging level |

---

## 🩺 Health Monitoring

### Health Check Endpoint

```json
GET /api/health

Response:
{
  "status": "healthy",
  "model_loaded": true,
  "model_version": "1.0.0",
  "uptime_seconds": 3600,
  "total_requests": 1542,
  "average_latency_ms": 38.5
}
```

### Monitoring Checklist

- [ ] Application health endpoint responds
- [ ] Model is loaded and functional
- [ ] Camera/input source is accessible
- [ ] Memory usage is within limits
- [ ] CPU usage is within limits
- [ ] Response latency is acceptable (<100ms)

---

## 🔒 Security Considerations

1. **Input Validation:** Always validate uploaded images (size, format, dimensions)
2. **Rate Limiting:** Implement request rate limiting on API endpoints
3. **CORS:** Configure CORS appropriately for web deployments
4. **File Upload Limits:** Set maximum file size (e.g., 10MB)
5. **No Data Storage:** Don't store uploaded face images without explicit consent
6. **HTTPS:** Always use HTTPS in production

---

## 📊 Performance Benchmarks

| Environment | FPS (Detection) | Latency (API) | Memory |
|------------|-----------------|----------------|--------|
| Local CPU (i7) | ~20 FPS | ~45ms | ~500MB |
| Local GPU (GTX 1060) | ~60 FPS | ~15ms | ~800MB |
| Docker (CPU) | ~15 FPS | ~55ms | ~600MB |
| Raspberry Pi 4 | ~5 FPS | ~200ms | ~300MB |
| AWS t3.medium | ~18 FPS | ~50ms | ~500MB |

---

<p align="center"><i>This guide will be updated with actual deployment results and configurations.</i></p>

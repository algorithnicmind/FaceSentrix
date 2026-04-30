"""
FaceSentrix - Web Application Backend (Flask)
"""
import os
import sys
import base64
import cv2
import numpy as np
from flask import Flask, render_template, request, jsonify

# Ensure root directory is in sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.face_detector import FaceDetector
from src.emotion_classifier import EmotionClassifier
from src.utils import get_class_labels

app = Flask(__name__)

# Initialize ML components
detector = FaceDetector(backend="haar")
classifier = EmotionClassifier(model_path="models/emotion_model.h5")
EMOTION_LABELS = get_class_labels()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # 1. Get base64 image from request
        data = request.get_json()
        image_data = data.get('image')
        if not image_data:
            return jsonify({"error": "No image data"}), 400

        # 2. Decode base64 to numpy array
        header, encoded = image_data.split(",", 1)
        nparr = np.frombuffer(base64.b64decode(encoded), np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        # 3. Detect Faces
        faces = detector.detect_faces(frame)
        
        results = []
        for f in faces:
            box = f['box']
            face_img = f['face_image']
            
            # 4. Preprocess & Predict
            face_final = classifier.preprocess_face(face_img)
            class_id, confidence, probs = classifier.predict(face_final)
            
            # 5. Format results
            results.append({
                "box": [int(x) for x in box],
                "emotion": EMOTION_LABELS.get(class_id, "Unknown"),
                "confidence": float(confidence),
                "probabilities": {EMOTION_LABELS[i]: float(p) for i, p in enumerate(probs)}
            })

        return jsonify({
            "success": True,
            "faces_count": len(results),
            "results": results
        })

    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

if __name__ == '__main__':
    # Run server
    print("--- FaceSentrix Web Server Starting ---")
    app.run(debug=True, port=5000)

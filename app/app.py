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

# Global state for smoothing (In a production app, use session-based or client-ID based tracking)
prediction_history = {} # face_id -> moving_average_probs

@app.route('/predict', methods=['POST'])
def predict():
    global prediction_history
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
        for i, f in enumerate(faces):
            box = f['box']
            face_img = f['face_image']
            
            # 4. Preprocess & Predict
            face_final = classifier.preprocess_face(face_img)
            class_id, confidence, probs = classifier.predict(face_final)
            
            # 5. Apply Smoothing (EMA)
            # Simplified: Use index as ID for this session
            face_id = f"face_{i}"
            if face_id in prediction_history:
                # Alpha=0.3 for smooth transitions
                prediction_history[face_id] = 0.7 * prediction_history[face_id] + 0.3 * probs
            else:
                prediction_history[face_id] = probs
            
            smoothed_probs = prediction_history[face_id]
            smoothed_class_id = np.argmax(smoothed_probs)
            smoothed_conf = smoothed_probs[smoothed_class_id]
            
            # 6. Format results
            results.append({
                "box": [int(x) for x in box],
                "emotion": EMOTION_LABELS.get(smoothed_class_id, "Unknown"),
                "confidence": float(smoothed_conf),
                "probabilities": {EMOTION_LABELS[j]: float(p) for j, p in enumerate(smoothed_probs)}
            })

        # Clear history if no faces detected to reset for next person
        if not faces:
            prediction_history = {}

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

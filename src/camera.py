"""
FaceSentrix - Real-Time Camera Pipeline
"""

import cv2
import time
import argparse
import sys
import os

# Ensure the project root is in path for module resolution
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.face_detector import FaceDetector
from src.emotion_classifier import EmotionClassifier
from src.visualizer import Visualizer
from src.utils import get_class_labels

def main():
    parser = argparse.ArgumentParser(description="FaceSentrix Live Emotion Recognition")
    parser.add_argument("--source", type=int, default=0, help="Camera device index (default: 0)")
    parser.add_argument("--model", type=str, default="models/emotion_model.h5", help="Path to trained model")
    args = parser.parse_args()

    print("--- Initializing FaceSentrix Real-Time Pipeline ---")
    
    # Load Detector
    print("-> Loading Face Detector Cascade...")
    detector = FaceDetector(backend="haar")
    
    # Load CNN Model
    print(f"-> Loading CNN Emotion Classifier from {args.model}...")
    classifier = EmotionClassifier(model_path=args.model)
    labels = get_class_labels()
    
    # Bind Camera Stream
    print(f"-> Binding to Camera Source {args.source}...")
    cap = cv2.VideoCapture(args.source)
    
    if not cap.isOpened():
        print(f"[ERROR] Could not acquire video from camera source {args.source}.")
        return

    print("--- Pipeline Active ---")
    print("Controls:")
    print(" [q] - Quit Application")
    print(" [s] - Capture Frame Screenshot")
    
    frame_count = 0
    start_time = time.time()
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("[CRITICAL] Frame capture failed. Entering safe shutdown.")
            break
            
        # Hardware optimization: Mirror image for UX
        frame = cv2.flip(frame, 1)
        
        # 1. Pipeline: Face Localization
        faces = detector.detect_faces(frame)
        
        # 2. Pipeline: Iterative Emotion Classification
        for f in faces:
            box = f['box']
            face_img = f['face_image']
            
            try:
                # 1. Standardize input (Resize, Normalize, Reshape)
                face_final = classifier.preprocess_face(face_img)
                
                # 2. Inference
                class_id, conf, _ = classifier.predict(face_final)
                emotion_label = labels.get(class_id, "Unknown")
                
                # 3. Render Overlay Graphics
                Visualizer.draw_bounding_box(frame, box, emotion_label, conf)
            except Exception as e:
                # Failsafe if the CNN model isn't built yet or prediction fails
                print(f"[DEBUG] Inference error: {e}")
                Visualizer.draw_bounding_box(frame, box, "Untrained", 0.0)

        # 3. Telemetry: Framerate computation
        frame_count += 1
        elapsed = time.time() - start_time
        fps = frame_count / elapsed
        if frame_count > 60: 
            start_time = time.time()
            frame_count = 0
            
        Visualizer.draw_fps(frame, fps)
        
        # 4. GUI Window Push
        cv2.imshow("FaceSentrix Real-Time Inferencing", frame)
        
        # Control Handling
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print("Received terminal signal. Shutting down...")
            break
        elif key == ord('s'):
            filepath = f"assets/screenshot_{int(time.time())}.png"
            os.makedirs("assets", exist_ok=True)
            cv2.imwrite(filepath, frame)
            print(f"[*] Snapshot successfully exported to {filepath}")
            
    # Purge Memory
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

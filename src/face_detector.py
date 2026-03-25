"""
FaceSentrix - Face Detection Module
"""

import cv2
import os
import numpy as np

class FaceDetector:
    """
    Handles face localization in video frames using OpenCV Haar Cascades.
    """
    
    def __init__(self, backend="haar", scale_factor=1.3, min_neighbors=5):
        """
        Initializes the chosen detector backend.
        """
        self.backend = backend.lower()
        self.scale_factor = scale_factor
        self.min_neighbors = min_neighbors
        self.cascade = None
        
        if self.backend == "haar":
            cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            self.cascade = cv2.CascadeClassifier(cascade_path)
            if self.cascade.empty():
                raise IOError(f"Failed to load Haar cascade from {cascade_path}")
        else:
            raise NotImplementedError(f"Backend '{self.backend}' is not implemented yet. Use 'haar'.")
            
    def detect_faces(self, frame):
        """
        Detects faces in a BGR frame.
        
        Returns:
            list of dicts, each containing:
            - 'box': (x, y, w, h)
            - 'face_image': The cropped face image in grayscale (not resized)
        """
        if frame is None:
            return []
            
        faces_data = []
        # Convert to grayscale if not already
        if len(frame.shape) == 3:
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray_frame = frame
            
        if self.backend == "haar":
            detected_boxes = self.cascade.detectMultiScale(
                gray_frame, 
                scaleFactor=self.scale_factor, 
                minNeighbors=self.min_neighbors, 
                minSize=(48, 48)  # Our model expects 48x48
            )
            
            for (x, y, w, h) in detected_boxes:
                # Add a small padding margin to capture full facial contour
                padding = max(0, int(w * 0.1))
                x_str = max(0, x - padding)
                y_str = max(0, y - padding)
                x_end = min(gray_frame.shape[1], x + w + padding)
                y_end = min(gray_frame.shape[0], y + h + padding)
                
                face_crop = gray_frame[y_str:y_end, x_str:x_end]
                
                # Check for invalid crops at edges
                if face_crop.size == 0:
                    continue
                    
                faces_data.append({
                    'box': (x, y, w, h),
                    'face_image': face_crop
                })
                
        return faces_data

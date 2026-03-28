import os
import sys
import numpy as np
import cv2
import unittest

# Ensure root directory is in sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.face_detector import FaceDetector

class TestFaceDetector(unittest.TestCase):
    
    def setUp(self):
        self.detector = FaceDetector(backend="haar")
        
    def test_initialization(self):
        self.assertEqual(self.detector.backend, "haar")
        self.assertIsNotNone(self.detector.cascade)
        
    def test_detect_faces_empty_frame(self):
        faces = self.detector.detect_faces(None)
        self.assertEqual(faces, [])
        
    def test_detect_faces_black_frame(self):
        # Create a black frame (no faces)
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        faces = self.detector.detect_faces(frame)
        self.assertEqual(faces, [])

if __name__ == '__main__':
    unittest.main()

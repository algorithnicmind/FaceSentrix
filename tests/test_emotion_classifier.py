import os
import sys
import numpy as np
import unittest

# Ensure root directory is in sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.emotion_classifier import EmotionClassifier

class TestEmotionClassifier(unittest.TestCase):
    
    def setUp(self):
        self.classifier = EmotionClassifier()
        
    def test_build_model(self):
        model = self.classifier.build_model()
        self.assertIsNotNone(model)
        self.assertEqual(model.input_shape, (None, 48, 48, 1))
        # Final layer should have 7 outputs
        self.assertEqual(model.output_shape, (None, 7))
        
    def test_predict_without_model(self):
        # Should raise ValueError since model isn't built/loaded
        img = np.zeros((48, 48, 1))
        with self.assertRaises(ValueError):
            self.classifier.predict(img)

    def test_predict_output_shape(self):
        # Build an untrained model just to test the prediction shape
        self.classifier.build_model()
        img = np.zeros((48, 48, 1), dtype=np.float32)
        class_id, confidence, preds = self.classifier.predict(img)
        
        self.assertTrue(0 <= class_id < 7)
        self.assertTrue(0 <= confidence <= 1.0)
        self.assertEqual(len(preds), 7)

if __name__ == '__main__':
    unittest.main()

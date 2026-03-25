"""
FaceSentrix - Visualization Module
"""
import cv2

# BGR color mapping tailored for Emotion Visualization
EMOTION_COLORS = {
    "Angry": (0, 0, 255),       # Red
    "Disgust": (255, 0, 255),   # Magenta
    "Fear": (0, 165, 255),      # Orange
    "Happy": (0, 255, 0),       # Green
    "Sad": (255, 0, 0),         # Blue
    "Surprise": (0, 255, 255),  # Yellow
    "Neutral": (200, 200, 200)  # Gray
}

class Visualizer:
    @staticmethod
    def draw_bounding_box(frame, box, label_str, confidence):
        """
        Draws a dynamic bounding box and label context around detected faces.
        """
        x, y, w, h = box
        color = EMOTION_COLORS.get(label_str, (200, 200, 200))
        
        # Dynamic line thickness based on face scale
        thickness = max(2, int(w / 100))
        
        # Frame outline
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, thickness)
        
        # Formatted string
        text = f"{label_str} [{confidence*100:.0f}%]"
        font = cv2.FONT_HERSHEY_DUPLEX
        font_scale = max(0.5, w / 250.0)
        
        (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)
        
        # Solid background for label
        cv2.rectangle(frame, 
                      (x, y - text_height - baseline - 10), 
                      (x + text_width + 10, y), 
                      color, -1)
                      
        # White text over colored background
        cv2.putText(frame, text, (x + 5, y - 10), font, font_scale, (255, 255, 255), max(1, thickness - 1))
        
    @staticmethod
    def draw_fps(frame, fps):
        """
        Renders the processing FPS in top-left corner.
        """
        cv2.putText(frame, f"FPS: {fps:.1f}", (15, 45), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

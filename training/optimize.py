import tensorflow as tf
import os

def optimize_model(h5_path, tflite_path):
    print(f"Loading custom CNN from {h5_path}...")
    model = tf.keras.models.load_model(h5_path)
    
    print("Initializing TFLite Converter...")
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    
    # Enable aggressive optimization
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    
    # Quantization (optional - float16 for better precision preservation)
    converter.target_spec.supported_types = [tf.float16]
    
    print("Converting Model to TFLite (FP16 Quantized)...")
    tflite_model = converter.convert()
    
    os.makedirs(os.path.dirname(tflite_path), exist_ok=True)
    with open(tflite_path, "wb") as f:
        f.write(tflite_model)
        
    print(f"Successfully saved optimized model to: {tflite_path}")
    
    # Benchmark diff
    orig_size = os.path.getsize(h5_path) / (1024 * 1024)
    new_size = os.path.getsize(tflite_path) / (1024 * 1024)
    print(f"Original size: {orig_size:.2f} MB")
    print(f"Optimized size: {new_size:.2f} MB")
    print(f"Reduction: {100 - (new_size / orig_size) * 100:.2f}%")

if __name__ == "__main__":
    optimize_model("models/emotion_model.h5", "models/emotion_model.tflite")

"""
FaceSentrix - Root Execution Point
"""
import sys
import os

# Redirect logic to src.camera
if __name__ == '__main__':
    print("Initializing Core...")
    os.system(f"{sys.executable} src/camera.py " + " ".join(sys.argv[1:]))

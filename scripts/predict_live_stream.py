import os
import sys
import time
import cv2
import numpy as np
from tensorflow.keras.models import load_model

# --- CONFIGURATION ---
MODEL_PATH = './results/model/final_emotion_model.keras'
IMAGE_DIR = './results/preprocessing_test'
EMOTION_LABELS = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']


def predict_from_images():
    # 1. Load the Keras model
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model file not found at {MODEL_PATH}")
        sys.exit()
    
    model = load_model(MODEL_PATH)
    
    if not os.path.exists(IMAGE_DIR):
        print(f"Error: Target image directory {IMAGE_DIR} does not exist.")
        sys.exit()

    
    image_files = [f for f in os.listdir(IMAGE_DIR) if f.startswith('image') and f.endswith('.png')]
    
  
    image_files.sort(key=lambda x: int(''.join(filter(str.isdigit, x))))

    if not image_files:
        print("No images found to process.")
        return

    for filename in image_files:
        image_path = os.path.join(IMAGE_DIR, filename)
        
        print("Preprocessing ...")
        
        img = cv2.imread(image_path)
        if img is None:
            continue
            
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        resized_img = cv2.resize(gray_img, (48, 48))
        x = resized_img.reshape(-1, 48, 48, 1) / 255.0
        
        preds = model.predict(x, verbose=0)
        class_idx = np.argmax(preds, axis=1)[0]
        confidence = preds[0][class_idx] * 100
        emotion = EMOTION_LABELS[class_idx]
        
        timestamp = time.strftime("%H:%M:%S")
        print(f"{timestamp}s : {emotion} , {int(confidence)}%\n")
        
        time.sleep(0.2)


if __name__ == "__main__":
    predict_from_images()
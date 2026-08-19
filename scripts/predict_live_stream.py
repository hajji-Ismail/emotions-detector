import os
import sys
import time
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from preprocess import video_preprocessing , predict_from_images, get_frame_x

# --- CONFIGURATION ---
MODEL_PATH = './results/model/final_emotion_model.keras'
EMOTION_LABELS = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

model = load_model(MODEL_PATH)





def predict() : 
    cam = cv2.VideoCapture(0, cv2.CAP_V4L2)

    cam.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
    if not cam.isOpened():

        video_preprocessing()
        exes = predict_from_images()
        for i, x in enumerate(exes):
            print("Preprocessing ...")
            x_batch = np.expand_dims(x, axis=0)
        
            preds = model.predict(x_batch, verbose=0)
            class_idx = np.argmax(preds, axis=1)[0]
            confidence = preds[0][class_idx] * 100
            emotion = EMOTION_LABELS[class_idx]
            
            timestamp = time.strftime("%H:%M:%S")
            print(f"{timestamp}s : {emotion} ,{confidence:.1f}%")
    else :
        while True :
            
            print("Preprocessing ...")
            x = get_frame_x(cam)
            if x is None :
                break 
            preds = model.predict(x, verbose=0)
            class_idx = np.argmax(preds, axis=1)[0]
            confidence = preds[0][class_idx] * 100
            emotion = EMOTION_LABELS[class_idx]
                        
            timestamp = time.strftime("%H:%M:%S")
            print(f"{timestamp}s : {emotion} ,{confidence:.1f}%")
            
        
if __name__ == "__main__":
    predict()
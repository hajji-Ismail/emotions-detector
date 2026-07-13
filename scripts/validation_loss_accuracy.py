import os
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import to_categorical

# -----------------------------
# Configuration
# -----------------------------
MODEL_PATH = "results/model/final_emotion_model.keras"
TEST_PATH = "data/test_with_emotions.csv"

IMG_SIZE = 48
NUM_CLASSES = 7

# -----------------------------
# Load model
# -----------------------------
model = load_model(MODEL_PATH)

# -----------------------------
# Load test dataset
# -----------------------------
df = pd.read_csv(TEST_PATH)

pixels_list = df['pixels'].apply(lambda x: np.fromstring(x, sep=' ')).values
    
X = np.vstack(pixels_list).reshape(-1, 48, 48, 1)
X = X / 255.0  

y = df["emotion"]

# -----------------------------
# Evaluate
# -----------------------------
loss, accuracy = model.evaluate(X, y, verbose=1)

print(f"\nValidation Loss: {loss:.4f}")
print(f"Validation Accuracy: {accuracy*100:.2f}%")
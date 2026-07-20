import tensorflow as tf
import pandas as pd
import numpy as np

# 1. Load the saved .keras model
model = tf.keras.models.load_model('./results/model/final_emotion_model.keras')
df = pd.read_csv("./data/test_with_emotions.csv")
pixels_list = df['pixels'].apply(lambda x: np.fromstring(x, sep=' ')).values
    
x = np.vstack(pixels_list).reshape(-1, 48, 48, 1)
x = x / 255.0  

predictions = model.predict(x, verbose=0)


y_true = df['emotion'].values

y_pred = np.argmax(predictions, axis=1)

accuracy = np.mean(y_true == y_pred) * 100

print(f"Accuracy on test set: {accuracy:.0f}%")
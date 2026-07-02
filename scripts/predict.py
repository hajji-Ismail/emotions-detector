import tensorflow as tf
import pandas as pd
import numpy as np

# 1. Load the saved .keras model
model = tf.keras.models.load_model('./results/modelfinal_emotion_model.keras')
df = pd.read_csv("./data/test_with_emotions.csv")
pixels_list = df['pixels'].apply(lambda x: np.fromstring(x, sep=' ')).values
    
x = np.vstack(pixels_list).reshape(-1, 48, 48, 1)
x = x / 255.0  

# 3. Run the prediction (verbose=0 turns off the progress bar for clean output)
predictions = model.predict(x, verbose=0)

# --- ACCURACY CALCULATION ---

# 1. Get the true labels from the CSV (adjust 'emotion' if your column name is different)
y_true = df['emotion'].values

# 2. Get the index of the highest probability for each prediction row
y_pred = np.argmax(predictions, axis=1)

# 3. Calculate accuracy (percentage of matching predictions)
accuracy = np.mean(y_true == y_pred) * 100

# 4. View the results in your exact desired format
print(f"Accuracy on test set: {accuracy:.0f}%")
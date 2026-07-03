import os
import datetime
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard

# Environment setup & compatibility fixes

def Extrat_X_Y(df) :
    pixels_list = df['pixels'].apply(lambda x: np.fromstring(x, sep=' ')).values
    
    x = np.vstack(pixels_list).reshape(-1, 48, 48, 1)
    x= x / 255.0  
    y=df["emotion"]
    return x,y

def build_cnn_model():
    """Defines and compiles the CNN architecture with integrated data augmentation."""
    # Data Augmentation Layer block
    data_augmentation = tf.keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.05),  # Slightly lowered to preserve face structures
        layers.RandomZoom(0.05),
    ])

    model = models.Sequential()
    
    # Input Layer + Augmentation
    model.add(layers.Input(shape=(48, 48, 1)))
    model.add(data_augmentation)
    
    # Block 1: Conv -> Batch Normalization -> MaxPooling -> Dropout
    model.add(layers.Conv2D(32, (3, 3), padding='same', activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.Conv2D(64, (3, 3), padding='same', activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Dropout(0.25))
    
    # Block 2: Deeper features
    model.add(layers.Conv2D(128, (3, 3), padding='same', activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Dropout(0.25))
    
    # Block 3: Extracting complex patterns
    model.add(layers.Conv2D(256, (3, 3), padding='same', activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Dropout(0.4))
    
    # Flattening & Fully Connected Layers
    model.add(layers.Flatten())
    model.add(layers.Dense(512, activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(0.5))
    
    # Output Layer (7 Classes for Emotions)
    model.add(layers.Dense(7, activation='softmax'))
    
    # Compile the model
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def get_callbacks():
    """Configures and returns TensorBoard, Early Stopping, and Model Checkpointing callbacks."""
    log_dir = os.path.join("logs", "fit", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    
    tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)
    
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True,
        verbose=1
    )
    
    model_checkpoint = ModelCheckpoint(
        filepath='./results/model/final_emotion_model.keras',
        monitor='val_accuracy',
        save_best_only=True,
        verbose=1
    )
    
    return [tensorboard_callback, early_stopping, model_checkpoint]

def plot_and_save_learning_curves(history, filename='./results/model/learning_curves.png'):
    """Generates and saves a training history plot showing validation metrics."""
    plt.figure(figsize=(12, 4))

    # Plot Accuracy Curves
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Val Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    # Plot Loss Curves
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.tight_layout()
    plt.savefig(filename)
    print(f"Learning curves successfully saved as {filename}.")

def main():
    df = pd.read_csv("./data/train.csv")
    print(df.head(1))
    x,y = Extrat_X_Y(df)
    model = build_cnn_model()
    EPOCHS = 100
    BATCH_SIZE = 64
    callbacks_list = get_callbacks()
    print("Starting training routine...")
    history = model.fit(
        x, y,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=callbacks_list
    )
    plot_and_save_learning_curves(history)

    


if __name__ == '__main__':
    main()
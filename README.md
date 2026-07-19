# Facial Emotion Recognition with a CNN

This project trains and serves a convolutional neural network (CNN) that classifies a **48 × 48 grayscale face image** into one of seven facial-expression categories:

| ID | Emotion |
| ---: | --- |
| 0 | Angry |
| 1 | Disgust |
| 2 | Fear |
| 3 | Happy |
| 4 | Sad |
| 5 | Surprise |
| 6 | Neutral |

The project uses FER2013-format data: each sample is represented by 2,304 space-separated pixel values and, where labels are available, an integer `emotion` column. It provides a complete workflow for model training, held-out evaluation, image-sequence inference, and live camera capture.

> This model predicts an expression visible in an image; it should not be treated as a reliable measure of a person's internal emotional state.

## Architecture at a glance

```text
FER2013 CSV / captured frame
            │
            ▼
Parse pixels → reshape to 48×48×1 → scale values to [0, 1]
            │
            ├── Training only: flip / slight rotation / slight zoom
            ▼
CNN feature extractor
  32 filters → 64 filters → pool
  128 filters → pool
  256 filters → pool
            │
            ▼
Flatten → Dense(512) → Softmax(7)
            │
            ▼
Emotion probabilities and highest-probability class
```

The system has two operational paths:

1. **Offline path** — `train.py` loads labelled CSV data, trains the CNN, writes the final restored model, learning curves, and a text model summary. `predict.py` and `validation_loss_accuracy.py` evaluate that saved model against labelled test data.
2. **Camera path** — `preprocess.py` records a 20-second V4L2 camera stream and saves one image per second. `predict_live_stream.py` converts those images to the same model input format and prints an emotion and confidence for each one.

## Repository layout

```text
.
├── data/
│   ├── train.csv                    # 28,709 labelled examples
│   ├── test.csv                     # 7,178 unlabelled examples
│   └── test_with_emotions.csv       # 7,178 labelled examples for evaluation
├── scripts/
│   ├── train.py                     # training pipeline and CNN definition
│   ├── predict.py                   # batch test-set accuracy
│   ├── validation_loss_accuracy.py  # test loss and accuracy
│   ├── preprocess.py                # camera recording and snapshots
│   └── predict_live_stream.py       # predictions for saved snapshots
├── results/model/
│   ├── final_emotion_model.keras    # trained Keras model
│   ├── final_emotion_model_arch.txt # exported Keras summary
│   └── learning_curves.png          # training/validation curves
└── requirement.txt                  # Conda environment export
```

## Data and preprocessing

`train.csv` has `emotion` and `pixels` columns. `pixels` is a string containing 2,304 intensity values in row-major order. The training and evaluation scripts perform the same essential conversion:

```python
pixels → NumPy array → (48, 48, 1) → pixels / 255.0
```

This final division is important: it converts the original `[0, 255]` pixel range to `[0.0, 1.0]`, the range used when the CNN was trained. Camera images receive the equivalent preprocessing: OpenCV loads the BGR image, it is converted to grayscale, resized to 48 × 48, reshaped to `(1, 48, 48, 1)`, and normalized.

For training, the 28,709 labelled records are split into **80% training** and **20% validation** sets with `random_state=42` and `stratify=y`. Stratification preserves the emotion distribution across both splits, which is useful because the classes are imbalanced (for example, Disgust has far fewer samples than Happy).

## CNN model in detail

The implementation is a Keras `Sequential` CNN. All convolutional layers use **3 × 3 kernels**, `padding="same"`, and ReLU activation. Same padding retains spatial dimensions during convolution, while pooling progressively reduces the feature-map size.

| Stage | Layers | Output shape | Purpose |
| --- | --- | --- | --- |
| Input | `Input(48, 48, 1)` | 48 × 48 × 1 | One normalized grayscale face image. |
| Augmentation* | horizontal flip, rotation ±5%, zoom ±5% | 48 × 48 × 1 | Produces small, plausible variations during training. |
| Feature block 1 | Conv2D(32) → BatchNorm → Conv2D(64) → BatchNorm → MaxPool(2×2) → Dropout(0.25) | 24 × 24 × 64 | Learns low-level edges, contours, and local facial textures. |
| Feature block 2 | Conv2D(128) → BatchNorm → MaxPool(2×2) → Dropout(0.25) | 12 × 12 × 128 | Combines local features into parts such as eyes, brows, and mouth regions. |
| Feature block 3 | Conv2D(256) → BatchNorm → MaxPool(2×2) → Dropout(0.40) | 6 × 6 × 256 | Encodes higher-level arrangements that distinguish expressions. |
| Classifier | Flatten → Dense(512, ReLU) → BatchNorm → Dropout(0.50) | 512 | Combines the spatial feature maps into a compact decision representation. |
| Output | Dense(7, Softmax) | 7 | Produces one probability per emotion class. |

\* The augmentation layers run only when the model is in training mode. They are inactive during validation and inference, so predictions use the original normalized image.

### What each component contributes

- **Convolutions** slide learned 3 × 3 filters over the image. Early filters respond to simple patterns such as edges; deeper filters use those responses to recognize expression-relevant structures.
- **ReLU** (`max(0, x)`) introduces non-linearity, allowing the network to model complex visual relationships rather than only linear combinations of pixels.
- **Batch normalization** normalizes intermediate activations during training and learns a stable scale and offset. This generally improves optimization stability and allows the network to train more reliably.
- **Max pooling** keeps the strongest activation in each 2 × 2 area. It halves width and height at each block (48 → 24 → 12 → 6), reducing computation and making features less sensitive to small shifts.
- **Dropout** randomly disables a fraction of activations during training. Rates rise from 0.25 to 0.50 as features become more abstract, helping reduce overfitting in the high-capacity classifier head.
- **Flatten and Dense(512)** turn the final 6 × 6 × 256 feature tensor into 9,216 values, then learn combinations that support a class decision.
- **Softmax** turns the final seven scores into probabilities that sum to 1. The predicted emotion is `argmax(probabilities)`; the live script reports that class and its probability as confidence.

The network has **5,114,503 model parameters**: 5,112,519 trainable parameters and 1,984 non-trainable batch-normalization statistics. The 512-unit dense layer contains most of the trainable capacity (4,719,104 parameters), because it connects all 9,216 flattened features to every dense unit.

## Training procedure

`scripts/train.py` trains the model with the following configuration:

| Setting | Value |
| --- | --- |
| Optimizer | Adam |
| Learning rate | 0.001 |
| Loss | Sparse categorical cross-entropy |
| Metric | Accuracy |
| Batch size | 64 |
| Maximum epochs | 100 |
| Early stopping | Monitor `val_loss`, patience 10, restore best weights |
| Checkpoint during training | Save the best `val_accuracy` checkpoint to `results/model/final_emotion_model.keras` |

Sparse categorical cross-entropy is appropriate because labels are stored as integer class IDs, rather than one-hot vectors. During each update, Adam adjusts the convolution and dense weights to increase the predicted probability of the correct emotion.

TensorBoard logs are written under `logs/fit/<timestamp>/`. Early stopping restores the weights from the epoch with the lowest validation loss. At the end of training, the script saves those restored weights to `results/model/final_emotion_model.keras`, overwriting the same path used by the intermediate validation-accuracy checkpoint; it also writes the Keras summary and creates `results/model/learning_curves.png` with training/validation accuracy and loss.

## Setup

The checked-in `requirement.txt` is a Conda explicit package export. Create the environment from it if it matches your platform, then install the Python libraries used by the scripts:

```bash
conda create --name emotions --file requirement.txt
conda activate emotions
python -m pip install tensorflow pandas numpy matplotlib scikit-learn opencv-python
```

On machines without Conda, create a virtual environment and install the same Python libraries:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install tensorflow pandas numpy matplotlib scikit-learn opencv-python
```

## How to run

Run commands from the repository root.

### Train or retrain the model

```bash
python scripts/train.py
```

This expects `data/train.csv` and overwrites the saved model/checkpoint in `results/model/` as training progresses.

### Evaluate the saved model

```bash
python scripts/predict.py
python scripts/validation_loss_accuracy.py
```

Both commands use `data/test_with_emotions.csv`. The first prints rounded test accuracy; the second runs Keras evaluation and prints loss plus accuracy with two decimal places.

### Capture and classify a camera sequence

```bash
python scripts/preprocess.py
python scripts/predict_live_stream.py
```

`preprocess.py` opens camera index `0` through the Linux V4L2 backend, records up to 20 seconds to `results/preprocessing_test/input_video.mp4`, and saves `image0.png`, `image1.png`, and so on at one-second intervals. Press `q` to stop early. The prediction script processes those numbered PNGs in order and prints output similar to:

```text
14:32:08s : Happy , 87%
```

The camera workflow requires a working V4L2-compatible device and desktop access for the OpenCV preview window. If your camera is not at index `0`, update `cv2.VideoCapture(0, cv2.CAP_V4L2)` in `scripts/preprocess.py`.

## Outputs and interpretation

- `final_emotion_model.keras` is the reusable Keras model artifact used by all prediction scripts.
- `learning_curves.png` helps identify underfitting or overfitting: a widening gap between training and validation metrics can indicate that the model is memorizing the training set.
- The reported confidence is the model's softmax probability for its top class. It is useful for ranking predictions, but it is not a calibrated guarantee that the prediction is correct.

## Limitations

Performance can vary significantly with lighting, head pose, occlusion, camera quality, demographic representation, and expressions outside the training distribution. The live pipeline classifies the full resized frame; it does not perform face detection or alignment first. For more robust real-world use, add face detection/alignment, assess class-wise metrics and confusion matrices, and evaluate on representative data before deployment.

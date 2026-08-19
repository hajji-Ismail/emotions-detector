import os
import sys
import cv2
import numpy as np 

MODEL_PATH = './results/model/final_emotion_model.keras'
IMAGE_DIR = './results/preprocessing_test'
output_dir = os.path.join("./results", "preprocessing_test")
os.makedirs(output_dir, exist_ok=True)

fallback_video_path = os.path.join(output_dir, "input_video.mp4")

def predict_from_images():
    if not os.path.exists(IMAGE_DIR):
        print(f"Error: Target image directory {IMAGE_DIR} does not exist.")
        sys.exit()

    image_files = [f for f in os.listdir(IMAGE_DIR) if f.startswith('image') and f.endswith('.png')]
    image_files.sort(key=lambda x: int(''.join(filter(str.isdigit, x))))

    if not image_files:
        print("No images found to process.")
        return np.empty((0, 48, 48, 1), dtype=np.float32)

    x_list = []

    for filename in image_files:
        image_path = os.path.join(IMAGE_DIR, filename)

        img = cv2.imread(image_path)
        if img is None:
            continue

        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Detect faces in the grayscale image
        faces = face_cascade.detectMultiScale(
            gray_img, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
        )

        # Skip frame if no face is detected
        if len(faces) == 0:
            continue

        # Get the largest face
        x_box, y_box, w_box, h_box = max(faces, key=lambda b: b[2] * b[3])
    
        # Crop face from the original image (img, not frame)
        face_crop = img[y_box : y_box + h_box, x_box : x_box + w_box]
        if face_crop.size == 0:
            continue

        # Convert face crop to 48x48 grayscale
        gray_crop = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
        resized_img = cv2.resize(gray_crop, (48, 48))
        
        # Shape: (48, 48, 1)
        x_single = np.expand_dims(resized_img, axis=-1).astype(np.float32) / 255.0
        x_list.append(x_single)

    if not x_list:
        return np.empty((0, 48, 48, 1), dtype=np.float32)

    # Returns array of shape (N, 48, 48, 1)
    return np.array(x_list, dtype=np.float32)
def video_preprocessing():
    if not os.path.exists(fallback_video_path):
        print(f"Error: Fallback video path '{fallback_video_path}' does not exist.")
        sys.exit()

    cam = cv2.VideoCapture(fallback_video_path)
    if not cam.isOpened():
        print("Error: Could not open the pre-recorded video stream.")
        sys.exit()

    fps = cam.get(cv2.CAP_PROP_FPS)
    if fps == 0 or fps is None:
        fps = 30.0  
    snapshot_count = 0
    max_duration_sec = 20
    frame_index = 0

    while True:
        ret, frame = cam.read()

        if not ret or frame is None:
            print("\nEnd of video stream reached or frame read error.")
            break

        video_time_sec = frame_index / fps

        if video_time_sec >= max_duration_sec:
            print(
                f"\nReached {max_duration_sec} seconds threshold. Stopping execution."
            )
            break

        if frame_index % int(fps) == 0:
            snapshot_filename = f"image{snapshot_count}.png"
            snapshot_path = os.path.join(output_dir, snapshot_filename)

            cv2.imwrite(snapshot_path, frame)
            print(f"Saved: {snapshot_path} at video time {int(video_time_sec)}s")

            snapshot_count += 1

        frame_index += 1

        cv2.imshow("Camera Stream", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            print("\nStream stopped by user.")
            break

    cam.release()
    cv2.destroyAllWindows()
    print(f"Finished! Total snapshots saved to {output_dir}: {snapshot_count}")



face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)


def get_frame_x(cam):
    ret, frame = cam.read()
    if not ret or frame is None:
        return None

    cv2.imshow("Camera Stream", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        print("\nStream stopped by user.")
        return None

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
    )

    if len(faces) == 0:

        return None

    # Get the largest face in the frame
    x_box, y_box, w_box, h_box = max(faces, key=lambda b: b[2] * b[3])
    
    face_crop = frame[y_box : y_box + h_box, x_box : x_box + w_box]
    if face_crop.size == 0:

        return None

    cv2.rectangle(frame, (x_box, y_box), (x_box + w_box, y_box + h_box), (0, 255, 0), 2)

    gray_img = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
    resized_img = cv2.resize(gray_img, (48, 48))
    x = np.expand_dims(resized_img, axis=(0, -1)).astype(np.float32) / 255.0

    return x

import cv2
import sys
import time
import os

# --- PATH CONFIGURATION ---
# Define the target output directory
output_dir = os.path.join("./results", "preprocessing_test")

# Create the directory structure dynamically if it doesn't exist yet
os.makedirs(output_dir, exist_ok=True)

video_path = os.path.join(output_dir, "input_video.mp4")
# ---------------------------

# 1. Force OpenCV to use the V4L2 backend for Linux explicitly
cam = cv2.VideoCapture(0, cv2.CAP_V4L2)

# 2. Force MJPEG pixel format so it plays nice with the loopback driver
cam.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))

if not cam.isOpened():
    print("Error: Could not open the DroidCam device stream.")
    sys.exit()

# Get the frame dimensions after setting the backend/format
frame_width = int(cam.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Define the codec and create VideoWriter object pointing to the results folder
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(video_path, fourcc, 20.0, (frame_width, frame_height))

print(f"Starting camera stream. Recording for 20 seconds...")
print(f"Saving output video to: {video_path}")
print("Press 'q' to quit early.")

# Variables to handle the 20-second duration and 1-second interval snapshots
start_time = time.time()
last_snapshot_time = start_time
snapshot_count = 0
duration = 20  # Total recording time in seconds

while True:
    ret, frame = cam.read()

    # CRITICAL: Check if the frame was successfully grabbed
    if not ret or frame is None:
        print("Warning: Blank frame grabbed or camera disconnected. Skipping...")
        continue

    current_time = time.time()
    elapsed_time = current_time - start_time

    # Stop automatically if 20 seconds have passed
    if elapsed_time >= duration:
        print(f"\nReached {duration} seconds. Stopping recording.")
        break

    # Write the frame to the video output file
    out.write(frame)

    # Capture a snapshot every 1 second
    if current_time - last_snapshot_time >= 1.0:
        # Match your image0.png, image1.png structure (0-indexed based on snapshot_count)
        snapshot_filename = f"image{snapshot_count}.png"
        snapshot_path = os.path.join(output_dir, snapshot_filename)
        
        cv2.imwrite(snapshot_path, frame)
        print(f" Saved: {snapshot_path} at {int(elapsed_time)}s")
        
        snapshot_count += 1
        last_snapshot_time = current_time  # Reset the interval anchor

    # Display the captured frame live
    cv2.imshow('Camera', frame)

    # Press 'q' to exit the loop manually before 20 seconds
    if cv2.waitKey(1) == ord('q'):
        print("\nRecording stopped early by user.")
        break

# Release everything when done
cam.release()
out.release()
cv2.destroyAllWindows()
print(f"Finished! Total snapshots saved to {output_dir}: {snapshot_count}")
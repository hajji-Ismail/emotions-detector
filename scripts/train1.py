import cv2

# Replace with the IP address and port shown on your DroidCam app
# The suffix '/video' or '/mjpegfeed' points to the stream endpoint
ip_address = "192.168.1.104"  # <-- Change this to your phone's IP
port = "4747"
stream_url = f"http://{ip_address}:{port}"

# Initialize the video capture with the network stream URL
cap = cv2.VideoCapture(stream_url)

if not cap.isOpened():
    print("Error: Could not connect to the DroidCam stream. Check your IP and network.")
    exit()

print("Streaming started! Press 'q' to quit.")

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    
    if not ret:
        print("Error: Failed to grab a frame.")
        break

    # Display the video feed
    cv2.imshow("DroidCam Feed", frame)

    # Break the loop when 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Clean up and close windows
cap.release()
cv2.destroyAllWindows()
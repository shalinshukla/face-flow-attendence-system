import os
import time
import cv2
from detector import recognize_faces  # Import the recognize_faces function from detector.py

# Create a directory if it doesn't exist
directory = "class-snapshot"
if not os.path.exists(directory):
    os.makedirs(directory)

# Initialize the webcam
webcam = cv2.VideoCapture(0)

# Wait for 10 seconds
time.sleep(10)

# Get the current timestamp
timestamp = int(time.time())

# Capture a frame
ret, frame = webcam.read()

# Save the captured image with the new filename
if ret:
    filename = f"{directory}/class-snapshot-{timestamp}.jpg"
    cv2.imwrite(filename=filename, img=frame)
    webcam.release()

    print(f"Original image saved as {filename}")
    # Call the recognize_faces function from detector.py with the image path
    recognize_faces(filename, "hog");
else:
    print("Failed to capture frame from the webcam.")
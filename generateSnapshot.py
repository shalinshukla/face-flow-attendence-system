import os
import time
import cv2
from detector import recognize_faces

# Create a directory if it doesn't exist
directory = "class-snapshot"
if not os.path.exists(directory):
    os.makedirs(directory)

try:

    # Initialize the webcam
    webcam = cv2.VideoCapture(0)  # Assuming the first webcam is used

    # Wait for 10 seconds
    time.sleep(10)

    # Get the current timestamp
    timestamp = int(time.time())

    # Capture a frame
    ret, frame = webcam.read()

    # Save the captured image with the new filename
    if ret:
        filename = f"{directory}/class-snapshot-{timestamp}.jpg"
        print(filename);

        cv2.imwrite(filename=filename, img=frame)
        webcam.release()

        print(f"Original image saved as {filename}")
        # Call the recognize_faces function from detector.py with the image path
        names = recognize_faces(filename, "hog")
    else:
        print("Failed to capture frame from the webcam.")

except Exception as e:
    print("An error occurred:", e)
    webcam.release()  # Release the webcam resource in case of error

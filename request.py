import csv
from flask import Flask, jsonify, send_file
import os
import time
import cv2
from detector import recognize_faces
from datetime import datetime
from io import StringIO
import subprocess

app = Flask(__name__)

@app.route('/date', methods=['GET'])
def get_date():
    # Create a directory if it doesn't exist
    # directory = "class-snapshot"
    # if not os.path.exists(directory):
    #     os.makedirs(directory)

    try:
        # Initialize the webcam
        # webcam = cv2.VideoCapture(0)  # Assuming the first webcam is used

        # Wait for 5 seconds
        # time.sleep(5)

        # Get the current timestamp
        # timestamp = int(time.time())

        # Capture a frame
        # ret, frame = webcam.read()

        # Save the captured image with the new filename
        filename = "IMG_5477-converted.jpg"
        if filename:


            # cv2.imwrite(filename=filename, img=frame)
            # webcam.release()

            # Call the recognize_faces function from detector.py with the image path
            names = recognize_faces(filename, "hog")

            # Prepare CSV data with timestamp and names
            # csv_data = StringIO()
            # csv_writer = csv.writer(csv_data)
            # csv_writer.writerow(["Timestamp", "Name"])
            # Create a CSV string from the user data
            csv_data = "Timestamp,Name\n"
            for name in names:
                if name is not None:
                    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    csv_data += f"{timestamp},{name}\n"

            # Create a temporary CSV file and serve it for download
            with open("report.csv", "w") as csv_file:
                csv_file.write(csv_data)
            return send_file("report.csv", as_attachment=True, download_name="users.csv")
        else:
            print("Failed to capture frame from the webcam.")
            return jsonify({"error": "Failed to capture frame from the webcam."}), 500

    except Exception as e:
        print("An error occurred:", e)
        webcam.release()  # Release the webcam resource in case of error
        return jsonify({"error": "An error occurred."}), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=3000, debug=True)
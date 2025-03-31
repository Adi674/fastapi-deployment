import cv2
import requests
import numpy as np
import base64

# 🔗 API Endpoint
API_URL = "http://127.0.0.1:8000/start_detection"

# Open webcam on your local machine
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not access the webcam.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Couldn't read frame.")
        break

    # Encode frame as JPEG
    _, img_encoded = cv2.imencode(".jpg", frame)
    files = {"file": ("frame.jpg", img_encoded.tobytes(), "image/jpeg")}

    # Send frame to API
    response = requests.post(API_URL, files=files)

    if response.status_code == 200:
        data = response.json()

        # Decode Base64 image received from API
        img_bytes = base64.b64decode(data["image"])
        img_np = np.frombuffer(img_bytes, np.uint8)
        processed_frame = cv2.imdecode(img_np, cv2.IMREAD_COLOR)

        # Show processed video
        cv2.imshow("Processed Webcam Feed", processed_frame)

        # Print anomaly detection results
        print("Anomalies:", data["anomalies"])

    else:
        print(f"Error {response.status_code}: {response.text}")

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

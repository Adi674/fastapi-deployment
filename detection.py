from ultralytics import YOLO
import cv2
import numpy as np
import os
from anomaly import detect_anomaly

# Load YOLO model
model = YOLO("yolov8n.pt")  # Use yolov8s or larger for better accuracy

# Ensure output folder exists
output_folder = "output"
os.makedirs(output_folder, exist_ok=True)

def run_webcam_detection(frame):  # Remove webcam initialization here
    anomaly_detected = False
    anomalies = []

    # Run YOLO object detection
    results = model(frame)
    detected_objects = []

    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])  # Bounding box coordinates
            class_id = int(box.cls[0])  # Object class
            confidence = float(box.conf[0])  # Confidence score

            # Draw bounding box
            label = f"Class: {class_id}, Conf: {confidence:.2f}"
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # Store detected object data
            detected_objects.append({"x1": x1, "y1": y1, "x2": x2, "y2": y2, "class_id": class_id, "confidence": confidence})

    # Pass detected objects to anomaly detection module
    anomalies = detect_anomaly(detected_objects)

    # If anomaly is detected, save the frame
    if anomalies and not anomaly_detected:
        anomaly_image_path = os.path.join(output_folder, "anomaly_detected.jpg")
        cv2.imwrite(anomaly_image_path, frame)
        anomaly_detected = True  # Avoid multiple saves

    return anomalies, frame  # Return the processed frame

from ultralytics import YOLO
import cv2
import numpy as np
import os
from anomaly import detect_anomaly
import torch
import time

# Singleton model loader to avoid reloading on each call
class ModelLoader:
    _instance = None
    
    @classmethod
    def get_model(cls):
        if cls._instance is None:
            print("Loading YOLOV8 model...")
            # Use a larger model for better accuracy
            cls._instance = YOLO("yolov8m.pt")  # Upgrade from yolov8n to yolov8m
            
            # Optional: Set inference parameters for better accuracy
            if torch.cuda.is_available():
                print("CUDA is available! Using GPU acceleration.")
            else:
                print("CUDA not available. Using CPU for inference.")
        return cls._instance

# Ensure output folder exists
output_folder = "output/anomalies"
os.makedirs(output_folder, exist_ok=True)

# Track previous detections for temporal consistency
previous_detections = []
anomaly_cooldown = 0

def run_webcam_detection(frame, min_confidence=0.55):
    global previous_detections, anomaly_cooldown
    
    # Resize for consistent processing (optional)
    # frame = cv2.resize(frame, (640, 480))
    
    # Image enhancement for better detection
    # Apply slight contrast enhancement
    alpha = 1.1  # Contrast control
    beta = 10    # Brightness control
    enhanced_frame = cv2.convertScaleAbs(frame, alpha=alpha, beta=beta)
    
    # Get the YOLO model
    model = ModelLoader.get_model()
    
    # Run YOLO object detection with improved parameters
    results = model(enhanced_frame, conf=min_confidence, iou=0.45)
    
    detected_objects = []
    
    for result in results:
        # Extract classes and names for better context
        names = result.names
        
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])  # Bounding box coordinates
            class_id = int(box.cls[0])  # Object class
            confidence = float(box.conf[0])  # Confidence score
            
            # Get class name instead of just ID for better context
            class_name = names[class_id]
            
            # Draw bounding box with improved visuals
            color = (0, 255, 0)  # Default color is green
            
            # Store detected object data with enhanced information
            detected_objects.append({
                "x1": x1, 
                "y1": y1, 
                "x2": x2, 
                "y2": y2, 
                "class_id": class_id,
                "class_name": class_name,
                "confidence": confidence,
                "area": (x2-x1) * (y2-y1),
                "frame_time": time.time()
            })
            
            # Draw bounding box
            label = f"{class_name}: {confidence:.2f}"
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    # Use temporal consistency by combining with previous detections
    temporal_objects = detected_objects.copy()
    if previous_detections:
        # Add previous detections that might have been missed in this frame
        # Implement object tracking logic here if needed
        pass
    
    # Update previous detections
    previous_detections = detected_objects
    
    # Pass detected objects to anomaly detection module with full context
    anomalies = detect_anomaly(temporal_objects, frame)
    
    # Handle anomaly detection results - FIXED COOLDOWN LOGIC
    if anomalies:
        # Always display anomalies on the frame, regardless of cooldown
        for anomaly in anomalies:
            anomaly_text = anomaly["anomaly"]
            cv2.putText(frame, f"ANOMALY: {anomaly_text}", (10, 50), 
                      cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # Only save images when not in cooldown
        if anomaly_cooldown <= 0:
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            for i, anomaly in enumerate(anomalies):
                # Save frame with annotation
                anomaly_image_path = os.path.join(output_folder, f"anomaly_{timestamp}_{i}.jpg")
                
                # Save the frame
                cv2.imwrite(anomaly_image_path, frame)
                
            # Set cooldown to avoid saving too many frames of the same anomaly
            anomaly_cooldown = 15  # Adjust as needed
    else:
        anomaly_cooldown = max(0, anomaly_cooldown - 1)
    
    return anomalies, frame
import cv2
import numpy as np
from collections import deque
import time

# Store frame history for movement pattern detection
frame_history = deque(maxlen=30)  # Store last 30 frames worth of detections
anomaly_history = deque(maxlen=100)  # Store recent anomalies to prevent repeats

# Define zones (example: areas where certain objects shouldn't be)
restricted_zones = [
    # Format: [x1, y1, x2, y2, restricted_classes]
    # Example: bottom half of the frame is restricted for class "person" after hours
    # [0, 0.5, 1.0, 1.0, [0]]  # x1, y1, x2, y2 as ratios of image dimensions
]

# Define time-based restrictions (e.g., no people during certain hours)
restricted_hours = {
    # Format: class_id: {start_hour, end_hour}
    # Example: persons (class 0) not allowed between 10PM and 6AM
    # 0: {'start': 22, 'end': 6}
}

def detect_anomaly(detected_objects, frame=None):
    anomalies = []
    
    if not detected_objects:
        return anomalies
    
    # Get frame dimensions if available
    frame_height, frame_width = 0, 0
    if frame is not None:
        frame_height, frame_width = frame.shape[:2]
    
    # Store detections in history for pattern recognition
    frame_history.append({
        'time': time.time(),
        'objects': detected_objects
    })
    
    # Current time
    current_hour = time.localtime().tm_hour
    
    # Check individual object anomalies
    for obj in detected_objects:
        class_id = obj["class_id"]
        class_name = obj.get("class_name", f"Class {class_id}")
        confidence = obj["confidence"]
        
        # 1. Unauthorized person detection (with higher confidence threshold)
        if class_id == 0 and confidence > 0.75:  # Person
            # Check if person is in a restricted time
            if class_id in restricted_hours:
                time_rule = restricted_hours[class_id]
                start_hour, end_hour = time_rule['start'], time_rule['end']
                if start_hour > end_hour:  # Handles overnight rules (e.g., 22:00 - 06:00)
                    if current_hour >= start_hour or current_hour < end_hour:
                        anomalies.append({
                            "anomaly": f"Unauthorized {class_name} detected during restricted hours",
                            "details": obj,
                            "confidence": confidence,
                            "severity": "high"
                        })
            
            # Example: Person in unusual pose (would require pose estimation)
            # This is a placeholder - for real implementation, integrate with a pose estimation model
            
        # 2. Prohibited object detection with enhanced context
        if class_id in [67, 56, 43]:  # Cell phone (67), chair (56), knife (43)
            # Enhanced: detect based on context (e.g., phone where it shouldn't be)
            anomalies.append({
                "anomaly": f"Prohibited object ({class_name}) detected",
                "details": obj,
                "confidence": confidence,
                "severity": "medium"
            })
        
        # 3. Zone-based restrictions if frame dimensions are available
        if frame_height > 0 and len(restricted_zones) > 0:
            obj_x1, obj_y1 = obj["x1"], obj["y1"]
            obj_x2, obj_y2 = obj["x2"], obj["y2"]
            
            # Convert to ratio coordinates for zone comparison
            obj_center_x = (obj_x1 + obj_x2) / 2 / frame_width
            obj_center_y = (obj_y1 + obj_y2) / 2 / frame_height
            
            for zone in restricted_zones:
                zone_x1, zone_y1, zone_x2, zone_y2, restricted_classes = zone
                
                # Check if object is in restricted zone
                if (zone_x1 <= obj_center_x <= zone_x2 and 
                    zone_y1 <= obj_center_y <= zone_y2 and 
                    class_id in restricted_classes):
                    anomalies.append({
                        "anomaly": f"{class_name} detected in restricted zone",
                        "details": obj,
                        "confidence": confidence,
                        "severity": "high"
                    })
    
    # 4. Detect unusual patterns (requires temporal analysis)
    if len(frame_history) >= 10:
        # For example, rapid increase in number of objects
        current_count = len(detected_objects)
        past_counts = [len(frame['objects']) for frame in list(frame_history)[:-1]]
        
        if past_counts and current_count > 3 * sum(past_counts) / len(past_counts):
            anomalies.append({
                "anomaly": "Sudden increase in object count",
                "details": {"count": current_count, "average": sum(past_counts) / len(past_counts)},
                "confidence": 0.8,
                "severity": "medium"
            })
    
    # 5. Filter out repetitive anomalies
    filtered_anomalies = []
    for anomaly in anomalies:
        # Create a hash of the anomaly to check if it's a repeat
        anomaly_key = f"{anomaly['anomaly']}_{anomaly.get('details', {}).get('class_id', 0)}"
        
        # Check if we've seen this exact anomaly recently
        if anomaly_key not in [a['key'] for a in anomaly_history]:
            anomaly['key'] = anomaly_key
            anomaly_history.append({'key': anomaly_key, 'time': time.time()})
            filtered_anomalies.append(anomaly)
    
    return filtered_anomalies
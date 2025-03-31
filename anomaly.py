def detect_anomaly(detected_objects):
    anomalies = []
    
    for obj in detected_objects:
        # Example anomaly: Detecting a person with high confidence (Unauthorized entry)
        if obj["class_id"] == 0 and obj["confidence"] > 0.8:  # Class 0 = Person
            anomalies.append({"anomaly": "Unauthorized person detected", "details": obj})

        # Example anomaly: Detecting an object that shouldn’t be there
        if obj["class_id"] == 67:  # Class 67 = Cell phone (example)
            anomalies.append({"anomaly": "Prohibited object detected", "details": obj})
    
    return anomalies

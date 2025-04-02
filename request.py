import cv2
import requests
import numpy as np
import base64
import time

# 🔗 API Endpoint
API_URL = "http://127.0.0.1:8000/start_detection"

# Open webcam on your local machine
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not access the webcam.")
    exit()

# Initialize frame counter
frame_count = 0

# For FPS calculation
prev_time = time.time()
fps = 0

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Couldn't read frame.")
        break
    
    # Calculate FPS
    current_time = time.time()
    if current_time - prev_time > 1:
        fps = frame_count
        frame_count = 0
        prev_time = current_time
    
    # Always display the original frame (will be replaced by processed frame when received)
    display_frame = frame.copy()
    
    # Add FPS counter to display
    cv2.putText(display_frame, f"FPS: {fps}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    # Process only every 10th frame
    if frame_count % 10 == 0:
        try:
            # Encode frame as JPEG
            _, img_encoded = cv2.imencode(".jpg", frame)
            files = {"file": ("frame.jpg", img_encoded.tobytes(), "image/jpeg")}
            
            # Send frame to API with a timeout
            response = requests.post(API_URL, files=files, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                
                # Decode Base64 image received from API
                img_bytes = base64.b64decode(data["image"])
                img_np = np.frombuffer(img_bytes, np.uint8)
                processed_frame = cv2.imdecode(img_np, cv2.IMREAD_COLOR)
                
                # Update display frame
                display_frame = processed_frame
                
                # Print anomaly detection results if any were found
                if data["anomalies"]:
                    print(f"Anomalies detected: {data['anomalies']}")
            else:
                print(f"Error {response.status_code}: {response.text}")
                
        except requests.exceptions.RequestException as e:
            print(f"Request error: {e}")
        except Exception as e:
            print(f"Error processing frame: {e}")
    
    # Increment frame counter
    frame_count += 1
    
    # Show processed video
    cv2.imshow("AI Surveillance Feed", display_frame)
    
    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Clean up
cap.release()
cv2.destroyAllWindows()
print("Client closed successfully")
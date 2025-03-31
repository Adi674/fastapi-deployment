from fastapi import FastAPI, UploadFile, File
import cv2
import numpy as np
import base64
from detection import run_webcam_detection

app = FastAPI()

@app.get("/")
def home():
    return {"message": "Webcam Object Detection API is running!"}

@app.post("/start_detection")
async def start_detection(file: UploadFile = File(...)):
    # Read uploaded image
    image_bytes = await file.read()
    nparr = np.frombuffer(image_bytes, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # Perform object detection & anomaly detection
    anomalies, processed_frame = run_webcam_detection(frame)

    # Encode the processed image as Base64
    _, img_encoded = cv2.imencode(".jpg", processed_frame)
    img_base64 = base64.b64encode(img_encoded.tobytes()).decode("utf-8")

    return {"anomalies": anomalies, "image": img_base64}

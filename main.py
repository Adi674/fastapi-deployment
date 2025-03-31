from fastapi import FastAPI
from detection import run_webcam_detection
from fastapi import FastAPI, UploadFile, File
import numpy as np
import cv2

app = FastAPI()

@app.get("/")
def read_root():
    return {"message": "Webcam Object Detection API is running!"}

@app.post("/start_detection") 
async def start_detection(file: UploadFile = File(...)):
    # Read image from request
    image_bytes = await file.read()
    nparr = np.frombuffer(image_bytes, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # Run detection
    anomalies, anomaly_image = run_webcam_detection(frame)

    return {"anomalies": anomalies, "anomaly_image": anomaly_image.tolist()}
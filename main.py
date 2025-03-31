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
async def start_detection():
    anomalies, anomaly_image = run_webcam_detection()
    return {"anomalies": anomalies, "anomaly_image": anomaly_image}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.1", port=8000)
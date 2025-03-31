from fastapi import FastAPI
from detection import run_webcam_detection

app = FastAPI()

@app.get("/")
def read_root():
    return {"message": "Webcam Object Detection API is running!"}

@app.get("/start_detection")
async def start_detection():
    anomalies, anomaly_image = run_webcam_detection()
    return {"anomalies": anomalies, "anomaly_image": anomaly_image}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.1", port=8000)

import io
import cv2
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from PIL import Image
import asyncio
from ultralytics import YOLO  # YOLOv8

app = FastAPI()

# Load YOLO model (small version for low latency)
yolo_model = YOLO("yolov8n.pt")  # nano model, super fast

# Load hand landmark model (TFLite)
import tensorflow as tf
interpreter = tf.lite.Interpreter(model_path="hand_landmark.tflite")
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

@app.get("/")
async def get():
    with open("index.html") as f:
        return HTMLResponse(f.read())

def predict_hand_landmarks(frame):
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (input_details[0]['shape'][2], input_details[0]['shape'][1]))
    img = img.astype(np.float32) / 255.0
    img = np.expand_dims(img, axis=0)

    interpreter.set_tensor(input_details[0]['index'], img)
    interpreter.invoke()

    landmarks = interpreter.get_tensor(output_details[0]['index'])
    return landmarks[0].tolist()

def detect_objects(frame):
    results = yolo_model(frame, verbose=False)[0]
    boxes = results.boxes.xyxy.cpu().numpy().tolist()  # [[x1,y1,x2,y2],...]
    classes = results.boxes.cls.cpu().numpy().tolist()  # class ids
    confidences = results.boxes.conf.cpu().numpy().tolist()
    return {"boxes": boxes, "classes": classes, "confidences": confidences}

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    print("Client connected")
    try:
        while True:
            data = await websocket.receive_bytes()
            img = Image.open(io.BytesIO(data)).convert('RGB')
            frame = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

            # Hand landmarks
            landmarks = predict_hand_landmarks(frame)

            # Object detection
            objects = detect_objects(frame)

            await websocket.send_json({
                "landmarks": landmarks,
                "objects": objects
            })
    except WebSocketDisconnect:
        print("Client disconnected")
    except Exception as e:
        print("Error:", e)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

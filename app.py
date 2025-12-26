from fastapi import FastAPI, WebSocket
import cv2
import numpy as np
import tensorflow as tf
from ultralytics import YOLO

app = FastAPI()

# ----------------------
# TensorFlow Lite Hand Tracking
# ----------------------
hand_model = tf.lite.Interpreter(model_path="hand_landmark.tflite")
hand_model.allocate_tensors()

input_details = hand_model.get_input_details()
output_details = hand_model.get_output_details()

def detect_hand(frame):
    img = cv2.resize(frame, (input_details[0]['shape'][2], input_details[0]['shape'][1]))
    img = np.expand_dims(img, axis=0).astype(np.float32)
    
    hand_model.set_tensor(input_details[0]['index'], img)
    hand_model.invoke()
    
    output_data = hand_model.get_tensor(output_details[0]['index'])
    return output_data  # hand landmarks

# ----------------------
# YOLOv8 Object Detection
# ----------------------
yolo_model = YOLO("yolov8n.pt")  # or yolov8n.onnx for ONNX

def detect_objects(frame):
    results = yolo_model.predict(frame, verbose=False)
    detected_objects = []
    for r in results:
        for box in r.boxes.xyxy:
            detected_objects.append(box.tolist())
    return detected_objects

# ----------------------
# WebSocket for streaming
# ----------------------
@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    cap = cv2.VideoCapture(0)  # your camera
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        hand_landmarks = detect_hand(frame)
        objects = detect_objects(frame)

        # Here you can send hand + object data to frontend
        await ws.send_json({
            "hand": hand_landmarks.tolist(),
            "objects": objects
        })


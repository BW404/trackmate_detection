import cv2
import numpy as np
import tensorflow as tf
from fastapi import FastAPI, WebSocket
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
import asyncio
import base64

# FastAPI setup
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# HTML page
html = """
<!DOCTYPE html>
<html>
<head>
    <title>TrackMate Detection</title>
</head>
<body>
    <h1>TrackMate: Hand & Object Detection</h1>
    <img id="frame" width="640" />
    <script>
        let ws = new WebSocket("ws://localhost:8000/ws");
        ws.onmessage = function(event) {
            let img = document.getElementById("frame");
            img.src = "data:image/jpeg;base64," + event.data;
        };
    </script>
</body>
</html>
"""

@app.get("/")
async def index():
    return HTMLResponse(html)

# Load TFLite hand landmark model
hand_model = tf.lite.Interpreter(model_path="hand_landmark.tflite")
hand_model.allocate_tensors()
input_details = hand_model.get_input_details()
output_details = hand_model.get_output_details()

# Load YOLOv8 model via OpenCV DNN
# Replace 'yolov8n.onnx' with your YOLO ONNX model
yolo_net = cv2.dnn.readNet("yolov8n.onnx")
yolo_classes = ["person", "phone", "mouse", "keyboard", "monitor", "laptop", "cup", "waterbottle"]

# Helper function to run hand detection
def detect_hand(frame):
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img, (input_details[0]['shape'][2], input_details[0]['shape'][1]))
    input_data = np.expand_dims(img_resized, axis=0).astype(np.float32)
    hand_model.set_tensor(input_details[0]['index'], input_data)
    hand_model.invoke()
    keypoints = hand_model.get_tensor(output_details[0]['index'])
    return keypoints[0]

# Helper function for YOLO detection
def detect_objects(frame):
    blob = cv2.dnn.blobFromImage(frame, 1/255.0, (640, 640), swapRB=True, crop=False)
    yolo_net.setInput(blob)
    outputs = yolo_net.forward()
    h, w = frame.shape[:2]
    boxes, confidences, class_ids = [], [], []

    for detection in outputs[0]:  # assuming output shape [N, 85]
        scores = detection[5:]
        class_id = np.argmax(scores)
        confidence = scores[class_id]
        if confidence > 0.4:
            cx, cy, bw, bh = detection[0:4] * [w, h, w, h]
            x = int(cx - bw/2)
            y = int(cy - bh/2)
            boxes.append([x, y, int(bw), int(bh)])
            confidences.append(float(confidence))
            class_ids.append(class_id)
    indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.4, 0.5)
    result = []
    for i in indices:
        i = i[0]
        result.append((boxes[i], class_ids[i], confidences[i]))
    return result

# WebSocket streaming
@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)  # lower resolution for faster streaming
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                continue

            # Object detection
            objects = detect_objects(frame)
            for box, class_id, conf in objects:
                x, y, w_box, h_box = box
                cv2.rectangle(frame, (x, y), (x + w_box, y + h_box), (0, 255, 0), 2)
                cv2.putText(frame, f"{yolo_classes[class_id]}:{conf:.2f}", (x, y-5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # Hand landmarks
            keypoints = detect_hand(frame)
            for kp in keypoints:
                x = int(kp[0] * frame.shape[1])
                y = int(kp[1] * frame.shape[0])
                cv2.circle(frame, (x, y), 3, (0, 0, 255), -1)  # smaller dots for smoother appearance

            # Encode frame as JPEG
            _, buffer = cv2.imencode(".jpg", frame)
            jpg_as_text = base64.b64encode(buffer).decode("utf-8")
            await ws.send_text(jpg_as_text)
            await asyncio.sleep(0.01)  # small sleep to reduce CPU load
    except Exception as e:
        print("WebSocket error:", e)
    finally:
        cap.release()
        await ws.close()

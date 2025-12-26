import cv2
import numpy as np
import tensorflow as tf
from fastapi import FastAPI, WebSocket
from fastapi.responses import HTMLResponse
from ultralytics import YOLO
import base64

app = FastAPI()

# -------------------------------
# Load models
# -------------------------------
# YOLO ONNX
yolo_net = cv2.dnn.readNet("yolov8n.onnx")

# TFLite hand model
hand_interpreter = tf.lite.Interpreter(model_path="hand_landmark.tflite")
hand_interpreter.allocate_tensors()
input_details = hand_interpreter.get_input_details()
output_details = hand_interpreter.get_output_details()

# -------------------------------
# Webpage HTML
# -------------------------------
html = """
<!DOCTYPE html>
<html>
    <head>
        <title>TrackMate Detection</title>
    </head>
    <body>
        <h1>TrackMate Detection</h1>
        <video id="video" width="640" height="480" autoplay></video>
        <canvas id="canvas" width="640" height="480"></canvas>
        <script>
            const video = document.getElementById('video');
            const canvas = document.getElementById('canvas');
            const ws = new WebSocket("ws://localhost:8000/ws");

            // Access webcam
            navigator.mediaDevices.getUserMedia({ video: true }).then(stream => {
                video.srcObject = stream;
            });

            function sendFrame() {
                const ctx = canvas.getContext('2d');
                ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
                const data = canvas.toDataURL('image/jpeg');
                ws.send(data);
                requestAnimationFrame(sendFrame);
            }

            ws.onmessage = (event) => {
                const img = new Image();
                img.src = event.data;
                const ctx = canvas.getContext('2d');
                ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
            };

            sendFrame();
        </script>
    </body>
</html>
"""

@app.get("/")
async def get():
    return HTMLResponse(html)

# -------------------------------
# WebSocket endpoint
# -------------------------------
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    while True:
        data = await websocket.receive_text()
        # Decode base64
        header, encoded = data.split(",", 1)
        nparr = np.frombuffer(base64.b64decode(encoded), np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        h, w, _ = frame.shape

        # -------------------------------
        # YOLO detection
        # -------------------------------
        blob = cv2.dnn.blobFromImage(frame, 1/255.0, (640, 640), swapRB=True, crop=False)
        yolo_net.setInput(blob)
        outputs = yolo_net.forward()
        # Example: draw dummy box (you can improve YOLO postprocessing)
        cv2.rectangle(frame, (50, 50), (200, 200), (0, 255, 0), 2)

        # -------------------------------
        # Hand TFLite
        # -------------------------------
        hand_input = cv2.resize(frame, (128, 128))
        hand_input = hand_input.astype(np.float32) / 255.0
        hand_input = np.expand_dims(hand_input, axis=0)
        hand_interpreter.set_tensor(input_details[0]['index'], hand_input)
        hand_interpreter.invoke()
        hand_output = hand_interpreter.get_tensor(output_details[0]['index'])
        for point in hand_output[0]:
            cx, cy = int(point[0] * w), int(point[1] * h)
            cv2.circle(frame, (cx, cy), 3, (0, 0, 255), -1)

        # Encode frame back to base64
        _, buffer = cv2.imencode('.jpg', frame)
        b64_frame = "data:image/jpeg;base64," + base64.b64encode(buffer).decode()
        await websocket.send_text(b64_frame)

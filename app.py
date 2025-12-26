import asyncio
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
import socketio
import cv2
import numpy as np
from PIL import Image
from io import BytesIO
import base64

# --- YOLOv8 ---
from ultralytics import YOLO

# Initialize FastAPI + Socket.IO
sio = socketio.AsyncServer(async_mode='asgi', cors_allowed_origins="*")
app = FastAPI()
app = socketio.ASGIApp(sio, app)

# Load YOLOv8 model
yolo_model = YOLO("yolov8n.pt")  # small & fast
classes_of_interest = ["cell phone", "mouse", "keyboard", "laptop", "cup", "bottle", "monitor"]

# --- TFLite hand tracking ---
import tflite_runtime.interpreter as tflite
hand_model = tflite.Interpreter(model_path="hand_landmark.tflite")
hand_model.allocate_tensors()
input_details = hand_model.get_input_details()
output_details = hand_model.get_output_details()


@app.get("/")
async def index():
    with open("index.html") as f:
        return HTMLResponse(f.read())


@sio.event
async def connect(sid, environ):
    print("Client connected:", sid)


@sio.event
async def disconnect(sid):
    print("Client disconnected:", sid)


@sio.event
async def frame(sid, data):
    """
    Receives base64 image from client, runs hand tracking + YOLO, 
    and sends back processed frame.
    """
    try:
        # Decode base64 image
        img_bytes = base64.b64decode(data.split(",")[1])
        img = np.array(Image.open(BytesIO(img_bytes)))
        frame = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        h, w, _ = frame.shape

        # --- YOLO Object Detection ---
        yolo_results = yolo_model(frame)[0]  # first image
        for det in yolo_results.boxes.data.tolist():  # xyxy, conf, cls
            x1, y1, x2, y2, conf, cls_id = det
            class_name = yolo_results.names[int(cls_id)]
            if class_name in classes_of_interest:
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                cv2.putText(frame, f"{class_name} {conf:.2f}", (int(x1), int(y1)-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # --- Hand Landmarks (TFLite) ---
        # Preprocess hand image (resize etc.)
        img_input = cv2.resize(frame, (input_details[0]['shape'][2], input_details[0]['shape'][1]))
        img_input = np.expand_dims(img_input.astype(np.float32), axis=0)
        hand_model.set_tensor(input_details[0]['index'], img_input)
        hand_model.invoke()
        landmarks = hand_model.get_tensor(output_details[0]['index'])

        # Draw landmarks (dots on fingers)
        for lm in landmarks[0]:
            x, y, z = int(lm[0]*w), int(lm[1]*h), lm[2]
            cv2.circle(frame, (x, y), 4, (0, 0, 255), -1)  # red small dots

        # Encode back to base64
        _, buffer = cv2.imencode('.jpg', frame)
        frame_b64 = "data:image/jpeg;base64," + base64.b64encode(buffer).decode()
        await sio.emit("frame", frame_b64, to=sid)

    except Exception as e:
        print("Error processing frame:", e)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

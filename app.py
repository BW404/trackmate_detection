# app.py
import cv2
import base64
import numpy as np
import mediapipe as mp
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
import socketio
import asyncio

# FastAPI app
app = FastAPI()
sio = socketio.AsyncServer(async_mode='asgi', cors_allowed_origins='*')
app_sio = socketio.ASGIApp(sio, app)

# MediaPipe hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# HTML endpoint
@app.get("/", response_class=HTMLResponse)
async def index():
    with open("index.html") as f:
        return f.read()

# Socket.IO event: receive video frame from client
@sio.event
async def video_frame(sid, data):
    try:
        # Decode base64 image
        img_bytes = base64.b64decode(data.split(",")[1])
        np_arr = np.frombuffer(img_bytes, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        # Resize for speed
        frame = cv2.resize(frame, (320, 240))

        # Convert to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # MediaPipe hands detection
        results = hands.process(rgb_frame)

        hand_landmarks = []
        if results.multi_hand_landmarks:
            for hand_landmark in results.multi_hand_landmarks:
                landmarks = [{"x": lm.x, "y": lm.y} for lm in hand_landmark.landmark]
                hand_landmarks.append(landmarks)

        # Emit landmarks back to client
        await sio.emit("hand_landmarks", {"landmarks": hand_landmarks}, to=sid)

    except Exception as e:
        print("Error processing frame:", e)

@sio.event
async def connect(sid, environ):
    print("Client connected:", sid)

@sio.event
async def disconnect(sid):
    print("Client disconnected:", sid)

if __name__ == "__main__":
    import uvicorn
    print("Server running on http://0.0.0.0:8000")
    uvicorn.run(app_sio, host="0.0.0.0", port=8000)

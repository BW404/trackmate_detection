# app.py
import base64
import cv2
import numpy as np
import mediapipe as mp
import asyncio
import socketio
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

# -------------------
# Setup Socket.IO & FastAPI
# -------------------
sio = socketio.AsyncServer(async_mode="asgi", cors_allowed_origins="*")
app = FastAPI()
socket_app = socketio.ASGIApp(sio, app)

# Serve static files (JS, CSS)
app.mount("/static", StaticFiles(directory="static"), name="static")

# -------------------
# Mediapipe Hands
# -------------------
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands_detector = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)

# -------------------
# HTML Endpoint
# -------------------
@app.get("/")
async def index():
    with open("index.html") as f:
        return HTMLResponse(f.read())

# -------------------
# Socket.IO Events
# -------------------
@sio.event
async def connect(sid, environ):
    print(f"[INFO] Client connected: {sid}")

@sio.event
async def disconnect(sid):
    print(f"[INFO] Client disconnected: {sid}")

@sio.event
async def video_frame(sid, data):
    """
    Receive a base64 frame from the client, detect hand landmarks,
    and send back landmarks.
    """
    try:
        # Decode base64 to numpy image
        img_bytes = base64.b64decode(data.split(",")[1])
        nparr = np.frombuffer(img_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        # Resize for speed
        frame = cv2.resize(frame, (320, 240))

        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process hands
        results = hands_detector.process(rgb_frame)
        landmarks_data = []

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                single_hand = []
                for lm in hand_landmarks.landmark:
                    single_hand.append({
                        "x": lm.x,
                        "y": lm.y,
                        "z": lm.z
                    })
                landmarks_data.append(single_hand)

        # Send back landmarks
        await sio.emit("hand_landmarks", {"landmarks": landmarks_data}, to=sid)

    except Exception as e:
        print("Error processing frame:", e)
        await sio.emit("hand_landmarks", {"landmarks": []}, to=sid)

# -------------------
# Run server using uvicorn
# -------------------
if __name__ == "__main__":
    import uvicorn
    print("Server running on http://0.0.0.0:8000")
    uvicorn.run(socket_app, host="0.0.0.0", port=8000)

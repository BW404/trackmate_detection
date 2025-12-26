import io
import cv2
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
import tensorflow as tf
from PIL import Image
import asyncio

app = FastAPI()

# Load TFLite model
interpreter = tf.lite.Interpreter(model_path="hand_landmark.tflite")
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# HTML file
@app.get("/")
async def get():
    with open("index.html") as f:
        return HTMLResponse(f.read())

# Helper function to run inference
def predict_hand_landmarks(frame):
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (input_details[0]['shape'][2], input_details[0]['shape'][1]))
    img = img.astype(np.float32) / 255.0
    img = np.expand_dims(img, axis=0)
    
    interpreter.set_tensor(input_details[0]['index'], img)
    interpreter.invoke()
    
    landmarks = interpreter.get_tensor(output_details[0]['index'])
    # landmarks shape: (1, 21, 3) -> convert to list
    return landmarks[0].tolist()

# WebSocket endpoint
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    print("Client connected")
    try:
        while True:
            # Receive binary frame
            data = await websocket.receive_bytes()
            img = Image.open(io.BytesIO(data)).convert('RGB')
            frame = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
            
            # Predict landmarks
            landmarks = predict_hand_landmarks(frame)
            
            # Send landmarks back
            await websocket.send_json({"landmarks": landmarks})
    except WebSocketDisconnect:
        print("Client disconnected")
    except Exception as e:
        print("Error:", e)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

import cv2
import numpy as np
import tflite_runtime.interpreter as tflite
from fastapi import FastAPI, WebSocket
from fastapi.responses import HTMLResponse
import asyncio

app = FastAPI()

# Load TFLite hand landmark model
hand_model = tflite.Interpreter(model_path="hand_landmark.tflite")
hand_model.allocate_tensors()
input_details = hand_model.get_input_details()
output_details = hand_model.get_output_details()

# Load YOLO model using OpenCV DNN
yolo_net = cv2.dnn.readNet("yolov8.onnx")  # replace with your YOLOv8 ONNX path
yolo_classes = ["mobile phone", "mouse", "keyboard", "monitor", "laptop", "cup", "water bottle"]

# Simple index page
html_content = """
<!DOCTYPE html>
<html>
<head>
<title>TrackMate Detection</title>
</head>
<body>
<h2>TrackMate Detection</h2>
<video id="video" width="640" height="480" autoplay></video>
<canvas id="canvas" width="640" height="480"></canvas>
<script>
const video = document.getElementById('video');
const canvas = document.getElementById('canvas');
const ctx = canvas.getContext('2d');

navigator.mediaDevices.getUserMedia({ video: true })
.then(stream => { video.srcObject = stream; });

function drawDot(x, y){
    ctx.fillStyle = 'red';
    ctx.beginPath();
    ctx.arc(x, y, 5, 0, 2 * Math.PI);  // small dot
    ctx.fill();
}

// Here you can implement WebSocket to receive coordinates from Python
</script>
</body>
</html>
"""

@app.get("/")
async def index():
    return HTMLResponse(html_content)

# WebSocket endpoint for real-time coordinates
@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    try:
        while True:
            # Here you should receive frame or commands if needed
            # For demo, we send dummy points
            await ws.send_json({"fingertips": [{"x": 100, "y": 100}]})
            await asyncio.sleep(0.03)  # ~30 FPS
    except Exception as e:
        print("WebSocket error:", e)
    finally:
        await ws.close()

# Function to run hand landmark detection
def detect_hand(frame):
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    input_data = np.expand_dims(img.astype(np.float32), axis=0)
    hand_model.set_tensor(input_details[0]['index'], input_data)
    hand_model.invoke()
    output_data = hand_model.get_tensor(output_details[0]['index'])
    return output_data  # coordinates

# Function to run YOLO detection
def detect_objects(frame):
    blob = cv2.dnn.blobFromImage(frame, 1/255.0, (640,640), swapRB=True, crop=False)
    yolo_net.setInput(blob)
    outputs = yolo_net.forward()
    boxes = []
    confidences = []
    class_ids = []
    h, w = frame.shape[:2]
    for detection in outputs[0]:
        scores = detection[5:]
        class_id = np.argmax(scores)
        confidence = scores[class_id]
        if confidence > 0.5:
            cx, cy, bw, bh = detection[0:4]
            x = int((cx - bw/2) * w)
            y = int((cy - bh/2) * h)
            bw = int(bw * w)
            bh = int(bh * h)
            boxes.append([x, y, bw, bh])
            confidences.append(float(confidence))
            class_ids.append(class_id)
    indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    result = []
    for i in indices:
        i = i[0]
        box = boxes[i]
        label = yolo_classes[class_ids[i]] if class_ids[i] < len(yolo_classes) else "Unknown"
        result.append({"label": label, "box": box, "confidence": confidences[i]})
    return result

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

import eventlet
eventlet.monkey_patch()

from flask import Flask, render_template
from flask_socketio import SocketIO
import base64
import cv2
import numpy as np
from hand_tracking import process_frame

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*", async_mode="eventlet")

# Set low resolution for faster processing
FRAME_WIDTH = 320
FRAME_HEIGHT = 240
JPEG_QUALITY = 50  # 0-100, lower is more compressed

@app.route("/")
def index():
    return render_template("index.html")

@socketio.on("frame")
def handle_frame(data):
    try:
        # Decode base64 image from client
        header, encoded = data.split(',', 1)
        img_bytes = base64.b64decode(encoded)
        nparr = np.frombuffer(img_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        # Resize for low latency
        frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))

        # Process hand landmarks
        frame = process_frame(frame)

        # Encode to low-quality JPEG
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY]
        _, buffer = cv2.imencode('.jpg', frame, encode_param)
        frame_b64 = base64.b64encode(buffer).decode('utf-8')

        # Send back to client
        socketio.emit("frame", {'image': "data:image/jpeg;base64," + frame_b64})

    except Exception as e:
        print("Error processing frame:", e)

if __name__ == "__main__":
    socketio.run(app, host="0.0.0.0", port=5000)

from flask import Flask, render_template
from flask_socketio import SocketIO
import cv2
import base64
from hand_tracking import process_frame
import numpy as np

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*")

@app.route("/")
def index():
    return render_template("index.html")

@socketio.on("frame")
def handle_frame(data):
    # Decode base64 frame from browser
    header, encoded = data.split(',', 1)
    img_bytes = base64.b64decode(encoded)
    nparr = np.frombuffer(img_bytes, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # Process frame
    frame = process_frame(frame)

    # Encode back to base64 and send to browser
    _, buffer = cv2.imencode('.jpg', frame)
    frame_b64 = base64.b64encode(buffer).decode('utf-8')
    socketio.emit("frame", {'image': "data:image/jpeg;base64," + frame_b64})

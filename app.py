from flask import Flask, render_template
from flask_socketio import SocketIO
import base64
import cv2
import numpy as np
from hand_tracking import process_frame
import eventlet
eventlet.monkey_patch()

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='eventlet')

@app.route("/")
def index():
    return render_template("index.html")

@socketio.on("frame")
def handle_frame(data):
    try:
        header, encoded = data.split(',', 1)
        img_bytes = base64.b64decode(encoded)
        nparr = np.frombuffer(img_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        # Process hand landmarks
        frame = process_frame(frame)

        # Encode frame and send back
        _, buffer = cv2.imencode('.jpg', frame)
        frame_b64 = base64.b64encode(buffer).decode('utf-8')
        socketio.emit("frame", {'image': "data:image/jpeg;base64," + frame_b64})
    except Exception as e:
        print("Error processing frame:", e)

if __name__ == "__main__":
    socketio.run(app, host="0.0.0.0", port=5000)

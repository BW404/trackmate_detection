from flask import Flask, render_template
from flask_socketio import SocketIO, emit
import cv2
import numpy as np
import base64
from hand_tracking import process_frame

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*")  # allow cross-origin for VPS

@app.route('/')
def index():
    return render_template('index.html')

@socketio.on('frame')
def handle_frame(data):
    try:
        img_bytes = base64.b64decode(data.split(',')[1])
        np_arr = np.frombuffer(img_bytes, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        frame = process_frame(frame)

        _, buffer = cv2.imencode('.jpg', frame)
        img_str = base64.b64encode(buffer).decode('utf-8')
        emit('processed_frame', f'data:image/jpeg;base64,{img_str}')
    except Exception as e:
        print("Error:", e)

if __name__ == '__main__':
    socketio.run(app, host='0.0.0.0', port=5000)

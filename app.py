from flask import Flask, render_template, request, Response
import cv2
import numpy as np
import base64
from hand_tracking import process_frame

app = Flask(__name__)

# GLOBAL frame counter
frame_count = 0


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/process", methods=["POST"])
def process():
    global frame_count
    frame_count += 1

    data = request.json["image"]
    img_data = base64.b64decode(data.split(",")[1])

    np_arr = np.frombuffer(img_data, np.uint8)
    frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    # Process every 3rd frame for speed
    run_detection = (frame_count % 1 == 0)
    frame = process_frame(frame, run_detection)

    _, buffer = cv2.imencode(".jpg", frame)
    return Response(buffer.tobytes(), mimetype="image/jpeg")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)

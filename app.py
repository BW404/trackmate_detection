from flask import Flask, render_template, request, jsonify
import cv2
import numpy as np
import base64
from hand_tracking import process_frame

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process', methods=['POST'])
def process():
    try:
        data = request.json['image']  # base64 string from JS
        img_bytes = base64.b64decode(data.split(',')[1])
        np_arr = np.frombuffer(img_bytes, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        frame = process_frame(frame)

        _, buffer = cv2.imencode('.jpg', frame)
        img_str = base64.b64encode(buffer).decode('utf-8')
        return jsonify({'image': f'data:image/jpeg;base64,{img_str}'})
    except Exception as e:
        print("Error:", e)
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)

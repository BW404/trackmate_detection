import cv2
import numpy as np
import tensorflow as tf
from ultralytics import YOLO

# -------------------------------
# Load YOLOv8 ONNX model
# -------------------------------
yolo_model_path = "yolov8n.onnx"  # make sure this exists
yolo_net = cv2.dnn.readNet(yolo_model_path)

# -------------------------------
# Load TFLite hand landmark model
# -------------------------------
hand_model_path = "hand_landmark.tflite"
hand_interpreter = tf.lite.Interpreter(model_path=hand_model_path)
hand_interpreter.allocate_tensors()

input_details = hand_interpreter.get_input_details()
output_details = hand_interpreter.get_output_details()

# -------------------------------
# Utility function: preprocess hand ROI
# -------------------------------
def preprocess_hand(image, size=(128, 128)):
    img = cv2.resize(image, size)
    img = img.astype(np.float32)
    img = img / 255.0
    img = np.expand_dims(img, axis=0)  # shape (1, H, W, C)
    return img

# -------------------------------
# Start webcam capture
# -------------------------------
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w, _ = frame.shape

    # -------------------------------
    # YOLOv8 ONNX detection
    # -------------------------------
    blob = cv2.dnn.blobFromImage(frame, 1/255.0, (640, 640), swapRB=True, crop=False)
    yolo_net.setInput(blob)
    outputs = yolo_net.forward()
    # YOLO postprocessing (simple)
    for det in outputs[0, 0]:
        confidence = det[2]
        if confidence > 0.5:
            x1 = int(det[3] * w)
            y1 = int(det[4] * h)
            x2 = int(det[5] * w)
            y2 = int(det[6] * h)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
            cv2.putText(frame, f"{confidence:.2f}", (x1, y1-5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)

    # -------------------------------
    # Hand detection with TFLite
    # -------------------------------
    hand_input = preprocess_hand(frame)
    hand_interpreter.set_tensor(input_details[0]['index'], hand_input)
    hand_interpreter.invoke()
    hand_output = hand_interpreter.get_tensor(output_details[0]['index'])
    
    # Draw landmarks (assuming 21 points)
    for point in hand_output[0]:
        cx, cy = int(point[0] * w), int(point[1] * h)
        cv2.circle(frame, (cx, cy), 3, (0,0,255), -1)

    # -------------------------------
    # Show frame
    # -------------------------------
    cv2.imshow("TrackMate Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

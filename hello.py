from flask import Flask, request, jsonify
from flask_cors import CORS
from flask_socketio import SocketIO, emit
import cv2
import math
from ultralytics import YOLO
import numpy as np

app = Flask(__name__)
CORS(app, origins=["http://localhost:3000"])
socketio = SocketIO(app)
model = YOLO("/best.pt")

@socketio.on('detect_labels')
def detect_labels(data):
    video_bytes = data['video']

    # Convert video bytes to numpy array
    nparr = np.frombuffer(video_bytes, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # Perform inference
    detections = model(frame)

    results = []

    # Process detections
    for d in detections:
        boxes = d.boxes

        for box in boxes:
            confidence = math.ceil((box.conf[0] * 100)) / 100

            if confidence >= 0.8:
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                results.append({'confidence': confidence, 'coordinates': (x1, y1, x2, y2)})

    emit('results', results)

if __name__ == '__main__':
    socketio.run(app, debug=True)

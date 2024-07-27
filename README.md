# cameraModuleBackend

## Project Summary
The main aim of the code is to create a continuous communication from a mobile interface to the YOLO8 model. The YOLO8 model is used for object detection and is pre-trained to detect labels, scan barcodes, etc. The mobile application is designed for visually impaired people to access information on a medicine. When the user scans the barcode on an object with their mobile phone, a continuous communication is established via WebSocket, and the pre-trained YOLO8 model provides the coordinates for the detected labels, barcodes, etc.

## Code Commands
Below are the commands needed to run the code:

```bash
pip install flask-socketio
export FLASK_APP=hello.py
flask run

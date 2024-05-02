import cv2
import numpy as np
import tensorflow as tf

# Load SSD MobileNet V2 COCO model
model_path = 'ssd_mobilenet_v2_coco/saved_model'
model = tf.saved_model.load(model_path)

# Fixed focal length (replace with your own estimate or fixed value)
fixed_focal_length_value = 1000  

# Function to perform inference on an image
def run_inference(model, image):
    # Convert the input image to uint8
    input_image = tf.cast(image, tf.uint8)

    # Create a batched tensor
    input_tensor = tf.convert_to_tensor([input_image])

    # Get the inference function from the model
    infer = model.signatures["serving_default"]

    # Perform inference
    detections = infer(input_tensor)

    return detections

# Function to draw bounding boxes on the image
def draw_boxes(image, boxes, scores, classes, class_names):
    h, w, _ = image.shape

    for i in range(boxes.shape[0]):
        if scores[i] > 0.5 and int(classes[i]) == 44:  # Assuming class 44 corresponds to a bottle in COCO dataset
            box = boxes[i] * np.array([h, w, h, w])
            ymin, xmin, ymax, xmax = box.astype(int)

            # Draw bounding box
            cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)

            # Display label "Bottle" inside the bounding box
            label = "Product"
            cv2.putText(image, label, (xmin, ymin - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # Calculate object width in pixels
            object_width_pixels = xmax - xmin

            # Calculate distance using a hypothetical bottle width of 0.1 meters
            real_world_bottle_width = 0.1
            distance = (real_world_bottle_width * fixed_focal_length_value) / object_width_pixels

            # Display the distance information
            cv2.putText(image, f"Distance: {distance:.2f} meters", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

# Open a video capture stream (0 for default camera)
cap = cv2.VideoCapture(1)

while True:
    ret, frame = cap.read()

    # Check if the frame is not empty
    if not ret or frame is None:
        continue

    # Perform inference
    try:
        input_image = cv2.resize(frame, (300, 300))
        input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)
        input_image = input_image.astype(np.uint8)  # Convert to uint8
    except cv2.error as e:
        print(f"Error resizing frame: {e}")
        continue

    detections = run_inference(model, input_image)

    # Draw bounding boxes and measure distance on the frame for the detected bottles
    draw_boxes(frame, detections['detection_boxes'][0].numpy(),
               detections['detection_scores'][0].numpy(),
               detections['detection_classes'][0].numpy(),
               class_names=[str(i) for i in range(91)])  # COCO class names

    # Display the resulting frame
    cv2.imshow('Bottle Detection', frame)

    # Break the loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture
cap.release()
cv2.destroyAllWindows()

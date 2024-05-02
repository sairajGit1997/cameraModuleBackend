import cv2
import pytesseract

# Load the input image
image_path = 'test1.png'  # Update the image path
image = cv2.imread(image_path)

# Function to preprocess the image
def preprocess_image(image):
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    # Perform adaptive thresholding to binarize the image
    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 11, 4)
    return thresh

# Preprocess the image
processed_image = preprocess_image(image)

try:
    # Perform OCR on the preprocessed image
    results = pytesseract.image_to_data(processed_image, output_type=pytesseract.Output.DICT)

    # Iterate over the detected text regions
    for i in range(len(results["text"])):
        # Extract the coordinates and dimensions of the bounding box
        x, y, w, h = results["left"][i], results["top"][i], results["width"][i], results["height"][i]

        # Extract the detected text
        text = results["text"][i]

        # Draw the bounding box around the detected text
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)


        # Put the detected text on the image
        cv2.putText(image, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Display the image with bounding boxes and detected text in a separate window
    cv2.imshow("Text Detection", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

except Exception as e:
    print(f"Error during OCR processing: {e}")


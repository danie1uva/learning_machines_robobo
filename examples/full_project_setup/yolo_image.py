from ultralytics import YOLO
import cv2
import torch
import numpy as np
import time

# Check if GPU is available
print("CUDA available:", torch.cuda.is_available())
print("Device name:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")

# Load YOLO model
model = YOLO("yolo11x.pt")

# Helper function for green detection using color segmentation
def detect_green_areas(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_green = np.array([40, 50, 50])  # Adjust HSV range for green
    upper_green = np.array([80, 255, 255])
    mask = cv2.inRange(hsv, lower_green, upper_green)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours, mask

# Process a single image
image_path = "testimage.jpg"  # Replace with your image path
frame = cv2.imread(image_path)

if frame is None:
    print(f"Error: Could not read image {image_path}")
else:
    # Measure YOLO inference time
    start_time = time.time()
    results = model.predict(source=image_path, device="cuda", save=False, verbose=False)
    yolo_time = time.time() - start_time
    print(f"YOLO inference time: {yolo_time:.4f} seconds")

    # Get bounding boxes from YOLO results
    yolo_boxes = results[0].boxes.xyxy.cpu().numpy()  # Bounding box coordinates
    yolo_scores = results[0].boxes.conf.cpu().numpy()  # Confidence scores
    yolo_classes = results[0].boxes.cls.cpu().numpy()  # Class indices

    # Measure green detection time
    start_time = time.time()
    contours, mask = detect_green_areas(frame)
    green_detection_time = time.time() - start_time
    print(f"Green detection time: {green_detection_time:.4f} seconds")

    # Draw YOLO bounding boxes filtered by green detection
    for box, score, cls in zip(yolo_boxes, yolo_scores, yolo_classes):
        x_min, y_min, x_max, y_max = map(int, box)
        roi = mask[y_min:y_max, x_min:x_max]
        green_area = cv2.countNonZero(roi)

        if green_area > 0.5 * (roi.shape[0] * roi.shape[1]):  # Threshold for green content
            label = f"Class {int(cls)}: {score:.2f}"
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
            cv2.putText(frame, label, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Highlight contours for green areas
    for contour in contours:
        (x, y, w, h) = cv2.boundingRect(contour)
        if w * h > 5000:  # Ignore small green areas
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, "Green Box", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Display the result
    cv2.imshow("Processed Image", frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

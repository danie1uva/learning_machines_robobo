from ultralytics import YOLO
import cv2
import torch
import numpy as np

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

# Open live video feed (0 = default webcam)
video_stream = cv2.VideoCapture(0)

if not video_stream.isOpened():
    print("Error: Could not open video stream.")
    exit()

while True:
    ret, frame = video_stream.read()
    if not ret:
        print("Error: Could not read frame.")
        break

    # Detect objects with YOLO
    results = model.predict(source=frame, device="cuda", save=False, verbose=False)
    
    # Get bounding boxes from YOLO results
    yolo_boxes = results[0].boxes.xyxy.cpu().numpy()  # Bounding box coordinates
    yolo_scores = results[0].boxes.conf.cpu().numpy()  # Confidence scores
    yolo_classes = results[0].boxes.cls.cpu().numpy()  # Class indices

    # Perform green detection
    contours, mask = detect_green_areas(frame)

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
    cv2.imshow("Live Green Object Detection", frame)

    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release video stream and close all OpenCV windows
video_stream.release()
cv2.destroyAllWindows()

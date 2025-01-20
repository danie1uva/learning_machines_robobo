from torchvision.models import detection
from imutils.video import VideoStream
from imutils.video import FPS
import numpy as np
import argparse
import imutils
import pickle
import torch
import time
import cv2
''' it says best to use "frcnn-mobilenet" for fast object detection and I think we want to be fast'''

ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", type=str, default="frcnn-resnet",
	choices=["frcnn-resnet", "frcnn-mobilenet", "retinanet"],
	help="name of the object detection model")
ap.add_argument("-l", "--labels", type=str, default="coco_classes.pickle",
	help="path to file containing list of categories in COCO dataset")
ap.add_argument("-c", "--confidence", type=float, default=0.5,
	help="minimum probability to filter weak detections")
args = vars(ap.parse_args())
# set the device we will be using to run the model
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# load the list of categories in the COCO dataset and then generate a
# set of bounding box colors for each class
CLASSES = pickle.loads(open(args["labels"], "rb").read())
# print('classes: ', CLASSES)
COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))
# initialize a dictionary containing model name and its corresponding 
# torchvision function call
MODELS = {
	"frcnn-resnet": detection.fasterrcnn_resnet50_fpn,
	"frcnn-mobilenet": detection.fasterrcnn_mobilenet_v3_large_320_fpn,
	"retinanet": detection.retinanet_resnet50_fpn
}
# load the model and set it to evaluation mode
model = MODELS[args["model"]](pretrained=True, progress=True,
	num_classes=len(CLASSES), pretrained_backbone=True).to(DEVICE)
model.eval()
# initialize the video stream, allow the camera sensor to warmup,
# and initialize the FPS counter
print("[INFO] starting video stream...")
vs = VideoStream(src=0).start()
time.sleep(2.0)
fps = FPS().start()

# Additional function for color segmentation
def detect_green_areas(frame):
    # Convert the frame to HSV color space
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    # Define the range for green color
    lower_green = np.array([40, 50, 50])  # Adjust values for your use case
    upper_green = np.array([80, 255, 255])
    # Create a mask for green areas
    mask = cv2.inRange(hsv, lower_green, upper_green)
    # Find contours of the green areas
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours, mask

# Integrate into the existing loop
while True:
    # Grab the frame from the video stream and resize it
    frame = vs.read()
    frame = imutils.resize(frame, width=400)
    orig = frame.copy()

    # Perform color segmentation to detect green areas
    contours, mask = detect_green_areas(frame)

    # Draw bounding boxes around detected green areas
    for contour in contours:
        # Compute the bounding box of the contour
        (x, y, w, h) = cv2.boundingRect(contour)
        # Filter out small regions
        if w * h > 500:  # Adjust the size threshold as needed
            cv2.rectangle(orig, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(orig, "Green Box", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Convert the frame to RGB channel ordering for the object detection model
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = frame.transpose((2, 0, 1))
    frame = np.expand_dims(frame, axis=0)
    frame = frame / 255.0
    frame = torch.FloatTensor(frame).to(DEVICE)

    # Run the frame through the object detection model
    detections = model(frame)[0]

    # Loop over the detections
    for i in range(len(detections["boxes"])):
        confidence = detections["scores"][i]
        if confidence > args["confidence"]:
            idx = int(detections["labels"][i])
            box = detections["boxes"][i].detach().cpu().numpy()
            (startX, startY, endX, endY) = box.astype("int")
            label = "{}: {:.2f}%".format(CLASSES[idx - 1], confidence * 100)
            cv2.rectangle(orig, (startX, startY), (endX, endY),
                          COLORS[idx], 2)
            y = startY - 15 if startY - 15 > 15 else startY + 15
            cv2.putText(orig, label, (startX, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx], 2)

    # Show the combined output frame
    cv2.imshow("Frame", orig)
    cv2.imshow("Mask", mask)

    # Break the loop if 'q' is pressed
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break

    # Update the FPS counter
    fps.update()

# Cleanup
fps.stop()
print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))
cv2.destroyAllWindows()
vs.stop()

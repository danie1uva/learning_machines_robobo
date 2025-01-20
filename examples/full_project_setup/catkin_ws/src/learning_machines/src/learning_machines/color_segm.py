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
import os
import time
''' it says best to use "frcnn-mobilenet" for fast object detection and I think we want to be fast'''

def detect_green_areas(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_green = np.array([40, 50, 50])
    upper_green = np.array([80, 255, 255])
    mask = cv2.inRange(hsv, lower_green, upper_green)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours, mask

def process_image(input_image_folder):
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    CLASSES = ['person', 'bicycle', 'car', 'motorbike', 'aeroplane', 'bus', 'train', 'truck', 'boat', 'trafficlight', 'firehydrant', 'streetsign', 'stopsign', 'parkingmeter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'hat', 'backpack', 'umbrella', 'shoe', 'eyeglasses', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sportsball', 'kite', 'baseballbat', 'baseballglove', 'skateboard', 'surfboard', 'tennisracket', 'bottle', 'plate', 'wineglass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hotdog', 'pizza', 'donut', 'cake', 'chair', 'sofa', 'pottedplant', 'bed', 'mirror', 'diningtable', 'window', 'desk', 'toilet', 'door', 'tvmonitor', 'laptop', 'mouse', 'remote', 'keyboard', 'cellphone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'blender', 'book', 'clock', 'vase', 'scissors', 'teddybear', 'hairdrier', 'toothbrush', 'hairbrush']
    COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))
    model = detection.fasterrcnn_mobilenet_v3_large_320_fpn(pretrained=True, progress=True,
                                                            num_classes=len(CLASSES), pretrained_backbone=True).to(DEVICE)
    model.eval()
    # input_image_folder = "catkin_ws/src/learning_machines/src/learning_machines/images_detection"  # Replace with your folder path
    image_files = [f for f in os.listdir(input_image_folder) if f.endswith(('.png', '.jpg', '.jpeg'))]

    for image_file in image_files:
        image_path = os.path.join(input_image_folder, image_file)
        frame = cv2.imread(image_path)

        if frame is None:
            print(f"Error: Could not read image {image_path}")
            continue

        orig = frame.copy()
        # start_time = time.time()  # Start timing

        contours, mask = detect_green_areas(frame)  # Perform color segmentation

        # elapsed_time = time.time() - start_time  # End timing
        # print(f"Segmentation time for {image_file}: {elapsed_time:.4f} seconds")  # Print elapsed time
        coordinates_boxes = []
        num_boxes = 0
        for contour in contours:
            (x, y, w, h) = cv2.boundingRect(contour)
            if w * h > 5000:
                # cv2.rectangle(orig, (x, y), (x + w, y + h), (0, 255, 0), 2)
                # cv2.putText(orig, "Green Box", (x, y - 10),
                #             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                coordinates_boxes.append(tuple([x,y,w,h]))
                num_boxes += 1
            print(coordinates_boxes)
        print(image_file)
        height, width, channels = frame.shape
        # Object detection
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = frame.transpose((2, 0, 1))
        frame = np.expand_dims(frame, axis=0)
        frame = frame / 255.0
        frame = torch.FloatTensor(frame).to(DEVICE)
        detections = model(frame)[0]
        print(frame.shape)
        # height, width, channels = frame.shape
        # print(f"Image Size: {width}x{height}")
        # print(f"Number of Channels: {channels}")

    # for i in range(len(detections["boxes"])):
    #     confidence = detections["scores"][i]
    #     if confidence > 0.5:
    #         print('here')
    #         idx = int(detections["labels"][i])
    #         box = detections["boxes"][i].detach().cpu().numpy()
    #         (startX, startY, endX, endY) = box.astype("int")
    #         label = "{}: {:.2f}%".format(CLASSES[idx - 1], confidence * 100)
    #         print(CLASSES[idx - 1])
    #         cv2.rectangle(orig, (startX, startY), (endX, endY), COLORS[idx], 2)
    #         y = startY - 15 if startY - 15 > 15 else startY + 15
    #         cv2.putText(orig, label, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx], 2)

    # cv2.imshow("Processed Image", orig)
    # cv2.waitKey(0)  # Wait for keypress to move to the next image
    # cv2.destroyAllWindows()
    return width, height, coordinates_boxes, num_boxes


'''
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
print('classes: ', CLASSES)
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
# print("[INFO] starting video stream...")
# vs = VideoStream(src=0).start()
# time.sleep(2.0)
# fps = FPS().start()


# Helper function for green box detection
def detect_green_areas(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_green = np.array([40, 50, 50])
    upper_green = np.array([80, 255, 255])
    mask = cv2.inRange(hsv, lower_green, upper_green)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours, mask

# Set up the model and labels
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CLASSES = pickle.loads(open("catkin_ws/src/learning_machines/src/learning_machines/coco_classes.pickle", "rb").read())
COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))
model = detection.fasterrcnn_mobilenet_v3_large_320_fpn(pretrained=True, progress=True,
                                                        num_classes=len(CLASSES), pretrained_backbone=True).to(DEVICE)
model.eval()

# Path to image folder
image_folder = "catkin_ws/src/learning_machines/src/learning_machines/images_detection"  # Replace with your folder path
image_files = [f for f in os.listdir(image_folder) if f.endswith(('.png', '.jpg', '.jpeg'))]

for image_file in image_files:
    image_path = os.path.join(image_folder, image_file)
    frame = cv2.imread(image_path)

    if frame is None:
        print(f"Error: Could not read image {image_path}")
        continue

    orig = frame.copy()
    # start_time = time.time()  # Start timing

    contours, mask = detect_green_areas(frame)  # Perform color segmentation

    # elapsed_time = time.time() - start_time  # End timing
    # print(f"Segmentation time for {image_file}: {elapsed_time:.4f} seconds")  # Print elapsed time


    for contour in contours:
        (x, y, w, h) = cv2.boundingRect(contour)
        if w * h > 5000:
            cv2.rectangle(orig, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(orig, "Green Box", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Object detection
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = frame.transpose((2, 0, 1))
    frame = np.expand_dims(frame, axis=0)
    frame = frame / 255.0
    frame = torch.FloatTensor(frame).to(DEVICE)
    detections = model(frame)[0]

    # for i in range(len(detections["boxes"])):
    #     confidence = detections["scores"][i]
    #     if confidence > 0.5:
    #         print('here')
    #         idx = int(detections["labels"][i])
    #         box = detections["boxes"][i].detach().cpu().numpy()
    #         (startX, startY, endX, endY) = box.astype("int")
    #         label = "{}: {:.2f}%".format(CLASSES[idx - 1], confidence * 100)
    #         print(CLASSES[idx - 1])
    #         cv2.rectangle(orig, (startX, startY), (endX, endY), COLORS[idx], 2)
    #         y = startY - 15 if startY - 15 > 15 else startY + 15
    #         cv2.putText(orig, label, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx], 2)

    cv2.imshow("Processed Image", orig)
    cv2.waitKey(0)  # Wait for keypress to move to the next image
    cv2.destroyAllWindows()
    '''
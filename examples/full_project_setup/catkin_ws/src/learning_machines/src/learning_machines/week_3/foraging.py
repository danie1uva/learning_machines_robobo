import cv2 
from datetime import datetime
from data_files import FIGURES_DIR
from robobo_interface import IRobobo
import numpy as np
import torch
# from torchvision.models import detection

from robobo_interface import SimulationRobobo

def take_picture(rob: IRobobo):
     image = rob.read_image_front()
     return image
    

def center_camera(rob: IRobobo):
    rob.set_phone_pan_blocking(123, 100)
    take_picture(rob)

def set_pan(rob: IRobobo, pan: int):
    rob.set_phone_pan_blocking(pan, 100)
    take_picture(rob)

def set_tilt(rob: IRobobo, tilt: int):
    rob.set_phone_tilt_blocking(tilt, 100)
    take_picture(rob)

def pivot(rob):
    rob.move_blocking(100, 0, 100)

def check_centering(list_of_coords):

    if list_of_coords[0] > 0.4 and list_of_coords[0] < 0.6:
        return True
    else:
        return False

def drive_straight(rob):
    sensors = rob.read_irs()
    while max(sensors) < 800:
        rob.move_blocking(100, 100, 1000)

def detect_green_areas(frame):
    """
    Detect green areas in the input image.
    
    Args:
        frame (numpy.ndarray): Input image.
    
    Returns:
        list: Contours of detected green areas.
    """
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_green = np.array([40, 50, 50])
    upper_green = np.array([80, 255, 255])
    mask = cv2.inRange(hsv, lower_green, upper_green)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours

def process_image(frame):
    """
    Process an image to detect bounding boxes and return relevant information.

    Args:
        frame (numpy.ndarray): Input image.

    Returns:
        tuple: (width, height, coordinates of bounding boxes, number of boxes)
    """
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load pre-trained model
    # model = detection.fasterrcnn_mobilenet_v3_large_320_fpn(pretrained=True).to(DEVICE)
    # model.eval()

    # Get dimensions of the image
    height, width, _ = frame.shape

    # Detect green areas
    contours = detect_green_areas(frame)

    # Extract bounding boxes
    coordinates_boxes = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if w * h > 5000:  # Minimum size threshold
            coordinates_boxes.append((x, y, w, h))

    # # Prepare image for object detection
    # frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # frame_tensor = torch.FloatTensor(frame_rgb.transpose((2, 0, 1)) / 255.0).unsqueeze(0).to(DEVICE)

    # # Perform object detection
    # with torch.no_grad():
    #     detections = model(frame_tensor)[0]

    # # Combine detections with green areas (optional; depends on task requirements)
    # # Additional processing logic can be added here if needed

    return width, height, coordinates_boxes, len(coordinates_boxes)

    
def detect_box(rob):    
    is_box_ahead = False 

    while not is_box_ahead:
        image = take_picture(rob) # stored in results/figures 
        width, height, coord_box, num_boxes = process_image(image) # outputs coordinates of box in image 
        print(width, height, coord_box, num_boxes)
        # if check_centering == True:
        #     drive_straight
        pivot(rob)
    

def forage(rob):
    if isinstance(rob, SimulationRobobo):
        rob.play_simulation()
    set_pan(rob, 123) # centers the camera
    set_tilt(rob, 90) # tilts the camera down
    counter = 0
    while True:
        detect_box(rob)
        counter += 1
        if counter == 50:
            break
    
    if isinstance(rob, SimulationRobobo):
        rob.stop_simulation()




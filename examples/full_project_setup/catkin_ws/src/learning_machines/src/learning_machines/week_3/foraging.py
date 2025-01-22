import cv2
from datetime import datetime
from pathlib import Path
from data_files import FIGURES_DIR, READINGS_DIR
from robobo_interface import (
    IRobobo,
    SimulationRobobo,
    HardwareRobobo,
)
import numpy as np

irs_positions = {
    "BackL": 0,
    "BackR": 1,
    "FrontL": 2,
    "FrontR": 3,
    "FrontC": 4,
    "FrontRR": 5,
    "BackC": 6,
    "FrontLL": 7,
}

def take_picture(rob: IRobobo):
    return rob.read_image_front()

def center_camera(rob: IRobobo):
    rob.set_phone_pan_blocking(123, 100)
    take_picture(rob)

def set_pan(rob: IRobobo, pan: int):
    rob.set_phone_pan_blocking(pan, 100)
    take_picture(rob)

def set_tilt(rob: IRobobo, tilt: int):
    rob.set_phone_tilt_blocking(tilt, 109)
    take_picture(rob)

def pivot(rob: IRobobo, direction: str):
    """Pivot the robot left or right."""
    if direction == "right":
        rob.move_blocking(50, -25, 150)
    elif direction == "left":
        rob.move_blocking(-25, 50, 150)

def save_debug_image(image, name):
    """Save image for debugging purposes."""
    Path(FIGURES_DIR).mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filepath = FIGURES_DIR / f"{name}_{timestamp}.png"
    cv2.imwrite(str(filepath), image)

def process_irs(irs):
    return [irs[7], irs[2], irs[4], irs[3], irs[5]]

def is_white_wall(image):
    """
    Determines if the obstacle in front is a white wall based on image analysis.
    """
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_white = np.array([0, 0, 200])
    upper_white = np.array([180, 50, 255])
    mask = cv2.inRange(hsv, lower_white, upper_white)
    white_area = cv2.countNonZero(mask)

    # If a significant portion of the frame is white, classify it as a wall
    return white_area > (image.shape[0] * image.shape[1] * 0.5)  # 30% of the frame

def avoid_collision(rob):
    """
    Avoid collision by reversing and turning right if an obstacle is detected.
    Uses front sensors for detection.
    """
    sensors = rob.read_irs()
    front_sensors = [sensors[irs_positions["FrontL"]], sensors[irs_positions["FrontR"]], sensors[irs_positions["FrontC"]]]

    if max(front_sensors) > 200:  # Adjust the threshold as needed
        image = take_picture(rob)
        if is_white_wall(image):
            print("White wall detected! Avoiding collision...")
            rob.move_blocking(-100, -100, 200)  # Move backward
            rob.move_blocking(50, -50, 300)  # Turn right
            return True  # Collision avoided
    return False  # No collision detected

def drive_straight_with_collision_avoidance(rob, margin, center_of_frame):
    """
    Drives straight while continuously checking for green boxes and avoiding collisions.
    Stops if the package is no longer in the frame or an obstacle is detected.
    """
    while True:
        if avoid_collision(rob):
            continue  # If collision avoided, reassess environment

        sensors = process_irs(rob.read_irs())
        if max(sensors) >= 500:
            break

        image = take_picture(rob)
        detected_boxes = detect_green_areas(image)

        if not detected_boxes:  # No green boxes detected
            print("Package successfully collected!")
            return True  # Confirm package is collected

        rob.move_blocking(100, 100, 450)  # Continue moving forward

def put_it_in_reverse_terry(rob):
    rob.move_blocking(-100, -100, 200)
    rob.move_blocking(50, -30, 100)

def detect_green_areas(frame):
    """
    Returns detected green boxes as bounding boxes.
    """
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_green = np.array([40, 50, 50])
    upper_green = np.array([80, 255, 255])
    mask = cv2.inRange(hsv, lower_green, upper_green)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    detected_boxes = []

    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        detected_boxes.append((x, y, w, h))

    return detected_boxes

def detect_box_with_collision_avoidance(rob, margin, debug=False):
    """
    Detects a green box in the frame, drives towards it, and avoids collisions.
    """
    width = rob.read_image_front().shape[1]
    center_of_frame = width // 2
    center_margin = width * 0.10  # Middle 10% margin

    while True:
        if avoid_collision(rob):
            continue  # If collision avoided, reassess environment

        image = take_picture(rob)
        detected_boxes = detect_green_areas(image)

        if detected_boxes:
            # Sort boxes: closest in y-axis first, then closest to the center
            detected_boxes.sort(key=lambda b: (b[1], abs((b[0] + b[2] / 2) - center_of_frame)))

            # Save the image with detected boxes
            if debug:
                save_debug_image(image, "detected_boxes")

            # Choose the closest box
            target_box = detected_boxes[0]
            box_center_x = target_box[0] + target_box[2] / 2

            # Determine pivot direction based on the box's position
            if box_center_x < center_of_frame:  # Box is on the left
                pivot(rob, direction="left")
            elif box_center_x > center_of_frame:  # Box is on the right
                pivot(rob, direction="right")

            # Reassess box positioning and drive straight
            collected = drive_straight_with_collision_avoidance(rob, margin, center_of_frame)
            if collected:
                print("Box collection confirmed.")
                return True  # Confirm the box is collected

        # No boxes detected; pivot
        pivot(rob, direction="right")  # Default pivot direction is right
        print("Searching for box...")

def forage(rob):
    if isinstance(rob, SimulationRobobo):
        rob.play_simulation()

    if isinstance(rob, HardwareRobobo):
        set_pan(rob, 123)

    set_tilt(rob, 100)

    boxes = 100

    while boxes > 0:
        success = detect_box_with_collision_avoidance(rob, margin=100, debug=True)
        if success:
            boxes -= 1
            print(f"Boxes left: {boxes}")

    if isinstance(rob, SimulationRobobo):
        rob.stop_simulation()

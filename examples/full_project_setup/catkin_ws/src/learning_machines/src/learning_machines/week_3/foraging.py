import cv2
from datetime import datetime
from data_files import FIGURES_DIR
from robobo_interface import IRobobo
import numpy as np
from robobo_interface import SimulationRobobo, HardwareRobobo


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


def pivot(rob: IRobobo):
    rob.move_blocking(50, -25, 100)


def check_centering(width, list_of_coords, margin=50):
    center_of_frame = width / 2
    left_margin = center_of_frame - margin
    right_margin = center_of_frame + margin
    for x, _, w, _ in list_of_coords:
        if x < right_margin and x + w > left_margin:
            box_center = x + w / 2
            return True, box_center
    return False, None


def process_irs(irs):
    return [irs[7], irs[2], irs[4], irs[3], irs[5]]


def drive_straight(rob, margin, center_of_frame):
    '''
    Drives straight while continuously checking for green boxes.
    Stops and pivots if no boxes are detected.
    '''
    while True:
        sensors = process_irs(rob.read_irs())
        if max(sensors) >= 500:
            break

        # Continuously check for green boxes
        image = take_picture(rob)
        detected_boxes = detect_green_areas(image)

        if not detected_boxes:  # No green boxes detected
            print("No boxes detected, pivoting...")
            pivot(rob)
            return

        valid_boxes = [
            box for box in detected_boxes
            if center_of_frame - margin <= box[0] + box[2] / 2 <= center_of_frame + margin
        ]

        if not valid_boxes:  # No valid boxes in the middle region
            print("No valid boxes detected, pivoting...")
            pivot(rob)
            return

        rob.move_blocking(100, 100, 250)  # Continue moving forward


def put_it_in_reverse_terry(rob):
    rob.move_blocking(-100, -100, 100)
    rob.move_blocking(50, 0, 100)


def detect_green_areas(frame):
    '''
    Returns detected green boxes as bounding boxes.
    '''
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


def detect_box(rob, margin, debug=False):
    '''
    Detects a green box in the frame and drives towards it.
    '''
    width = rob.read_image_front().shape[1]
    center_of_frame = width // 2
    center_margin = width * 0.15  # Middle 50% margin

    while True:
        image = take_picture(rob)
        detected_boxes = detect_green_areas(image)

        if detected_boxes:
            # Sort boxes: closest in y-axis first, then closest to the center
            detected_boxes.sort(key=lambda b: (b[1], abs((b[0] + b[2] / 2) - center_of_frame)))

            # Filter boxes in the middle 50% of the frame
            valid_boxes = [
                box for box in detected_boxes
                if center_of_frame - center_margin <= box[0] + box[2] / 2 <= center_of_frame + center_margin
            ]

            if valid_boxes:
                target_box = valid_boxes[0]  # Closest box
                box_center_x = target_box[0] + target_box[2] / 2

                # Check if turning is necessary
                while abs(box_center_x - center_of_frame) > margin:
                    turn_direction = 50 if box_center_x > center_of_frame else -50
                    rob.move_blocking(turn_direction, -turn_direction, 100)
                    image = take_picture(rob)
                    detected_boxes = detect_green_areas(image)
                    detected_boxes.sort(key=lambda b: (b[1], abs((b[0] + b[2] / 2) - center_of_frame)))
                    target_box = detected_boxes[0]
                    box_center_x = target_box[0] + target_box[2] / 2

                drive_straight(rob, margin, center_of_frame)
                print("Box collected!")
                return True

        pivot(rob)
        print("Searching for box...")

        sensors = rob.read_irs()
        front_sensors = process_irs(sensors)
        if min(front_sensors) > 50:
            print("Back it up now y'all")
            put_it_in_reverse_terry(rob)


def forage(rob):
    if isinstance(rob, SimulationRobobo):
        rob.play_simulation()

    if isinstance(rob, HardwareRobobo):
        set_pan(rob, 123)

    set_tilt(rob, 100)

    boxes = 8

    while boxes > 0:
        bool = detect_box(rob, margin=100, debug=True)
        if bool:
            boxes -= 1
            print(f"Boxes left: {boxes}")

    if isinstance(rob, SimulationRobobo):
        rob.stop_simulation()

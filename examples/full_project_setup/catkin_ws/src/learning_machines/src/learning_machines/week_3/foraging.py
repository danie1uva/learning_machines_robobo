import cv2
from datetime import datetime
from pathlib import Path
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


def pivot(rob: IRobobo, direction: str):
    """Pivot the robot left or right."""
    if direction == "right":
        rob.move_blocking(50, -25, 100)
    elif direction == "left":
        rob.move_blocking(-25, 50, 100)


def save_debug_image(image, name):
    """Save image for debugging purposes."""
    Path(FIGURES_DIR).mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filepath = FIGURES_DIR / f"{name}_{timestamp}.png"
    cv2.imwrite(str(filepath), image)


def process_irs(irs):
    return [irs[7], irs[2], irs[4], irs[3], irs[5]]


def drive_straight(rob, margin, center_of_frame):
    """
    Drives straight while continuously checking for green boxes.
    Stops if the package is no longer in the frame.
    """
    while True:
        sensors = process_irs(rob.read_irs())
        if max(sensors) >= 500:
            break

        image = take_picture(rob)
        detected_boxes = detect_green_areas(image)

        if not detected_boxes:  # No green boxes detected
            print("Package successfully collected!")
            return True  # Confirm package is collected

        rob.move_blocking(100, 100, 250)  # Continue moving forward


def put_it_in_reverse_terry(rob):
    rob.move_blocking(-100, -100, 100)
    rob.move_blocking(50, 0, 100)


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


def detect_box(rob, margin, debug=False):
    """
    Detects a green box in the frame and drives towards it.
    """
    width = rob.read_image_front().shape[1]
    center_of_frame = width // 2
    center_margin = width * 0.10  # Middle 50% margin

    while True:
        image = take_picture(rob)
        detected_boxes = detect_green_areas(image)

        if detected_boxes:
            # Sort boxes: closest in y-axis first, then closest to the center
            detected_boxes.sort(key=lambda b: (b[1], abs((b[0] + b[2] / 2) - center_of_frame)))

            # Save the image with detected boxes
            if debug:
                save_debug_image(image, "detected_boxes")

            # Process the closest box
            target_box = detected_boxes[0]
            box_center_x = target_box[0] + target_box[2] / 2

            # Determine pivot direction
            if abs(box_center_x - center_of_frame) > margin:
                turn_direction = "right" if box_center_x > center_of_frame else "left"
                pivot(rob, direction=turn_direction)

            # Drive forward to collect the box
            collected = drive_straight(rob, margin, center_of_frame)
            if collected:
                print("Box collection confirmed.")

                # Recheck for additional boxes
                image = take_picture(rob)
                detected_boxes = detect_green_areas(image)

                if detected_boxes:  # Another box detected
                    print("Another box detected, continuing collection process.")
                    continue  # Restart processing the next box

                return True  # Confirm the box is collected

        # No boxes detected; pivot to search
        pivot(rob, direction="right")  # Default pivot direction
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

    boxes = 7

    while boxes > 0:
        bool = detect_box(rob, margin=100, debug=True)
        if bool:
            boxes -= 1
            print(f"Boxes left: {boxes}")

    if isinstance(rob, SimulationRobobo):
        rob.stop_simulation()

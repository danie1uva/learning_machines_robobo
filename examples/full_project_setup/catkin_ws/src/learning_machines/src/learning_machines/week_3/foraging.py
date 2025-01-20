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

def drive_straight(rob):
    while True:
        sensors = process_irs(rob.read_irs())
        if max(sensors) >= 500:
            break
        rob.move_blocking(100, 100, 100)

def put_it_in_reverse_terry(rob):
    rob.move_blocking(-100, -100, 100)
    rob.move_blocking(50, 0, 100)

def detect_green_areas(frame, margin, width):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_green = np.array([40, 50, 50])
    upper_green = np.array([80, 255, 255])
    mask = cv2.inRange(hsv, lower_green, upper_green)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    center_of_frame = width / 2
    left_margin = center_of_frame - margin
    right_margin = center_of_frame + margin
    for contour in contours:
        x, _, w, _ = cv2.boundingRect(contour)
        if x < right_margin and x + w > left_margin:
            return True
    return False

def plot_contours_and_margin(frame, margin=100):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_green = np.array([40, 50, 50])
    upper_green = np.array([80, 255, 255])
    mask = cv2.inRange(hsv, lower_green, upper_green)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    frame_copy = frame.copy()
    center_x = frame_copy.shape[1] // 2
    cv2.line(frame_copy, (center_x - margin, 0), (center_x - margin, frame_copy.shape[0]), (0, 0, 255), 2)
    cv2.line(frame_copy, (center_x + margin, 0), (center_x + margin, frame_copy.shape[0]), (0, 0, 255), 2)
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(frame_copy, (x, y), (x + w, y + h), (0, 255, 0), 2)
    return frame_copy

def detect_box(rob, margin, debug=False, max_attempts=50):
    width = rob.read_image_front().shape[1]
    attempts = 0
    while attempts < max_attempts:
        image = take_picture(rob)
        if detect_green_areas(image, margin, width):

            if debug:
                current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                cv2.imwrite(str(FIGURES_DIR / f"contoured_image_{current_time}.png"), plot_contours_and_margin(image))

            drive_straight(rob)
            return
        
        pivot(rob)

        attempts += 1
        sensors = rob.read_irs()
        front_sensors = process_irs(sensors)
        if min(front_sensors) > 50:
            put_it_in_reverse_terry(rob)
    print("No box detected after maximum attempts. Exiting search.")

def forage(rob):
    if isinstance(rob, SimulationRobobo):
        rob.play_simulation()

    if isinstance(rob, HardwareRobobo):
        set_pan(rob, 123)

    set_tilt(rob, 100)

    counter = 0
    while counter < 50:

        detect_box(rob, margin=100, debug=True, max_attempts=50)
        counter += 1

    if isinstance(rob, SimulationRobobo):
        rob.stop_simulation()
import cv2
from datetime import datetime
from data_files import FIGURES_DIR
import numpy as np
from robobo_interface import SimulationRobobo, HardwareRobobo, IRobobo

class RobotNavigator:
    def __init__(self, rob: IRobobo, debug=False):
        self.rob = rob
        self.hardware = isinstance(rob, HardwareRobobo)
        self.sim = isinstance(rob, SimulationRobobo)
        self.debug = debug

    def take_picture(self):
        return self.rob.read_image_front()

    def center_camera(self):
        self.rob.set_phone_pan_blocking(123, 100)

    def set_pan(self, pan):
        self.rob.set_phone_pan_blocking(pan, 100)

    def set_tilt(self, tilt):
        self.rob.set_phone_tilt_blocking(tilt, 100)

    def pivot(self, direction):
        if self.sim:
            if direction == "right":
                self.rob.move_blocking(100, -25, 100)
            elif direction == "left":
                self.rob.move_blocking(-25, 100, 100)

        if self.hardware:
            if direction == "right":
                self.rob.move_blocking(100, -25, 500)
            elif direction == "left":
                self.rob.move_blocking(-25, 100, 500)

    def process_irs(self, irs):
        return [irs[7], irs[2], irs[4], irs[3], irs[5]]

    def check_robot_centered(self, coords, width):
        closest_box = max(coords, key=lambda x: x[1])
        section = self.map_box_to_section(closest_box, width)
        return section == 3
    
    def drive_straight(self):

        image = self.take_picture()
        coords = self.detect_green_areas(image)
        target_box_centered = self.check_robot_centered(coords, image.shape[1])

        while target_box_centered:
            last_sensors = self.process_irs(self.rob.read_irs())

            if max(last_sensors) >= 500:
                break
            
            if self.sim:
                self.rob.move_blocking(100, 100, 750)
            if self.hardware:
                self.rob.move_blocking(100, 100, 1000)

            new_sensors = self.process_irs(self.rob.read_irs())
            if max(new_sensors) < max(last_sensors) or not coords:
                break
            
            image = self.take_picture()
            coords = self.detect_green_areas(image)
            target_box_centered = self.check_robot_centered(coords, image.shape[1])


    def reverse(self):
        if self.sim:
            self.rob.move_blocking(-100, -100, 150)
            self.rob.move_blocking(100, -100, 100)
        
        if self.hardware:
            self.rob.move_blocking(-100, -100, 300)
            self.rob.move_blocking(100, -100, 200)

    def plot_contours_and_sections(self, frame):
        '''
        given a frame, this function will plot the contours of the green areas + the vertical sections we move based on
        '''

        if self.hardware:
            frame = cv2.rotate(frame, cv2.ROTATE_180)

        sections = 7
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        lower_green = np.array([40, 50, 50])
        upper_green = np.array([80, 255, 255])
        mask = cv2.inRange(hsv, lower_green, upper_green)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        frame_copy = frame.copy()
        height, width = frame_copy.shape[:2]
        section_width = width // sections

        for i in range(sections):
            x = i * section_width
            cv2.line(frame_copy, (x, 0), (x, height), (0, 0, 0), 1)
            label_x = x + section_width // 2
            cv2.putText(frame_copy, str(i), (label_x, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        threshold = 1000

        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            if w * h > threshold:
                cv2.rectangle(frame_copy, (x, y), (x + w, y + h), (0, 255, 0), 2)

        return frame_copy

    def detect_green_areas(self, frame):
        '''
        given a frame, returns the coordinates of the green boxes visible
        '''

        if self.hardware:
            frame = cv2.rotate(frame, cv2.ROTATE_180)

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        lower_green = np.array([40, 50, 50])
        upper_green = np.array([80, 255, 255])
        mask = cv2.inRange(hsv, lower_green, upper_green)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        coordinates_boxes = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            sim_threshold = 1000
            hardware_threshold = 1000 
            threshold = sim_threshold if self.sim else hardware_threshold
            if w * h > threshold:
                coordinates_boxes.append((x, y, w, h))
        return coordinates_boxes

    def save_image(self, image):
        current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S_%f")
        cv2.imwrite(str(FIGURES_DIR / f"contoured_image_{current_time}.png"), self.plot_contours_and_sections(image))

    def map_box_to_section(self, box, width):

        sections=7
        x, _, w, _ = box
        section_width = width / sections
        overlaps = [(i, min(x + w, (i + 1) * section_width) - max(x, i * section_width)) for i in range(sections)]
        return max(overlaps, key=lambda item: item[1])[0]

    def pivot_to_box(self, box_coordinates, width):
        '''
        Given the coordinates of the boxes in view, this functiom will pivot the robot to face the closest box
        '''
        closest_box = max(box_coordinates, key=lambda x: x[1])
        section = self.map_box_to_section(closest_box, width)
        movement_sim = {
            0: (0, 65, 100),
            1: (0, 55, 100),
            2: (0, 40, 100),
            3: (0, 0, 100),
            4: (40, 0, 100),
            5: (55, 0, 100),
            6: (65, 0, 100)
        }

        movement_hardware = {
            0: (0, 100, 250),
            1: (0, 100, 190),
            2: (0, 100, 175),
            3: (0, 0, 100),
            4: (100, 0, 175),
            5: (100, 0, 190),
            6: (100, 0, 250) 
        }
        movement = movement_hardware[section] if self.hardware else movement_sim[section]
        self.rob.move_blocking(*movement)

    def detect_box(self):
        '''
        main logic. the robot first takes a picture. if there is no box in the frame
        it pivots left or right and takes another picture. if there is a box/boxes in the frame,
        the robot moves to face the closest box and drives straight until its collided 
        '''

        counter = 0
        while True:
            counter += 1
            image_pre_pivot = self.take_picture()
            width = image_pre_pivot.shape[1]

            if self.debug:
                    self.save_image(image_pre_pivot)

            box_coords = self.detect_green_areas(image_pre_pivot)

            if box_coords:
                self.pivot_to_box(box_coords, width)

                if self.debug:
                    image_after_pivot = self.take_picture()
                    self.save_image(image_after_pivot)

                self.drive_straight()
                break
            
            # this may seem unnecessary but it is to avoid the scenario where the robot is stuck against
            # a wall, but at an angle. it ensures the robot flattens against the wall. once the robot is flat against the wall,
            # it will reverse and pivot in the opposite direction
             
            if counter % 3 == 0:
                self.pivot("left")
            else:
                self.pivot("right")

            sensors = self.rob.read_irs()
            front_sensors = self.process_irs(sensors)

            if min(front_sensors) > 50:
                self.reverse()

    def forage(self):
        if self.sim:
            self.rob.play_simulation()

        if self.hardware:
            self.center_camera() # in sim the camera begins centered.

        self.set_tilt(100)

        counter = 0
        while counter < 30:
            self.detect_box()
            counter += 1

    # def calibrate(self):
    #     if self.hardware:
    #         self.set_pan(123)
        
    #     self.set_tilt(105)
    #     print("tilt done") 
    #     self.rob.sleep(1)
    #     init_image = self.take_picture() # take picture of starting point
    #     print("image taken")
    #     self.save_image(init_image) # save the image
    #     coords = self.detect_green_areas(init_image) # detect the green areas
    #     self.pivot_to_box(coords, init_image.shape[1]) # pivot to the closest green area
    #     print("pivoted")
    #     self.rob.sleep(1)
    #     post_image = self.take_picture()
    #     self.save_image(post_image)
    #     coords = self.detect_green_areas(post_image)
    #     self.pivot_to_box(coords, post_image.shape[1])
    #     print("calibration done")

    #     print('reached')
        

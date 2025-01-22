import numpy as np
import gym 
from gym import spaces
import random
import cv2

from robobo_interface import IRobobo, SimulationRobobo, HardwareRobobo
from robobo_interface.datatypes import Orientation

class CoppeliaSimEnv(gym.Env):
    """
    Example Gym environment interface for CoppeliaSim + Robobo,
    using normalised sensor values in [0,1] and coordinates of green boxes,
    with a 360° sweep and a hard step limit.
    """

    def __init__(self, rob: IRobobo, num_initial_boxes: int = 7):
        super().__init__()
        self.rob = rob
        self.setting = "sim" if isinstance(rob, SimulationRobobo) else "hardware"

        # Observation = 5 IR sensors + up to 5 boxes * 4 coords
        self.max_boxes = 5
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(5 + self.max_boxes * 4,),
            dtype=np.float32
        )
        self.action_space = spaces.Discrete(5)

        self.init_pos = rob.get_position()
        self.init_ori = rob.get_orientation()
        self.state = None
        self.done = False
        self.episode_count = 0

        # If hardware, optionally adjust phone camera orientation
        if self.setting == "hardware":
            self.rob.set_phone_pan_blocking(123, 100)
        self.rob.set_phone_tilt_blocking(100, 100)

        # Camera shape
        frame = self.rob.read_image_front()
        self.camera_height, self.camera_width = frame.shape[:2]

        # Environment constants
        self.MAX_SENSOR_VAL = 1000.0
        self.proximity_y_threshold = 0.5  # Normalised threshold for "close" boxes
        self.num_boxes_remaining = num_initial_boxes

        # Bookkeeping
        self.green_boxes_last = []
        self.steps_in_episode = 0

        # Configurable step limits
        self.sweep_step = 150
        self.max_steps = 200

    def reset(self):
        self.episode_count += 1
        self.steps_in_episode = 0

        orientation = Orientation(
            roll=self.init_ori.roll,
            pitch=self.init_ori.pitch,
            yaw=self.init_ori.yaw
        )

        if self.episode_count % 20 == 0:
            orientation.pitch = random.randint(-90, 90)

        self.rob.set_position(self.init_pos, orientation)

        # Read sensors
        irs = self.rob.read_irs()
        raw_state = self._process_irs(irs)
        normalised_sensors = self._clamp_and_normalise(raw_state)

        # Detect boxes
        frame = self.rob.read_image_front()
        green_boxes = self._detect_green_areas(frame)
        normalised_boxes = self._normalise_and_pad_boxes(green_boxes)

        # Build initial state
        self.state = np.concatenate((normalised_sensors, normalised_boxes))
        self.green_boxes_last = green_boxes
        self.done = False

        return self.state.astype(np.float32)

    def step(self, action):
        self.steps_in_episode += 1

        # Apply chosen action
        wheels = self._determine_action(action)
        self.rob.move_blocking(wheels[0], wheels[1], wheels[2])

        # Read sensors
        irs = self.rob.read_irs()
        raw_front_sensors = self._process_irs(irs)
        normalised_sensors = self._clamp_and_normalise(raw_front_sensors)

        # Detect boxes
        frame = self.rob.read_image_front()
        green_boxes = self._detect_green_areas(frame)
        normalised_boxes = self._normalise_and_pad_boxes(green_boxes)

        next_state = np.concatenate((normalised_sensors, normalised_boxes))

        # ----------------------------
        # REWARD CALCULATION
        # ----------------------------
        reward = 0.0

        # 1) Partial progress reward
        reward += self._calculate_partial_progress_reward(
            self.green_boxes_last, green_boxes
        )

        # 2) Box collection reward + count
        boxes_collected, collection_reward = self._calculate_box_collection_reward(
            self.green_boxes_last, green_boxes
        )
        reward += collection_reward

        if boxes_collected > 0:
            self.num_boxes_remaining = max(0, self.num_boxes_remaining - boxes_collected)

        # 3) Collision penalty
        collision = self._check_collision(normalised_sensors, green_boxes)
        if collision:
            reward -= 10

        # 4) Time penalty
        time_penalty = 0.1 + 0.0005 * self.steps_in_episode
        reward -= time_penalty

        # 5) Completion bonus if all boxes collected
        if self.num_boxes_remaining == 0 and not collision:
            reward += 50

        # ----------------------------
        # EPISODE TERMINATION CHECKS
        # ----------------------------

        # A) Collision or all boxes collected
        if collision or (self.num_boxes_remaining == 0):
            self.done = True

        # B) Perform 360° sweep at step 150 to confirm no boxes in view
        elif self.steps_in_episode == self.sweep_step:
            self.done = True
            for _ in range(4):
                self.rob.move_blocking(-25, 100, 500)
                frame = self.rob.read_image_front()
                green_boxes = self._detect_green_areas(frame) 
                if len(green_boxes) > 0:
                    self.done = False
                    break
        
        # C) Hard step limit
        elif self.steps_in_episode >= self.max_steps:
            self.done = True

        # Update internal states
        self.state = next_state
        self.green_boxes_last = green_boxes

        info = {
            "boxes_remaining": self.num_boxes_remaining,
            "collision": collision,
            "collected_this_step": boxes_collected
        }

        return self.state.astype(np.float32), float(reward), self.done, info

    # ----------------------------
    # HELPER METHODS
    # ----------------------------
    def _normalise_and_pad_boxes(self, green_boxes):
        normalised_boxes = []
        for x, y, w, h in green_boxes:
            norm_x = x / self.camera_width
            norm_y = y / self.camera_height
            norm_w = w / self.camera_width
            norm_h = h / self.camera_height
            normalised_boxes.append([norm_x, norm_y, norm_w, norm_h])

        while len(normalised_boxes) < self.max_boxes:
            normalised_boxes.append([0.0, 0.0, 0.0, 0.0])
        return np.array(normalised_boxes[:self.max_boxes]).flatten()

    def _detect_green_areas(self, frame):
        if self.setting == "hardware":
            frame = cv2.rotate(frame, cv2.ROTATE_180)

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        lower_green = np.array([40, 50, 50])
        upper_green = np.array([80, 255, 255])
        mask = cv2.inRange(hsv, lower_green, upper_green)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        coordinates_boxes = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            threshold = 1000
            if w * h > threshold:
                coordinates_boxes.append((x, y, w, h))
        return coordinates_boxes

    def _clamp_and_normalise(self, sensor_vals):
        MAX_SENSOR_VAL = 1000.0
        clamped = np.clip(sensor_vals, 0.0, MAX_SENSOR_VAL)
        return clamped / MAX_SENSOR_VAL

    def _process_irs(self, sensor_vals):
        return [
            sensor_vals[7],
            sensor_vals[2],
            sensor_vals[4],
            sensor_vals[3],
            sensor_vals[5]
        ]

    def _determine_action(self, action_idx):
        if action_idx == 0:    # left
            return [-25, 100, 500]
        elif action_idx == 1:  # left-forward
            return [-25, 100, 300]
        elif action_idx == 2:  # forward
            return [100, 100, 500]
        elif action_idx == 3:  # right-forward
            return [100, -25, 300]
        else:                  # right
            return [100, -25, 500]

    def _check_collision(self, sensors, green_boxes):
        proximity_threshold = 0.6
        if max(sensors) > proximity_threshold:
            for (x, y, w, h) in green_boxes:
                bottom_y = y + h
                if bottom_y / self.camera_height > self.proximity_y_threshold:
                    return False  # Close to a green box, not a wall
            return True
        return False

    def _calculate_box_collection_reward(self, previous_boxes, current_boxes):
        reward = 0.0
        boxes_collected = 0
        for prev_box in previous_boxes:
            x, y, w, h = prev_box
            prev_bottom = y + h
            disappeared = not any(self._iou(prev_box, c) > 0.5 for c in current_boxes)
            if disappeared and (prev_bottom / self.camera_height > self.proximity_y_threshold):
                reward += 20
                boxes_collected += 1
        return boxes_collected, reward

    def _calculate_partial_progress_reward(self, prev_boxes, curr_boxes):
        small_reward = 0.0
        for curr_box in curr_boxes:
            best_iou = 0.0
            matched_box = None
            for pb in prev_boxes:
                iou_val = self._iou(curr_box, pb)
                if iou_val > best_iou:
                    best_iou = iou_val
                    matched_box = pb
            if matched_box and best_iou > 0.5:
                curr_bottom = curr_box[1] + curr_box[3]
                prev_bottom = matched_box[1] + matched_box[3]
                if curr_bottom > prev_bottom:
                    small_reward += 0.2
        return small_reward

    def _iou(self, box1, box2):
        x1, y1, w1, h1 = box1
        x2, y2, w2, h2 = box2
        inter_x1 = max(x1, x2)
        inter_y1 = max(y1, y2)
        inter_x2 = min(x1 + w1, x2 + w2)
        inter_y2 = min(y1 + h1, y2 + h2)

        inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
        union_area = w1 * h1 + w2 * h2 - inter_area
        return inter_area / union_area if union_area > 0 else 0

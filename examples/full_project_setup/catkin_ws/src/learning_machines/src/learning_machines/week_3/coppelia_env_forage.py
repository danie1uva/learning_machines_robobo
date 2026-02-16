import numpy as np
import gym 
from gym import spaces
import random
import cv2
from datetime import datetime
import time 

from robobo_interface import IRobobo, SimulationRobobo, HardwareRobobo
from robobo_interface.datatypes import Orientation
from data_files import FIGURES_DIR

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

        self.max_boxes = 4
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
        self.step_count = 0

        # If hardware, optionally adjust phone camera orientation
        if self.setting == "hardware":
            self.rob.set_phone_pan_blocking(123, 100)
        self.rob.set_phone_tilt_blocking(105, 100)

        # Camera shape
        frame = self.rob.read_image_front()
        self.camera_height, self.camera_width = frame.shape[:2]

        # Environment constants
        self.MAX_SENSOR_VAL = 1000.0
        self.proximity_y_threshold = 0.80  # Normalised threshold for "close" boxes
        self.proximity_threshold = 0.5
        self.num_initial_boxes = num_initial_boxes

        # Bookkeeping
        self.green_boxes_last = []
        self.steps_in_episode = 0
        self.collected_count_previous = 0

        # Configurable step limits
        self.sweep_step = 100
        self.max_steps = 150

    def reset(self):
        # Stop old episode
        if self.setting == "sim":
            self.rob.stop_simulation()
        
        # Start fresh
        if self.setting == "sim":
            self.rob.play_simulation()

        # this must be done after play_simulation
        if self.setting == "hardware":
            self.rob.set_phone_pan_blocking(123, 100)
        self.rob.set_phone_tilt_blocking(105, 100)

        self.episode_count += 1
        self.steps_in_episode = 0

        orientation = Orientation(
            roll=self.init_ori.roll,
            pitch=self.init_ori.pitch,
            yaw=self.init_ori.yaw
        )

        if self.step_count < 2500:  # First 25% of training
            if self.episode_count % 20 == 0:
                orientation.pitch = random.randint(-45, 45)  # Smaller perturbations early on

        elif self.step_count < 5000:  # 25-50% of training
            if self.episode_count % 10 == 0:
                orientation.pitch = random.randint(-60, 60)  # Gradually increasing range

        elif self.step_count < 7500:  # 50-75% of training
            if self.episode_count % 5 == 0:
                orientation.pitch = random.randint(-90, 90)  # Full range perturbations begin

        else:  # Final 25% of training
            if self.episode_count % 2 == 0:
                orientation.pitch = random.randint(-90, 90)  # Frequent perturbations


        self.rob.set_position(self.init_pos, orientation)

        self.state, _, _ = self._compute_state()
        self.green_boxes_last = [] 
        self.num_boxes_remaining = self.num_initial_boxes
        self.collected_count_previous = 0
        self.done = False

        return self.state.astype(np.float32)
        
    def step(self, action):
        if self.done:
            raise RuntimeError("Step called after episode is already done! Call reset() first.")

        self.steps_in_episode += 1
        self.step_count += 1

        # Apply chosen action
        wheels = self._determine_action(action)
        self.rob.move_blocking(wheels[0], wheels[1], wheels[2])

        next_state, green_boxes, normalised_sensors = self._compute_state()

        if self.setting == "sim":
            # i guess we only need to do this in the sim, while learning
            # ---------------------------------
            # REWARD CALCULATION
            # ---------------------------------
            reward = 0.0

            # 1) Partial progress (vision-based reward shaping)
            reward += self._calculate_partial_progress_reward(
                self.green_boxes_last, green_boxes
            )

            # 2) Box collection reward (using the simulator's "food collected" count)
            collected_count = self.rob.get_nr_food_collected()
            print(f"Collected count: {collected_count}")

            newly_collected = collected_count - self.collected_count_previous
            if newly_collected > 0:
                # For each newly collected box, give +30 reward (example)
                reward += 30 * newly_collected

            self.collected_count_previous = collected_count
            self.num_boxes_remaining = max(0, self.num_initial_boxes - collected_count)

            # 3) Collision penalty
            collision = self._check_collision(normalised_sensors, green_boxes)
            if collision:
                reward -= 50

            # 4) Time penalty
            time_penalty = 0.1 + 0.001 * self.steps_in_episode
            reward -= time_penalty

            # 5) Completion bonus if all boxes collected
            if self.num_boxes_remaining == 0 and not collision:
                reward += 100

            # ---------------------------------
            # EPISODE TERMINATION CHECKS
            # ---------------------------------

            # A) Collision or all boxes collected
            if collision or (self.num_boxes_remaining == 0):
                print("episode ended due to collision or all boxes collected")
                self.done = True

            # B) Perform 360° sweep at step sweep_step to confirm no boxes in view
            elif self.steps_in_episode == self.sweep_step:
                self.done = True
                for _ in range(5):
                    self.rob.move_blocking(-25, 75, 500)
                    frame = self.rob.read_image_front()
                    green_boxes = self._detect_green_areas(frame) 
                    if len(green_boxes) > 0:
                        self.done = False
                        break

            # C) Hard step limit
            elif self.steps_in_episode >= self.max_steps:
                self.done = True

        else:
            # Hardware: Reward is 0, termination is based on time only
            reward = 0.0
            if self.steps_in_episode >= self.max_steps:
                self.done = True

        # Update internal states
        self.state = next_state
        self.green_boxes_last = green_boxes

        info = {
            "boxes_remaining": self.num_boxes_remaining,
            "collision": collision,
            "newly_collected_this_step": newly_collected,
            "total_collected_so_far": collected_count
        }

        return self.state.astype(np.float32), float(reward), self.done, info

    # ----------------------------
    # HELPER METHODS
    # ----------------------------
    def _compute_state(self):
        # Read sensors
        irs = self.rob.read_irs()
        raw_state = self._process_irs(irs)
        normalised_sensors = self._clamp_and_normalise(raw_state)

        # Detect boxes
        frame = self.rob.read_image_front()
        green_boxes = self._detect_green_areas(frame)
        normalised_boxes = self._normalise_and_pad_boxes(green_boxes)

        # Build state
        state = np.concatenate((normalised_sensors, normalised_boxes))
        return state, green_boxes, normalised_sensors
    
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
        if self.setting == "sim":
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
        else:
                if action_idx == 0:    # left
                    return [-25, 100, 600]
                elif action_idx == 1:  # left-forward
                    return [-25, 100, 300]
                elif action_idx == 2:  # forward
                    return [100, 100, 800]
                elif action_idx == 3:  # right-forward
                    return [100, -25, 300]
                else:                  # right
                    return [100, -25, 600]

    def _check_collision(self, sensors, green_boxes):
        if max(sensors) > self.proximity_threshold:
            for (x, y, w, h) in green_boxes:
                bottom_y = min(y + h, self.camera_height)
                if bottom_y / self.camera_height > self.proximity_y_threshold:
                    return False  # Close to a green box, not a wall
            return True
        return False

    def _calculate_box_collection_reward(self, previous_boxes, current_boxes):
        """
        Awards a "collection" reward if a box that was visible and reasonably centered
        in the previous frame has now disappeared from view in the current frame.
        """
        reward = 0.0
        boxes_collected = 0

        for prev_box in previous_boxes:
            x, y, w, h = prev_box
            center_x = x + w / 2
            bottom = min(y + h, self.camera_height)

            # 1) Check if it was near the bottom & horizontally centered *in the previous frame*
            if (bottom / self.camera_height > self.proximity_y_threshold
                and (1/4 * self.camera_width < center_x < 3/4 * self.camera_width)):

                # 2) Check if it has disappeared now (no overlap with current boxes)
                disappeared = True
                for curr_box in current_boxes:
                    if self._iou(prev_box, curr_box) > 0.5:
                        # It's still there or a near match => Not disappeared
                        disappeared = False
                        break

                if disappeared:
                    print("Box collected!")
                    # If it was close + centered previously, but now it's gone => collected
                    reward += 30
                    boxes_collected += 1

        return boxes_collected, reward


    def _calculate_partial_progress_reward(self, prev_boxes, curr_boxes):
        """
        Award a small 'partial progress' reward when we see the *same* box
        in both the previous and current frame (IOU > 0.5),
        and that box has moved further down in the frame (indicating we're closer).

        This version is consistent with a 'previous -> current' check
        like your updated box-collection function.
        """
        small_reward = 0.0

        for prev_box in prev_boxes:
            x_p, y_p, w_p, h_p = prev_box
            prev_bottom = y_p + h_p

            # 1) Find the best-matching current box (highest IOU)
            best_iou = 0.0
            matched_box = None
            for curr_box in curr_boxes:
                iou_val = self._iou(prev_box, curr_box)
                if iou_val > best_iou:
                    best_iou = iou_val
                    matched_box = curr_box

            # 2) If the best IOU exceeds 0.5, treat it as "the same" box
            if matched_box and best_iou > 0.5:
                x_c, y_c, w_c, h_c = matched_box
                curr_bottom = y_c + h_c

                # 3) If it is lower in the current frame => partial progress
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

    def _check_what_camera_sees(self):
        '''
        save the current (processed) image to the figures directory
        '''
        frame = self.rob.read_image_front()

        if self.setting == "hardware":
            frame = cv2.rotate(frame, cv2.ROTATE_180)

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        lower_green = np.array([40, 50, 50])
        upper_green = np.array([80, 255, 255])
        mask = cv2.inRange(hsv, lower_green, upper_green)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        frame_copy = frame.copy()
        threshold = 1000
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            if w * h > threshold:
                cv2.rectangle(frame_copy, (x, y), (x + w, y + h), (0, 255, 0), 2)

        current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S_%f")
        cv2.imwrite(str(FIGURES_DIR / f"contoured_image_{current_time}.png"), frame_copy)

class CoppeliaSimEnvHardware(gym.Env):
    """
    A bare-bones environment for running a trained RL agent on hardware.
    No simulation calls or environment randomization logic.
    Observations come from IR sensors + front camera (like the sim env).
    Actions move the real robot wheels. Done is always False unless you manually override.
    """

    def __init__(self, rob: IRobobo, num_initial_boxes: int = 7):
        super().__init__()
        # Must be a hardware instance
        if not isinstance(rob, HardwareRobobo):
            raise ValueError("CoppeliaSimEnvHardware requires a HardwareRobobo instance.")

        self.rob = rob
        self.setting = "hardware"

        # Same observation/action shape as in sim
        self.max_boxes = 4
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(5 + self.max_boxes * 4,),
            dtype=np.float32
        )
        self.action_space = spaces.Discrete(5)

        # Camera shape
        frame = self.rob.read_image_front()
        self.camera_height, self.camera_width = frame.shape[:2]

        # Environment constants
        self.proximity_y_threshold = 0.80
        self.proximity_threshold = 0.5
        self.num_initial_boxes = num_initial_boxes

        # Bookkeeping
        self.state = None
        self.done = False
        self.steps_in_episode = 0
        self.collected_count_previous = 0

        # You can still keep track if needed
        self.num_boxes_remaining = self.num_initial_boxes

        # If desired, set phone orientation here (hardware only)
        self.rob.set_phone_pan_blocking(122, 100)
        self.rob.set_phone_tilt_blocking(92, 100)

    def reset(self):
        """
        Minimal reset: 
        - We do not reposition the hardware robot or randomize anything here.
        - Just reset counters and build initial observation.
        """
        self.done = False
        self.steps_in_episode = 0
        self.collected_count_previous = 0
        self.num_boxes_remaining = self.num_initial_boxes

        # Read initial state from sensors/camera
        self.state, _, _ = self._compute_state()
        return self.state.astype(np.float32)

    def step(self, action):
        """
        Execute the chosen action on the hardware robot.
        Return next_state, reward, done, info.
        By default, we set reward=0, done=False. 
        You can override if you want a time limit or manual stop.
        """
        if self.done:
            raise RuntimeError("step() called after environment was done. Call reset().")

        self.steps_in_episode += 1

        # Apply the chosen action
        wheels = self._determine_action(action)
        self.rob.move_blocking(wheels[0], wheels[1], wheels[2])

        # self._check_what_camera_sees()

        # Compute next observation
        next_state, green_boxes, normalised_sensors = self._compute_state()

        # In hardware mode, we won't do complicated collision checks or random termination.
        # For demonstration, let's set a trivial reward and keep done = False.
        reward = 0.0
        done = False

        # # If you want to track boxes collected (optional):
        # collected_count = self.rob.get_nr_food_collected()
        # newly_collected = collected_count - self.collected_count_previous
        # if newly_collected > 0:
        #     # You could give a reward for each newly collected box
        #     reward += 30 * newly_collected

        # self.collected_count_previous = collected_count
        # self.num_boxes_remaining = max(0, self.num_initial_boxes - collected_count)

        # If you want to stop when all boxes are collected:
        # if self.num_boxes_remaining == 0:
        #     done = True

        # Or define a hardware-based step limit:
        # if self.steps_in_episode >= 300:
        #     done = True

        self.state = next_state
        info = {}

        self.done = done
        return self.state.astype(np.float32), reward, done, info

    # ----------------------------
    # HELPER METHODS
    # ----------------------------
    def _compute_state(self):
        # Similar to your sim code: read IRs, read camera, detect boxes, etc.
        irs = self.rob.read_irs()
        raw_state = self._process_irs(irs)
        normalised_sensors = self._clamp_and_normalise(raw_state)

        frame = self.rob.read_image_front()
        green_boxes = self._detect_green_areas(frame)
        normalised_boxes = self._normalise_and_pad_boxes(green_boxes)

        state = np.concatenate((normalised_sensors, normalised_boxes))
        return state, green_boxes, normalised_sensors

    def _determine_action(self, action_idx):
        """
        Adjust these to what you found works best on hardware:
        (left_speed, right_speed, duration).
        The logic can differ from sim if the real robot needs different timings.
        """
        if action_idx == 0:    # left
            return [-25, 100, 500]
        elif action_idx == 1:  # left-forward
            return [-25, 100, 300]
        elif action_idx == 2:  # forward
            return [100, 100, 1000]
        elif action_idx == 3:  # right-forward
            return [100, -25, 300]
        else:                  # right
            return [100, -25, 500]

    def _detect_green_areas(self, frame):
        # On hardware, phone camera might be rotated:
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

    def _process_irs(self, sensor_vals):
        """
        Same IR ordering as sim. Adjust if hardware indexing is different.
        """
        # e.g. the hardware might have these sensor indexes in a different arrangement
        return [
            sensor_vals[7],
            sensor_vals[2],
            sensor_vals[4],
            sensor_vals[3],
            sensor_vals[5]
        ]

    def _clamp_and_normalise(self, sensor_vals):
        MAX_SENSOR_VAL = 1000.0
        clamped = np.clip(sensor_vals, 0.0, MAX_SENSOR_VAL)
        return clamped / MAX_SENSOR_VAL
    
    def _check_what_camera_sees(self):
        '''
        save the current (processed) image to the figures directory
        '''
        frame = self.rob.read_image_front()

        
        frame = cv2.rotate(frame, cv2.ROTATE_180)

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        lower_green = np.array([40, 50, 50])
        upper_green = np.array([80, 255, 255])
        mask = cv2.inRange(hsv, lower_green, upper_green)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        frame_copy = frame.copy()
        threshold = 1000
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            if w * h > threshold:
                cv2.rectangle(frame_copy, (x, y), (x + w, y + h), (0, 255, 0), 2)

        current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S_%f")
        cv2.imwrite(str(FIGURES_DIR / f"contoured_image_{current_time}.png"), frame_copy)
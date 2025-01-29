import gym
import numpy as np
from gym import spaces
import cv2
import random
import datetime

from collections import deque

from data_files import FIGURES_DIR
from robobo_interface import IRobobo, HardwareRobobo, SimulationRobobo, Orientation, Position

class CoppeliaSimEnv(gym.Env):
    def __init__(
        self,
        rob: IRobobo,
        randomize_frequency: int = 0,  # Start with no randomization
        puck_pos_range: float = 0.4
    ):
        """
        Single-stage environment:
         - The robot tries to pick up (or 'collect') the red box (puck).
         - Once collected, the robot is rewarded for reducing distance to the green zone.
         - Large final reward if the puck is inside the zone.
         - If the robot loses the puck after collecting, a penalty is applied.
         - A small negative step penalty encourages quick solutions.
        """
        super().__init__()
        self.rob = rob
        self.randomize_frequency = randomize_frequency  # Episodes between randomization
        self.puck_pos_range = puck_pos_range

        # Continuous action space: [left wheel, right wheel], normalized to [-1, 1]
        self.action_space = spaces.Box(
            low=np.array([-1.0, -1.0]),
            high=np.array([1.0, 1.0]),
            dtype=np.float32
        )

        # Observations: 5 IR sensors + bounding boxes for red puck + green zone
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(8 + 4 * 2,),
            dtype=np.float32
        )

        # Initial robot position and orientation
        self.init_pos = rob.get_position()
        self.init_ori = rob.get_orientation()
        self.puck_init_pos = rob.get_food_position()
        self.rob.set_phone_tilt_blocking(108, 100)

        self.state = None
        self.done = False
        self.episode_count = 0
        self.total_step_count = 0
        self.steps_in_episode = 0
        self.gathered_puck = 0

        frame = self.rob.read_image_front()
        self.camera_height, self.camera_width = frame.shape[:2]

        self.MAX_SENSOR_VAL = 1000.0
        # Tracks whether the robot currently "has" the red box in its arms
        self.puck_collected = False

        # Max steps per episode
        self.max_steps = 300

        # Counter to track episodes since last randomization
        self.randomize_counter = 0

    def set_randomize_frequency(self, frequency: int):
        """
        Adjusts the frequency of randomizing the red box's position.
        :param frequency: Number of episodes between randomizations.
                          Set to 0 to disable randomization.
        """
        self.randomize_frequency = frequency
        self.randomize_counter = 0  # Reset counter when frequency changes

    def reset(self):
        self.rob.stop_simulation()
        self.rob.play_simulation()
        self.rob.set_phone_tilt_blocking(108, 100)

        self.episode_count += 1
        self.steps_in_episode = 0
        self.gathered_puck = 0
        self.puck_collected = False
        self.done = False

        # Handle randomization based on frequency
        if self.randomize_frequency > 0 and self.randomize_counter >= self.randomize_frequency:

            # Randomize red box position
            center_puck_location = Position(-3.3,0.73,0.01)
            px = center_puck_location.x + np.random.uniform(-self.puck_pos_range, self.puck_pos_range)
            py = center_puck_location.y + np.random.uniform(-self.puck_pos_range, self.puck_pos_range)

            self.rob._sim.setObjectPosition(
                self.rob._sim.getObject("/Food"),
                [px, py, self.puck_init_pos.z]
            )
            self.randomize_counter = 0
        else:
            # Keep the box in front of the robot without randomization
            self.rob.set_position(self.init_pos, self.init_ori)
            self.randomize_counter += 1

        self.state, _, _ = self._compute_state()
        return self.state.astype(np.float32)

    def step(self, action):
        """
        Executes a step in the environment.
        :param action: Array with two elements [-1, 1] for left and right wheel speeds.
        :return: Tuple (state, reward, done, info)
        """
        # Small step penalty to encourage faster completion
        reward = -0.01

        self.steps_in_episode += 1
        self.total_step_count += 1

        # Scale action from [-1, 1] to [-25, 100]
        left_wheel = self._scale_action(action[0], -1.0, 1.0, -25, 100)
        right_wheel = self._scale_action(action[1], -1.0, 1.0, -25, 100)

        # Execute the action with a fixed duration (e.g., 500ms)
        self.rob.move_blocking(left_wheel, right_wheel, 500)

        # Compute the next state
        next_state, puck_box, _ = self._compute_state()

        info = {}

        # Check for collisions
        if self._wall_collision(next_state):
            reward -= 1.0
            self.done = True
            info['success'] = False
        else:
            # Calculate additional rewards and check if done
            addl_reward, done_flag = self._calculate_reward_and_done(puck_box)
            reward += addl_reward
            self.done = done_flag

            # Set success flag if the puck is in the green zone
            if self.done and self._puck_in_green_zone():
                info['success'] = True
            else:
                info['success'] = False

        # Check if maximum steps are reached
        if self.steps_in_episode >= self.max_steps:
            self.done = True
            # Optionally, set success to False if not already done
            if not self._puck_in_green_zone():
                info['success'] = False

        self.state = next_state
        return self.state.astype(np.float32), float(reward), self.done, info

    # ---------------------------------------
    # Reward Logic
    # ---------------------------------------
    def _calculate_reward_and_done(self, puck_box):
        """
        Calculates the reward based on the current state.
        :param puck_box: Bounding box of the red puck detected by the camera.
        :return: Tuple (additional_reward, done_flag)
        """
        reward = 0.0
        done = False

        # If the puck was collected but is no longer in contact, apply penalty and lose the puck
        if self.puck_collected and not self._puck_contact(puck_box):
            self.puck_collected = False
            reward -= 10.0  # Penalty for losing the box

        if not self.puck_collected:
            # Reward for being closer to the puck
            dist_puck = self._distance_of_robot_to_puck()
            reward += 5*self._distance_reward(dist_puck, alpha=3.0, min_dist=0.1)

            # If contact with the puck is established
            if self._puck_contact(puck_box):
                self.puck_collected = True
                self.gathered_puck += 1
                reward += 5.0  # Small bonus for collecting the puck
                if self.gathered_puck > 3:
                    done = True
        else:
            # Reward for being closer to the green zone
            dist_zone = self._distance_to_green_zone()
            reward += 10*self._distance_reward(dist_zone, alpha=3.0, min_dist=0.0)

            # If the puck is successfully in the green zone
            if self._puck_in_green_zone():
                reward += 20.0  # Large bonus for successful delivery
                done = True

        # Optional: Clip the reward to maintain stability
        reward = np.clip(reward, -5.0, 50.0)
        return reward, done

    # ---------------------------------------
    # Helper Methods
    # ---------------------------------------
    def _compute_state(self):
        """
        Computes the current state of the environment.
        :return: Tuple (state, puck_box, green_zone_box)
        """
        # Read IR sensors
        irs = self.rob.read_irs()
        raw_state = self._process_irs(irs)
        normalised_sensors = self._clamp_and_normalise(raw_state)

        # Read camera image and detect bounding boxes
        frame = self.rob.read_image_front()
        puck_box = self._detect_red_areas(frame)
        green_zone_box = self._detect_green_areas(frame)

        puck_state = self._normalise_box(puck_box)
        green_zone_state = self._normalise_box(green_zone_box)

        # Concatenate all observations
        obs = np.concatenate((normalised_sensors, puck_state, green_zone_state))
        return obs, puck_box, green_zone_box

    def _scale_action(self, val, src_min, src_max, dst_min, dst_max):
        # Linear mapping from [src_min, src_max] -> [dst_min, dst_max]
        return dst_min + (val - src_min) * (dst_max - dst_min) / (src_max - src_min)

    def _wall_collision(self, obs):
        sensors = obs[:8]
        side_sensors = sensors[0]+sensors[4]
        back_sensors = sensors[5:]
        return np.max(side_sensors) > 0.2 or np.max(back_sensors) > 0.4 

    def _process_irs(self, sensor_vals):
        """
        Processes IR sensor readings.
        :param sensor_vals: Raw IR sensor values.
        :return: List of selected IR sensor readings.
        """
        # Adjust indexing based on your hardware's sensor arrangement
        return [
            sensor_vals[7],
            sensor_vals[2],
            sensor_vals[4],
            sensor_vals[3],
            sensor_vals[5],
            sensor_vals[0],
            sensor_vals[1],
            sensor_vals[6]
        ]

    def _puck_contact(self, puck_box):
        """
        Determines if the robot is in contact with the puck.
        :param puck_box: Bounding box of the puck.
        :return: True if in contact, else False.
        """
        if puck_box is None:
            return False
        dist = self._distance_of_robot_to_puck()
        center_of_puck = puck_box[0] + puck_box[2] / 2
        is_centered = (center_of_puck > 2 * self.camera_width / 5) and \
                      (center_of_puck < 3 * self.camera_width / 5)
        return is_centered and dist < 0.2

    def _distance_of_robot_to_puck(self):
        """
        Calculates the Euclidean distance between the robot and the puck.
        :return: Distance value.
        """
        robot_pos = self.rob.get_position()
        puck_pos = self.rob.get_food_position()
        return np.linalg.norm([robot_pos.x - puck_pos.x, robot_pos.y - puck_pos.y])

    def _distance_to_green_zone(self):
        """
        Calculates the Euclidean distance between the robot (holding the puck) and the green zone.
        :return: Distance value.
        """
        zone_pos = self.rob.get_base_position()
        puck_pos = self.rob.get_food_position()
        return np.linalg.norm([zone_pos.x - puck_pos.x, zone_pos.y - puck_pos.y])

    def _puck_in_green_zone(self):
        """
        Checks if the puck is in the green zone.
        :return: True if in zone, else False.
        """
        return self.rob.base_detects_food()

    def _distance_reward(self, dist, alpha=3.0, min_dist=0.1):
        """
        Calculates a reward based on distance, encouraging the robot to minimize it.
        :param dist: Current distance.
        :param alpha: Scaling factor for the exponential function.
        :param min_dist: Minimum distance threshold.
        :return: Reward value.
        """
        if dist <= min_dist:
            return 1.0
        return float(np.exp(-alpha * (dist - min_dist)))

    def _clamp_and_normalise(self, vals):
        """
        Clamps and normalizes sensor values.
        :param vals: Raw sensor values.
        :return: Normalized sensor values.
        """
        clamped = np.clip(vals, 0.0, self.MAX_SENSOR_VAL)
        return clamped / self.MAX_SENSOR_VAL

    def _normalise_box(self, box):
        """
        Normalizes bounding box coordinates.
        :param box: Bounding box tuple (x, y, w, h).
        :return: Normalized bounding box array.
        """
        if box is None:
            return np.array([0.0, 0.0, 0.0, 0.0])
        x, y, w, h = box
        return np.array([
            x / self.camera_width,
            y / self.camera_height,
            w / self.camera_width,
            h / self.camera_height
        ])

    def _detect_red_areas(self, frame):
        """
        Detects red areas in the camera frame.
        :param frame: Current camera frame.
        :return: Bounding box of the largest red area or None.
        """
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        lower_red = np.array([0, 50, 50])
        upper_red = np.array([10, 255, 255])
        mask = cv2.inRange(hsv, lower_red, upper_red)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            largest = max(contours, key=cv2.contourArea)
            return cv2.boundingRect(largest)
        return None

    def _detect_green_areas(self, frame):
        """
        Detects green areas in the camera frame.
        :param frame: Current camera frame.
        :return: Bounding box of the largest green area or None.
        """
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        lower_green = np.array([40, 50, 50])
        upper_green = np.array([80, 255, 255])
        mask = cv2.inRange(hsv, lower_green, upper_green)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            largest = max(contours, key=cv2.contourArea)
            return cv2.boundingRect(largest)
        return None
    
    def _check_what_camera_sees(self):
        """
        Save the current (processed) image to the figures directory.
        """
        frame = self.rob.read_image_front()
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        lower_red = np.array([0, 50, 50])
        upper_red = np.array([10, 255, 255])
        lower_green = np.array([40, 50, 50])
        upper_green = np.array([80, 255, 255])

        red_mask = cv2.inRange(hsv, lower_red, upper_red)
        green_mask = cv2.inRange(hsv, lower_green, upper_green)

        red_contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        green_contours, _ = cv2.findContours(green_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        frame_copy = frame.copy()

        for contour in red_contours:
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(frame_copy, (x, y), (x + w, y + h), (0, 0, 255), 2)

        for contour in green_contours:
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(frame_copy, (x, y), (x + w, y + h), (0, 255, 0), 2)

        current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S_%f")
        cv2.imwrite(str(FIGURES_DIR / f"contoured_image_{current_time}.png"), frame_copy)

import gym
import numpy as np
from gym import spaces
import cv2
import random
import datetime

from data_files import FIGURES_DIR
from robobo_interface import IRobobo, HardwareRobobo, SimulationRobobo, Orientation, Position

class CoppeliaSimEnv(gym.Env):
    def __init__(
        self,
        rob: IRobobo,
        randomize_frequency: int = 0,
        puck_pos_range: float = 0.4
    ):
        super().__init__()
        self.rob = rob
        self.randomize_frequency = randomize_frequency
        self.puck_pos_range = puck_pos_range

        # Continuous action space: [-1..1], mapped to [-25..100]
        self.action_space = spaces.Box(
            low=np.array([-1.0, -1.0]),
            high=np.array([1.0, 1.0]),
            dtype=np.float32
        )

        # Observation: 8 IR sensors + bounding box for puck (4) + bounding box for green zone (4) => total=16
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(8 + 4*2,),
            dtype=np.float32
        )

        # Store initial robot positions for resetting
        self.init_pos = rob.get_position()
        self.init_ori = rob.get_orientation()
        self.puck_init_pos = rob.get_food_position()
        self.rob.set_phone_tilt_blocking(109, 100)

        self.state = None
        self.done = False
        self.episode_count = 0
        self.steps_in_episode = 0

        # Tracking monotonic improvement
        self.best_dist_to_puck_so_far = None
        self.best_dist_to_zone_so_far = None

        # Whether we hold the puck
        self.puck_collected = False

        # Episode limit
        self.max_steps = 300

        # Randomization control
        self.randomize_counter = 0

        # Additional metrics
        self.collected_puck_step = None
        self.final_goal_step = None

        frame = self.rob.read_image_front()
        self.camera_height, self.camera_width = frame.shape[:2]
        self.MAX_SENSOR_VAL = 1000.0

    def set_randomize_frequency(self, frequency: int):
        self.randomize_frequency = frequency
        self.randomize_counter = 0

    def reset(self):
        # Stop + start simulation
        self.rob.stop_simulation()
        self.rob.play_simulation()
        self.rob.set_phone_tilt_blocking(108, 100)

        self.episode_count += 1
        self.steps_in_episode = 0
        self.done = False

        # Clear puck state
        self.puck_collected = False
        self.collected_puck_step = None
        self.final_goal_step = None

        # Randomisation
        if (self.randomize_frequency > 0) and (self.randomize_counter >= self.randomize_frequency):
            center_puck_location = Position(-3.3, 0.73, 0.01)
            px = center_puck_location.x + np.random.uniform(-self.puck_pos_range, self.puck_pos_range)
            py = center_puck_location.y + np.random.uniform(-self.puck_pos_range, self.puck_pos_range)
            self.rob._sim.setObjectPosition(
                self.rob._sim.getObject("/Food"),
                [px, py, self.puck_init_pos.z]
            )
            self.randomize_counter = 0
        else:
            # Place robot at initial position, do not randomize
            self.rob.set_position(self.init_pos, self.init_ori)
            self.randomize_counter += 1

        # Compute the initial observation
        self.state, _, _ = self._compute_state()

        # Monotonic trackers
        self.best_dist_to_puck_so_far = self._distance_of_robot_to_puck()
        self.best_dist_to_zone_so_far = self._distance_to_green_zone()

        return self.state.astype(np.float32)

    def step(self, action):
        # Step penalty
        reward = -0.02
        self.steps_in_episode += 1

        # Scale action from [-1,1] to [-25,100]
        left_wheel = self._scale_action(action[0], -1.0, 1.0, -25, 100)
        right_wheel = self._scale_action(action[1], -1.0, 1.0, -25, 100)
        self.rob.move_blocking(left_wheel, right_wheel, 500)

        # Compute next state
        next_state, puck_box, _ = self._compute_state()
        done = False
        info = {}

        # If collision => big penalty and end
        if self._wall_collision(next_state):
            reward -= 1.0
            done = True
            info["success"] = False
        else:
            # Normal reward logic
            addl_reward, done_flag = self._calculate_reward_and_done(puck_box)
            reward += addl_reward
            done = done_flag

            # if done + in zone => success
            info["success"] = (done and self._puck_in_green_zone())

        # Check step limit
        if self.steps_in_episode >= self.max_steps:
            done = True
            info["success"] = info.get("success", False)

        self.state = next_state

        # If episode ended => fill final info metrics
        if done:
            # Did we collect the puck at all?
            puck_collected_in_episode = (self.collected_puck_step is not None)
            info["puck_collected_in_episode"] = puck_collected_in_episode
            if puck_collected_in_episode:
                info["steps_to_puck_collection"] = self.collected_puck_step
            if self.final_goal_step is not None:
                info["steps_to_final_goal"] = self.final_goal_step

        return self.state.astype(np.float32), float(reward), done, info

    # --------------------------
    # Reward Logic
    # --------------------------
    def _calculate_reward_and_done(self, puck_box):
        """
        Simple monotonic shaping:
        - If not collected => only reward if we set a new best distance to the puck.
        - If collected => only reward if we set a new best distance to the zone.
        - If we lose the puck => penalty.
        - If we deliver the puck => big reward + done.
        """
        reward = 0.0
        done = False

        dist_puck = self._distance_of_robot_to_puck()
        dist_zone = self._distance_to_green_zone()

        # Lost puck?
        if self.puck_collected and not self._puck_contact(puck_box):
            self.puck_collected = False
            reward -= 10.0

        # Not holding puck => approach puck monotonic
        if not self.puck_collected:
            if dist_puck < self.best_dist_to_puck_so_far:
                improvement = self.best_dist_to_puck_so_far - dist_puck
                reward += 5.0 * improvement
                self.best_dist_to_puck_so_far = dist_puck

            # If contact, mark puck_collected
            if self._puck_contact(puck_box) and not self.puck_collected:
                self.puck_collected = True
                reward += 5.0
                if self.collected_puck_step is None:
                    self.collected_puck_step = self.steps_in_episode

        # If we hold the puck => approach zone monotonic
        else:
            if dist_zone < self.best_dist_to_zone_so_far:
                improvement = self.best_dist_to_zone_so_far - dist_zone
                reward += 5.0 * improvement
                self.best_dist_to_zone_so_far = dist_zone

            # If we got the puck in zone => success
            if self._puck_in_green_zone():
                reward += 50.0
                done = True
                if self.final_goal_step is None:
                    self.final_goal_step = self.steps_in_episode

        return reward, done

    # --------------------------
    # Helper Methods
    # --------------------------
    def _compute_state(self):
        # IR sensors
        irs = self.rob.read_irs()
        raw_state = self._process_irs(irs)
        normalised_sensors = self._clamp_and_normalise(raw_state)

        # Camera read + bounding boxes
        frame = self.rob.read_image_front()
        frame = cv2.rotate(frame, cv2.ROTATE_180)

        puck_box = self._detect_red_areas(frame)
        green_zone_box = self._detect_green_areas(frame)

        puck_state = self._normalise_box(puck_box)
        green_zone_state = self._normalise_box(green_zone_box)

        obs = np.concatenate((normalised_sensors, puck_state, green_zone_state))
        return obs, puck_box, green_zone_box

    def _scale_action(self, val, src_min, src_max, dst_min, dst_max):
        return dst_min + (val - src_min) * (dst_max - dst_min) / (src_max - src_min)

    def _wall_collision(self, obs):
        sensors = obs[:8]
        side_sensors = sensors[0] + sensors[4]
        back_sensors = sensors[5:]
        return np.max(side_sensors) > 0.2 or np.max(back_sensors) > 0.4

    def _process_irs(self, sensor_vals):
        # Indexing depends on your IR arrangement
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
        if puck_box is None:
            return False
        dist = self._distance_of_robot_to_puck()
        center_of_puck = puck_box[0] + puck_box[2]/2
        is_centered = (center_of_puck > 2*self.camera_width/5) and \
                      (center_of_puck < 3*self.camera_width/5)
        return is_centered and dist < 0.2

    def _distance_of_robot_to_puck(self):
        robot_pos = self.rob.get_position()
        puck_pos = self.rob.get_food_position()
        return np.linalg.norm([robot_pos.x - puck_pos.x, robot_pos.y - puck_pos.y])

    def _distance_to_green_zone(self):
        zone_pos = self.rob.get_base_position()
        puck_pos = self.rob.get_food_position()
        return np.linalg.norm([zone_pos.x - puck_pos.x, zone_pos.y - puck_pos.y])

    def _puck_in_green_zone(self):
        return self.rob.base_detects_food()

    def _clamp_and_normalise(self, vals):
        clamped = np.clip(vals, 0.0, self.MAX_SENSOR_VAL)
        return clamped / self.MAX_SENSOR_VAL

    def _normalise_box(self, box):
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
        frame = self.rob.read_image_front()
        frame = cv2.rotate(frame, cv2.ROTATE_180)
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
            cv2.rectangle(frame_copy, (x, y), (x+w, y+h), (0, 0, 255), 2)
        for contour in green_contours:
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(frame_copy, (x, y), (x+w, y+h), (0, 255, 0), 2)

        current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S_%f")
        cv2.imwrite(str(FIGURES_DIR / f"contoured_image_{current_time}.png"), frame_copy)


class HardwareInferenceEnv(gym.Env):
    """
    A barebones environment class for real-hardware inference. 
    No training logic, no reward shaping, no episode termination.
    Just continuous actions => updated observation => done=False => info dictionary.
    """
    def __init__(self, rob: IRobobo):
        super().__init__()
        self.rob = rob

        # Actions: Normalised in [-1..1], mapped to e.g. [-25..100]
        self.action_space = spaces.Box(
            low=np.array([-1.0, -1.0]),
            high=np.array([1.0, 1.0]),
            dtype=np.float32
        )

        # Observations: For instance, 8 IR sensors + bounding box for puck + bounding box for green zone => total=16
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(16,),
            dtype=np.float32
        )

        # Retrieve camera specs if needed
        frame = self.rob.read_image_front()
        self.camera_height, self.camera_width = frame.shape[:2]

        self.MAX_SENSOR_VAL = 1000.0

        # We do not track episodes or max steps in hardware inference
        self.state = None

    def reset(self):
        """
        Called once at the start of inference. 
        We do not randomise or reset positions in real hardware. 
        Return an initial observation so the model can start.
        """
        # Possibly do any hardware-specific re-initialisation or calibration
        
        self.rob.set_phone_tilt_blocking(108, 100)
        self.rob.set_phone_pan_blocking(122, 100)

        # Return first obs
        obs, _, _ = self._compute_state()
        self.state = obs
        return obs

    def step(self, action):
        """
        A single inference step. 
        1) Scale the action from [-1..1] to your real wheel speed range.
        2) Command the robot to move for a fixed duration or until next step.
        3) Read sensors and camera => produce next observation.
        4) Return (observation, reward=0, done=False, info={})
        """
        # Scale actions e.g. [-1,1] -> [-25,100]
        left_wheel = self._scale_action(action[0], -1.0, 1.0, -25, 100)
        right_wheel = self._scale_action(action[1], -1.0, 1.0, -25, 100)

        # Execute movement
        # e.g. 500ms drive
        self.rob.move_blocking(left_wheel, right_wheel, 700)

        # Compute new state
        obs, puck_box, green_zone_box = self._compute_state()
        self.state = obs

        # In hardware inference, no training => reward=0, done=False
        reward = 0.0
        done = False
        info = {}

        return obs, reward, done, info

    def _compute_state(self):
        """
        Minimal sensor/camera reading to produce the observation.
        """
        # IR sensors
        irs = self.rob.read_irs()
        processed_irs = self._process_irs(irs)
        normalised_irs = self._clamp_and_normalise(processed_irs)

        # Image reading, detect boxes
        frame = self.rob.read_image_front()
        # If camera is reversed physically, rotate if needed
        frame = cv2.rotate(frame, cv2.ROTATE_180)

        puck_box = self._detect_red_areas(frame)
        green_box = self._detect_green_areas(frame)

        puck_state = self._normalise_box(puck_box)
        green_state = self._normalise_box(green_box)

        obs = np.concatenate((normalised_irs, puck_state, green_state))
        return obs, puck_box, green_box

    # --------------------------
    #  Utility Methods
    # --------------------------
    def _scale_action(self, val, src_min, src_max, dst_min, dst_max):
        return dst_min + (val - src_min) * (dst_max - dst_min) / (src_max - src_min)

    def _process_irs(self, sensor_vals):
        """
        Robot/hardware-specific indexing. 
        Return 8 IR sensor values if you have them, or fewer if needed.
        """
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

    def _clamp_and_normalise(self, vals):
        clamped = np.clip(vals, 0.0, self.MAX_SENSOR_VAL)
        return clamped / self.MAX_SENSOR_VAL

    def _normalise_box(self, box):
        """
        Convert boundingRect => normalised coords [x/cam_width, y/cam_height, w/cam_width, h/cam_height].
        Return zeros if no box found.
        """
        if box is None:
            return np.zeros(4, dtype=np.float32)
        x, y, w, h = box
        return np.array([
            x / self.camera_width,
            y / self.camera_height,
            w / self.camera_width,
            h / self.camera_height
        ], dtype=np.float32)

    def _detect_red_areas(self, frame):
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
        (Optional) Quick debugging method to see or record frames.
        In hardware, might save an image for logging or debugging.
        """
        frame = self.rob.read_image_front()
        frame = cv2.rotate(frame, cv2.ROTATE_180)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        lower_red = np.array([0, 50, 50])
        upper_red = np.array([10, 255, 255])
        red_mask = cv2.inRange(hsv, lower_red, upper_red)

        lower_green = np.array([40, 50, 50])
        upper_green = np.array([80, 255, 255])
        green_mask = cv2.inRange(hsv, lower_green, upper_green)

        red_contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        green_contours, _ = cv2.findContours(green_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        frame_copy = frame.copy()
        for contour in red_contours:
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(frame_copy, (x, y), (x+w, y+h), (0, 0, 255), 2)
        for contour in green_contours:
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(frame_copy, (x, y), (x + w, y + h), (0, 255, 0), 2)

        current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S_%f")
        cv2.imwrite(str(FIGURES_DIR / f"hardware_snapshot_{current_time}.png"), frame_copy)

import gym 
import numpy as np
from gym import spaces
import cv2 

from robobo_interface import IRobobo, HardwareRobobo, SimulationRobobo

class CoppeliaSimEnv(gym.Env):
    def __init__(self, rob: IRobobo, stage: int = 1):
        super().__init__()
        self.rob = rob
        self.stage = stage  # Current curriculum stage

        self.action_space = spaces.Box(
            low=np.array([-100, -100]),
            high=np.array([100, 100]),
            dtype=np.float32
)
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(5 + 4 * 2,), dtype=np.float32  # Sensors + puck + green zone
        )

        self.init_pos = rob.get_position()
        self.init_ori = rob.get_orientation()
        self.state = None
        self.done = False
        self.episode_count = 0
        self.total_step_count = 0
        self.steps_in_episode = 0

        self.rob.set_phone_tilt_blocking(105, 100)

        frame = self.rob.read_image_front()
        self.camera_height, self.camera_width = frame.shape[:2]

        self.MAX_SENSOR_VAL = 1000.0
        self.proximity_threshold = 0.5

        # For curriculum learning
        self.puck_reached = False

    def reset(self):

        self.rob.stop_simulation()
        self.rob.play_simulation()

        self.episode_count += 1
        self.steps_in_episode = 0
        self.puck_reached = False

        self.rob.set_position(self.init_pos, self.init_ori)
        self.state, _, _ = self._compute_state()
        self.done = False

        return self.state.astype(np.float32)

    def step(self, action):
        self.steps_in_episode += 1
        self.total_step_count += 1

        # Extract actions: left wheel, right wheel, duration
        left_wheel = float(action[0])
        right_wheel = float(action[1])

        # Apply action to the robot
        self.rob.move_blocking(left_wheel, right_wheel, 300)

        # Compute the new state, reward, and done flag
        next_state, puck_box, green_zone_box = self._compute_state()
        reward, self.done = self._calculate_reward_and_done(puck_box, green_zone_box)

        self.state = next_state

        return self.state.astype(np.float32), float(reward), self.done, {}


    # ----------------------------
    # HELPER METHODS
    # ----------------------------
    def _compute_state(self) -> np.ndarray:
        '''
        
        '''
        irs = self.rob.read_irs()
        raw_state = self._process_irs(irs)
        normalised_sensors = self._clamp_and_normalise(raw_state)

        frame = self.rob.read_image_front()
        puck_box = self._detect_red_areas(frame)
        green_zone_box = self._detect_green_areas(frame)

        puck_state = self._normalise_box(puck_box)
        green_zone_state = self._normalise_box(green_zone_box)

        state = np.concatenate((normalised_sensors, puck_state, green_zone_state))
        return state, puck_box, green_zone_box
    
    def _process_irs(self, sensor_vals) -> list:
        """
        Extract front sensors from sim reading. 
        """
        # e.g. the hardware might have these sensor indexes in a different arrangement
        return [
            sensor_vals[7],
            sensor_vals[2],
            sensor_vals[4],
            sensor_vals[3],
            sensor_vals[5]
        ]
    
    def _calculate_reward_and_done(self, puck_box, green_zone_box):
        reward = 0.0
        done = False

        if self.stage == 1:
            # Stage 1: Reward reaching the puck and making contact
            if puck_box is not None:
                
                reward += 1.0 / (1 + self._distance_of_robot_to_puck())  # Proximity reward
                if self._puck_contact(puck_box):
                    reward += 10.0  # Bonus for contact
                    done = True

        elif self.stage == 2:
            # Stage 2: Reward pushing puck towards green zone
            if self.puck_reached or self._puck_contact(puck_box):
                self.puck_reached = True
                reward += 0.1 / (1 + self._distance_to_green_zone())
                if self._puck_in_green_zone():
                    reward += 50.0
                    done = True

        elif self.stage == 3:
            # Stage 3: Full task
            if self._puck_in_green_zone():
                reward += 100.0
                done = True
            else:
                reward += 0.1 / (1 + self._distance_to_green_zone(puck_box, green_zone_box))

        # Reward scaling
        reward = np.clip(reward, -1.0, 1.0)
        return reward, done

    def _normalise_box(self, box) -> np.ndarray:
        '''' Normalise the bounding box coordinates '''
        if box is None:
            return np.array([0.0, 0.0, 0.0, 0.0])
        x, y, w, h = box
        return np.array([x / self.camera_width, y / self.camera_height,
                         w / self.camera_width, h / self.camera_height])

    def _detect_red_areas(self, frame) -> np.ndarray:
        '''
        Detect the red areas in the frame and return the bounding box
        '''
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        lower_red = np.array([0, 50, 50])
        upper_red = np.array([10, 255, 255])
        mask = cv2.inRange(hsv, lower_red, upper_red)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            return cv2.boundingRect(largest_contour)
        return None
    
    def _detect_green_areas(self, frame) -> np.ndarray:
        '''
        Detect the green areas in the frame and return the bounding box
        '''
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        lower_green = np.array([40, 50, 50])
        upper_green = np.array([80, 255, 255])
        mask = cv2.inRange(hsv, lower_green, upper_green)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            return cv2.boundingRect(largest_contour)
        return None

    def _distance_of_robot_to_puck(self) -> float:
        '''
        Compute the distance between the robot and the puck, using sim info
        '''
        robot_pos = self.rob.get_position()
        puck_pos = self.rob.get_food_position()

        robot_pos = np.array([robot_pos.x, robot_pos.y, robot_pos.z])
        puck_pos = np.array([puck_pos.x, puck_pos.y, puck_pos.z])
        return np.linalg.norm(robot_pos - puck_pos) 

    def _distance_to_green_zone(self) -> float:
        ''' Compute the distance between the puck and the green, using sim info'''
        zone_pos = self.rob.get_base_position()
        puck_pos = self.rob.get_food_position()

        zone_pos = np.array([zone_pos.x, zone_pos.y, zone_pos.z])
        puck_pos = np.array([puck_pos.x, puck_pos.y, puck_pos.z])

        return np.linalg.norm(zone_pos - puck_pos)

    def _puck_contact(self, puck_box) -> bool:
        ''' Check if the puck is in contact with the robot '''
        dist = self._distance_of_robot_to_puck() 
        center_of_puck = puck_box[0] + puck_box[2] / 2
        is_puck_central = (center_of_puck > 2*self.camera_width / 5) and (center_of_puck < 3 * self.camera_width / 5)
        return is_puck_central and dist < .2 # this number i calibrated from sim trials

    def _clamp_and_normalise(self, sensor_vals) -> np.ndarray:
        '''
        normalizes the sensor values to be between 0 and 1
        '''
        MAX_SENSOR_VAL = 1000.0
        clamped = np.clip(sensor_vals, 0.0, MAX_SENSOR_VAL)
        return clamped / MAX_SENSOR_VAL

    def _puck_in_green_zone(self) -> bool:
        ''' Check if the puck is in the green zone, sim method '''
        return self.rob.base_detects_food()



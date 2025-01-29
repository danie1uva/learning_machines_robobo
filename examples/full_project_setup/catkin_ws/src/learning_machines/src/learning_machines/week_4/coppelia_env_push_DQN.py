import gym
import numpy as np
from gym import spaces
import cv2
import random
import datetime

from data_files import FIGURES_DIR
from robobo_interface import IRobobo, HardwareRobobo, SimulationRobobo, Orientation

class CoppeliaSimEnvDQN(gym.Env):
    def __init__(
        self,
        rob: IRobobo,
        stage: int = 1,
        randomize_frequency: int = 1,
        robot_ori_range: float = 45,
        puck_pos_range: float = 0.4
    ):
        super().__init__()
        self.rob = rob
        self.stage = stage
        self.randomize_frequency = randomize_frequency
        self.robot_ori_range = robot_ori_range
        self.puck_pos_range = puck_pos_range

        # 7 discrete actions
        self.action_space = spaces.Discrete(7)

        # Observations: 5 IR sensors + bounding boxes for puck & green zone
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(5 + 4 * 2,),
            dtype=np.float32
        )

        # Initial positions
        self.init_pos = rob.get_position()
        self.init_ori = rob.get_orientation()
        self.puck_init_pos = rob.get_food_position()

        self.state = None
        self.done = False
        self.episode_count = 0
        self.total_step_count = 0
        self.steps_in_episode = 0

        frame = self.rob.read_image_front()
        self.camera_height, self.camera_width = frame.shape[:2]

        self.MAX_SENSOR_VAL = 1000.0
        self.puck_reached = False  # True when robot is (currently) in contact with puck
        self.max_steps = 300

    def reset(self):
        self.rob.stop_simulation()
        self.rob.play_simulation()
        self.rob.set_phone_tilt_blocking(105, 100)

        self.episode_count += 1
        self.steps_in_episode = 0
        self.puck_reached = False
        self.done = False

        # Randomise positions/orientations if freq is met
        if (self.episode_count % self.randomize_frequency) == 0:
            new_pitch = random.randint(-45, 45)
            new_orientation = Orientation(
                yaw=self.init_ori.yaw,
                pitch=new_pitch,
                roll=self.init_ori.roll
            )

            px = self.puck_init_pos.x + np.random.uniform(-self.puck_pos_range, self.puck_pos_range)
            py = self.puck_init_pos.y + np.random.uniform(-self.puck_pos_range, self.puck_pos_range)

            self.rob.set_position(
                position=self.init_pos,
                orientation=new_orientation
            )
            self.rob._sim.setObjectPosition(
                self.rob._sim.getObject("/Food"),
                [px, py, self.puck_init_pos.z]
            )
        else:
            # No randomisation
            self.rob.set_position(self.init_pos, self.init_ori)

        self.state, _, _ = self._compute_state()
        return self.state.astype(np.float32)

    def step(self, action_idx: int):
        """
        action_idx is an integer in [0..6],
        which we map to (left_speed, right_speed, duration).
        """
        reward = -0.01  # small step penalty
        self.steps_in_episode += 1
        self.total_step_count += 1

        left_wheel, right_wheel, duration = self._determine_action(action_idx)
        print(f"Step {self.total_step_count}, Action Idx: {action_idx}, "
              f"Mapped to: (L={left_wheel}, R={right_wheel}, ms={duration})")

        # Execute the action
        self.rob.move_blocking(left_wheel, right_wheel, duration)

        next_state, puck_box, _ = self._compute_state()

        # Check collision
        if self._front_collision(next_state):
            reward -= 1.0
            self.done = True
        else:
            addl_reward, is_done = self._calculate_reward_and_done(puck_box)
            reward += addl_reward
            self.done = is_done

        # Max step check
        if self.steps_in_episode >= self.max_steps:
            self.done = True
            # Optionally penalise timeouts
            # reward -= 1.0

        self.state = next_state
        print(f"Step {self.total_step_count}, Reward: {reward}, Done: {self.done}")
        print("Distance to puck:", self._distance_of_robot_to_puck())
        return self.state.astype(np.float32), float(reward), self.done, {}

    def _determine_action(self, action_idx: int):
        """
        Discrete action mapping:
          0 -> [L=-25, R=100, dur=500]
          1 -> [L=-25, R=100, dur=400]
          2 -> [L=-25, R=100, dur=300]
          3 -> [L=100, R=100, dur=500]
          4 -> [L=100, R=-25, dur=300]
          5 -> [L=100, R=-25, dur=400]
          6 -> [L=100, R=-25, dur=500]
        """
        if action_idx == 0:
            return [-25, 100, 500]
        elif action_idx == 1:
            return [-25, 100, 400]
        elif action_idx == 2:
            return [-25, 100, 300]
        elif action_idx == 3:
            return [100, 100, 500]
        elif action_idx == 4:
            return [100, -25, 300]
        elif action_idx == 5:
            return [100, -25, 400]
        else:  # 6
            return [100, -25, 500]

    # ------------------------------------------------
    # Reward & Termination
    # ------------------------------------------------
    def _calculate_reward_and_done(self, puck_box):
        reward = 0.0
        done = False

        if self.stage == 1:
            # Stage 1: approach puck
            distance = self._distance_of_robot_to_puck()
            proximity_reward = self._exponential_distance_reward(distance, alpha=2.0, min_dist=0.1)
            reward += proximity_reward

            # If in contact => done
            if self._puck_contact(puck_box):
                reward += 10.0
                done = True

        elif self.stage == 2:
            # Stage 2: push puck to green zone

            # -------------------------------------------
            # 1) If we previously had the puck but lost it, revert.
            #    (Optional: add a penalty for losing contact.)
            # -------------------------------------------
            if self.puck_reached and not self._puck_contact(puck_box):
                self.puck_reached = False
                # reward -= 2.0  # If you want a penalty for losing it

            if not self.puck_reached:
                # Robot does NOT currently have the puck
                distance = self._distance_of_robot_to_puck()
                reward += self._exponential_distance_reward(distance, alpha=2.0, min_dist=0.1)

                # If it just contacts the puck => switch to "puck_reached"
                if self._puck_contact(puck_box):
                    self.puck_reached = True
                    reward += 10.0
            else:
                # Robot has the puck => reward moving puck to green zone
                distance = self._distance_to_green_zone()
                reward += self._exponential_distance_reward(distance, alpha=2.0, min_dist=0.0)

                if self._puck_in_green_zone():
                    reward += 50.0
                    done = True

        reward = np.clip(reward, -5.0, 100.0)
        return reward, done

    # ------------------------------------------------
    # State & Collision Logic
    # ------------------------------------------------
    def _compute_state(self):
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

    def _front_collision(self, obs):
        front_sensors = obs[:5]  # first 5 IR sensors
        return np.max(front_sensors) > 0.5

    def _process_irs(self, sensor_vals):
        # Adjust indexing for your hardware
        return [
            sensor_vals[7],
            sensor_vals[2],
            sensor_vals[4],
            sensor_vals[3],
            sensor_vals[5]
        ]

    def _distance_of_robot_to_puck(self):
        robot_pos = self.rob.get_position()
        puck_pos = self.rob.get_food_position()
        rp = np.array([robot_pos.x, robot_pos.y])
        pp = np.array([puck_pos.x, puck_pos.y])
        return np.linalg.norm(rp - pp)

    def _distance_to_green_zone(self):
        zone_pos = self.rob.get_base_position()
        puck_pos = self.rob.get_food_position()
        zp = np.array([zone_pos.x, zone_pos.y])
        pp = np.array([puck_pos.x, puck_pos.y])
        return np.linalg.norm(zp - pp)

    def _puck_contact(self, puck_box):
        dist = self._distance_of_robot_to_puck()
        if puck_box is None:
            return False
        center_of_puck = puck_box[0] + puck_box[2] / 2
        is_puck_central = (center_of_puck > 2 * self.camera_width / 5) and \
                          (center_of_puck < 3 * self.camera_width / 5)
        return is_puck_central and dist < 0.2

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
            largest_contour = max(contours, key=cv2.contourArea)
            return cv2.boundingRect(largest_contour)
        return None

    def _detect_green_areas(self, frame):
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        lower_green = np.array([40, 50, 50])
        upper_green = np.array([80, 255, 255])
        mask = cv2.inRange(hsv, lower_green, upper_green)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            return cv2.boundingRect(largest_contour)
        return None

    def _exponential_distance_reward(self, d, alpha=2.0, min_dist=0.1):
        if d <= min_dist:
            return 1.0
        return float(np.exp(-alpha * (d - min_dist)))

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
        threshold = 250

        for contour in red_contours:
            x, y, w, h = cv2.boundingRect(contour)
            if w * h > threshold:
                cv2.rectangle(frame_copy, (x, y), (x + w, y + h), (0, 0, 255), 2)

        for contour in green_contours:
            x, y, w, h = cv2.boundingRect(contour)
            if w * h > threshold:
                cv2.rectangle(frame_copy, (x, y), (x + w, y + h), (0, 255, 0), 2)

        current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S_%f")
        cv2.imwrite(str(FIGURES_DIR / f"contoured_image_{current_time}.png"), frame_copy)

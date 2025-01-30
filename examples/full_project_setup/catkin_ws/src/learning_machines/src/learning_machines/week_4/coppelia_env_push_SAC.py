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
        randomize_frequency: int = 0,
        puck_pos_range: float = 0.4
    ):
        super().__init__()
        self.rob = rob
        self.randomize_frequency = randomize_frequency
        self.puck_pos_range = puck_pos_range

        # Continuous action space: [-1..1] mapped internally to [-25..100].
        self.action_space = spaces.Box(
            low=np.array([-1.0, -1.0]),
            high=np.array([1.0, 1.0]),
            dtype=np.float32
        )

        # Observations: 8 IR sensors + bounding box (4) for puck + bounding box (4) for green zone
        # total = 16
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(8 + 4*2,),
            dtype=np.float32
        )

        # Initial positions
        self.init_pos = rob.get_position()
        self.init_ori = rob.get_orientation()
        self.puck_init_pos = rob.get_food_position()
        self.rob.set_phone_tilt_blocking(109, 100)

        self.state = None
        self.done = False
        self.episode_count = 0
        self.total_step_count = 0
        self.steps_in_episode = 0
        self.gathered_puck = 0

        # Tracking for difference-based distance rewards
        self.prev_dist_to_puck = None
        self.prev_dist_to_zone = None

        # Whether we currently hold the puck
        self.puck_collected = False

        # Episode can last up to 300 steps
        self.max_steps = 300

        # For randomization
        self.randomize_counter = 0

        # -------------------
        # Additional tracking for new metrics
        # -------------------
        self.collected_puck_step = None       # Step at which the puck was first collected
        self.final_goal_step = None           # Step at which the puck reached the green zone

        frame = self.rob.read_image_front()
        self.camera_height, self.camera_width = frame.shape[:2]
        self.MAX_SENSOR_VAL = 1000.0

    def set_randomize_frequency(self, frequency: int):
        self.randomize_frequency = frequency
        self.randomize_counter = 0

    def reset(self):
        self.rob.stop_simulation()
        self.rob.play_simulation()
        self.rob.set_phone_tilt_blocking(108, 100)

        self.episode_count += 1
        self.steps_in_episode = 0
        self.gathered_puck = 0
        self.puck_collected = False
        self.done = False
        # Initialize the monotonic "best" distances:
        self.best_dist_to_puck_so_far = self._distance_of_robot_to_puck()
        self.best_dist_to_zone_so_far = self._distance_to_green_zone()

        # New metrics tracking
        self.collected_puck_step = None
        self.final_goal_step = None

        # Randomization logic
        if self.randomize_frequency > 0 and self.randomize_counter >= self.randomize_frequency:
            center_puck_location = Position(-3.3, 0.73, 0.01)
            px = center_puck_location.x + np.random.uniform(-self.puck_pos_range, self.puck_pos_range)
            py = center_puck_location.y + np.random.uniform(-self.puck_pos_range, self.puck_pos_range)
            self.rob._sim.setObjectPosition(
                self.rob._sim.getObject("/Food"),
                [px, py, self.puck_init_pos.z]
            )
            self.randomize_counter = 0
        else:
            self.rob.set_position(self.init_pos, self.init_ori)
            self.randomize_counter += 1

        # Compute state
        self.state, _, _ = self._compute_state()

        # Reset distance trackers
        self.prev_dist_to_puck = self._distance_of_robot_to_puck()
        self.prev_dist_to_zone = self._distance_to_green_zone()
        return self.state.astype(np.float32)

    def step(self, action):
        reward = -0.01  # step penalty
        self.steps_in_episode += 1
        self.total_step_count += 1

        # Scale actions
        left_wheel = self._scale_action(action[0], -1.0, 1.0, -25, 100)
        right_wheel = self._scale_action(action[1], -1.0, 1.0, -25, 100)
        self.rob.move_blocking(left_wheel, right_wheel, 500)

        next_state, puck_box, _ = self._compute_state()

        info = {}

        # Check collisions
        if self._wall_collision(next_state):
            reward -= 1.0
            self.done = True
            info["success"] = False
        else:
            addl_reward, done_flag = self._calculate_reward_and_done(puck_box)
            reward += addl_reward
            self.done = done_flag

            if self.done and self._puck_in_green_zone():
                info["success"] = True
            else:
                info["success"] = False

        # Max steps
        if self.steps_in_episode >= self.max_steps:
            self.done = True
            if not self._puck_in_green_zone():
                info["success"] = False

        self.state = next_state

        # -----------------
        # Final step => populate custom metrics
        # -----------------
        if self.done:
            # Was the puck collected at any point?
            puck_collected_in_episode = (self.collected_puck_step is not None)
            info["puck_collected_in_episode"] = puck_collected_in_episode
            # If it was collected, how many steps did it take?
            if puck_collected_in_episode:
                info["steps_to_puck_collection"] = self.collected_puck_step
            # If final goal reached, how many steps total?
            if self.final_goal_step is not None:
                info["steps_to_final_goal"] = self.final_goal_step

        return self.state.astype(np.float32), float(reward), self.done, info

    # --------------------------
    # Reward Logic
    # --------------------------
    def _calculate_reward_and_done(self, puck_box):
        reward = 0.0
        done = False

        dist_puck = self._distance_of_robot_to_puck()
        dist_zone = self._distance_to_green_zone()

        # 1) If we had the puck but lost it => penalty
        if self.puck_collected and not self._puck_contact(puck_box):
            self.puck_collected = False
            reward -= 10.0

        # 2) If we do NOT have the puck => reward monotonic improvement in dist_puck
        if not self.puck_collected:
            # Compare current dist_puck to the best (smallest) so far
            if dist_puck < self.best_dist_to_puck_so_far:
                improvement = self.best_dist_to_puck_so_far - dist_puck
                # Scale the shaping; tweak factor (e.g. 5.0) as you see fit
                reward += 5.0 * improvement
                self.best_dist_to_puck_so_far = dist_puck  # Update best

            # If we just collected the puck now
            if self._puck_contact(puck_box) and not self.puck_collected:
                self.puck_collected = True
                self.gathered_puck += 1
                reward += 10.0
                if self.collected_puck_step is None:
                    self.collected_puck_step = self.steps_in_episode

                # If collects multiple times => end early
                if self.gathered_puck > 3:
                    done = True

        # 3) If we have the puck => reward monotonic improvement in dist_zone
        else:
            if dist_zone < self.best_dist_to_zone_so_far:
                improvement = self.best_dist_to_zone_so_far - dist_zone
                reward += 5.0 * improvement
                self.best_dist_to_zone_so_far = dist_zone

            # Check if we got the puck into the green zone => success
            if self._puck_in_green_zone():
                reward += 30.0
                done = True
                if self.final_goal_step is None:
                    self.final_goal_step = self.steps_in_episode

        return reward, done


    # --------------------------
    # Helper Methods
    # --------------------------
    def _compute_state(self):
        irs = self.rob.read_irs()
        raw_state = self._process_irs(irs)
        normalised_sensors = self._clamp_and_normalise(raw_state)

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
        center_of_puck = puck_box[0] + puck_box[2] / 2
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

    def _distance_reward(self, dist, alpha=3.0, min_dist=0.1):
        if dist <= min_dist:
            return 1.0
        return float(np.exp(-alpha * (dist - min_dist)))

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

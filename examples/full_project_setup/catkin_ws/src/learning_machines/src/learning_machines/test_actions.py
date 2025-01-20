import cv2

from data_files import FIGURES_DIR
from robobo_interface import (
    IRobobo,
    Emotion,
    LedId,
    LedColor,
    SoundEmotion,
    SimulationRobobo,
    HardwareRobobo,
)

from datetime import datetime


def test_emotions(rob: IRobobo):
    rob.set_emotion(Emotion.HAPPY)
    rob.talk("Hello")
    rob.play_emotion_sound(SoundEmotion.PURR)
    rob.set_led(LedId.FRONTCENTER, LedColor.GREEN)


def test_move_and_wheel_reset(rob: IRobobo):
    rob.move_blocking(100, 100, 1000)
    print("before reset: ", rob.read_wheels())
    rob.reset_wheels()
    rob.sleep(1)
    print("after reset: ", rob.read_wheels())


def test_sensors(rob: IRobobo):
    print("IRS data: ", rob.read_irs())
    image = rob.read_image_front()
    cv2.imwrite(str(FIGURES_DIR / "photo.png"), image)
    print("Phone pan: ", rob.read_phone_pan())
    print("Phone tilt: ", rob.read_phone_tilt())
    print("Current acceleration: ", rob.read_accel())
    print("Current orientation: ", rob.read_orientation())\
    
def take_picture(rob: IRobobo):

    time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    image = rob.read_image_front()
    cv2.imwrite(str(FIGURES_DIR / f"photo_{time}.png"), image)

def center_camera(rob: IRobobo):
    rob.set_phone_pan_blocking(123, 100)
    take_picture(rob)

def set_tilt(rob: IRobobo, tilt: int):
    rob.set_phone_tilt_blocking(tilt, 100)
    take_picture(rob)

def test_phone_movement(rob: IRobobo):
    rob.set_phone_pan_blocking(20, 100)
    print("Phone pan after move to 20: ", rob.read_phone_pan())
    rob.set_phone_tilt_blocking(50, 100)
    print("Phone tilt after move to 50: ", rob.read_phone_tilt())


def test_sim(rob: SimulationRobobo):
    print("Current simulation time:", rob.get_sim_time())
    print("Is the simulation currently running? ", rob.is_running())
    rob.stop_simulation()
    print("Simulation time after stopping:", rob.get_sim_time())
    print("Is the simulation running after shutting down? ", rob.is_running())
    rob.play_simulation()
    print("Simulation time after starting again: ", rob.get_sim_time())
    print("Current robot position: ", rob.get_position())
    print("Current robot orientation: ", rob.get_orientation())

    pos = rob.get_position()
    orient = rob.get_orientation()
    rob.set_position(pos, orient)
    print("Position the same after setting to itself: ", pos == rob.get_position())
    print("Orient the same after setting to itself: ", orient == rob.get_orientation())


def test_hardware(rob: HardwareRobobo):
    print("Phone battery level: ", rob.phone_battery())
    print("Robot battery level: ", rob.robot_battery())


def run_all_actions(rob: IRobobo):
    # if isinstance(rob, SimulationRobobo):
    #     rob.play_simulation()
    # test_emotions(rob)
    # test_sensors(rob)
    # test_move_and_wheel_reset(rob)
    # if isinstance(rob, SimulationRobobo):
    #     test_sim(rob)

    # if isinstance(rob, HardwareRobobo):
    #     test_hardware(rob)

    # test_phone_movement(rob)
    # print("IRS data: ", rob.read_irs())

    # if isinstance(rob, SimulationRobobo):
    #     rob.stop_simulation()

    center_camera(rob)
    for pan in [0, 45, 90, 135, 180]:
        rob.set_phone_pan_blocking(pan, 100)
    for tilt in [26, 50, 75, 105]:
        set_tilt(rob, tilt)
        take_picture(rob)

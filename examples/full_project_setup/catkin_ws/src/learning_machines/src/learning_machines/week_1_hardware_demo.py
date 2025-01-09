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

irs_positions = {
    "BackL": 0,
    "BackR": 1,
    "FrontL": 2,
    "FrontR": 3,
    "FrontC": 4,
    "FrontRR": 5,
    "BackC": 6,
    "FrontLL": 7,
}

def stop_at_obstacle(rob: IRobobo, sensor_id: str):
    sensor_index = irs_positions[sensor_id]
    print('waiting for obstacle...')
    while True:
        irs = rob.read_irs()
        print(irs[sensor_index])
        if irs[sensor_index] > 200:
            print('Obstacle detected!')
            break
        if sensor_id == 'FrontC':
            rob.move_blocking(10, 10, 500)
        else:
            rob.move_blocking(-10, -10, 500)


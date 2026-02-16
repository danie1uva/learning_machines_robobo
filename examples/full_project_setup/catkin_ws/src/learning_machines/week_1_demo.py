import cv2

from data_files import FIGURES_DIR, READINGS_DIR
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

def test_sensors(rob: IRobobo):
    print("IRS data: ", rob.read_irs())
    image = rob.read_image_front()
    cv2.imwrite(str(FIGURES_DIR / "photo.png"), image)
    print("Phone pan: ", rob.read_phone_pan())
    print("Phone tilt: ", rob.read_phone_tilt())
    print("Current acceleration: ", rob.read_accel())
    print("Current orientation: ", rob.read_orientation())

def stop_at_obstacle(rob: IRobobo, sensor_id: str):
    sensor_index = irs_positions[sensor_id]
    sensor_readings = []
    if isinstance(rob, HardwareRobobo):
        print('waiting for obstacle...')
        while True:
            irs = rob.read_irs()
            sensor_readings.append(irs) 
            print(irs[sensor_index])
            if irs[sensor_index] > 200:
                print('Obstacle detected!')
                file_path = str(READINGS_DIR / f"{sensor_id}_readings_hardware.txt")
                with open(file_path, "w") as f:
                    for reading in sensor_readings:
                        f.write(f"{reading}\n")
                break
            if sensor_id == 'FrontC':
                rob.move_blocking(10, 10, 500)
            else:
                rob.move_blocking(-10, -10, 500)
    
    if isinstance(rob, SimulationRobobo):
        print('waiting for obstacle...')
        rob.play_simulation() #this makes the simulation run
        while True:
            irs = rob.read_irs()
            sensor_readings.append(irs)
            print(irs[sensor_index])
            if irs[sensor_index] > 200:
                print('Obstacle detected!')
                file_path = str(READINGS_DIR / f"{sensor_id}_readings_simulation.txt")
                with open(file_path, "w") as f:
                    for reading in sensor_readings:
                        f.write(f"{reading}\n")
                if sensor_id == 'BackC':        
                    rob.stop_simulation()
                break
            if sensor_id == 'FrontC':
                rob.move_blocking(10, 10, 500)
            else:
                rob.move_blocking(-10, -10, 500)
        
        


import cv2 
from datetime import datetime
from data_files import FIGURES_DIR
from robobo_interface import IRobobo

def take_picture(rob: IRobobo):
    time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    image = rob.read_image_front()
    cv2.imwrite(str(FIGURES_DIR / f"photo_{time}.png"), image)

def center_camera(rob: IRobobo):
    rob.set_phone_pan_blocking(123, 100)
    take_picture(rob)

def set_pan(rob: IRobobo, pan: int):
    rob.set_phone_pan_blocking(pan, 100)
    take_picture(rob)

def set_tilt(rob: IRobobo, tilt: int):
    rob.set_phone_tilt_blocking(tilt, 100)
    take_picture(rob)

def pivot(rob):
    rob.move_blocking(100, 0, 100)

def check_centering(list_of_coords):
    
    if list_of_coords[0] > 0.4 and list_of_coords[0] < 0.6:
        return True
    else:
        return False

def drive_straight(rob):
    sensors = rob.read_irs()
    while max(sensors) < 800:
        rob.move_blocking(100, 100, 1000)
    
def detect_box(rob):    

    is_box_ahead = False 
    while not is_box_ahead:
        pivot(rob)
        image = take_picture(rob) # stored in results/figures 
        list_of_coords = process_image(image) # outputs coordinates of box in image 
        check_centering(list_of_coords)
        if check_centering == True:
            drive_straight
    

def forage(rob):
    set_pan(rob, 123) # centers the camera
    set_tilt(rob, 75) # tilts the camera down
    
    while True:
        detect_box(rob)




import sqlite3
import os
from database import add_to_database


if __name__ == "__main__":
    url = "https://tdcctv.data.one.gov.hk/id.JPG"
    folder_name = "Camera_Images"
    video = ""
    model_id = "yolov10m"
    image_size = 1280
    conf_threshold = 0.5
    input_type = 'Image'

    with open("CameraIds.txt") as file:
        id_list = [line.strip() for line in file]

    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
    db_name = "camera_data.db"
   
    vehicle_weights = 1, 2, 2.5
    add_to_database(id_list = id_list, iterations = 100, interval = 15, weights = vehicle_weights, folder_name = "Camera_Images", db_name = db_name)






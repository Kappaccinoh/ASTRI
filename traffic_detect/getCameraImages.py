import os
import time
import requests
from datetime import datetime
import cv2
from yolov10.detect import run_inference
import json
import sqlite3

url = "https://tdcctv.data.one.gov.hk/id.JPG"
folder_name = "Camera_Images"
video = ""
model_id = "yolov10m"
image_size = 1280
conf_threshold = 0.5
input_type = 'Image'

with open("CameraList.txt") as file:
    id_list = [line.strip() for line in file]

if not os.path.exists(folder_name):
    os.makedirs(folder_name)

def count_image(url):
    response = requests.get(url)
    if response.status_code == 200:
        summary = run_inference(response.content, "", "yolov10m", 320, 0.5, 'Image')
        car, truck, bus = count_objects(summary)
        return car, truck, bus
    else:
        print(f"Failed to load image. Status code: {response.status_code}")

def convert_to_greyscale(image_path):
    image = cv2.imread(image_path)
    grey_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    cv2.imwrite(image_path, grey_image)

def count_objects(detected_objects):
    car_count = 0
    truck_count = 0
    bus_count = 0

    for obj in detected_objects:
        if obj['name'] == 'car':
            car_count += 1
        elif obj['name'] == 'truck':
            truck_count += 1
        elif obj['name'] == 'bus':
            bus_count += 1

    return car_count, truck_count, bus_count

conn = sqlite3.connect("camera_data1.db")
cursor = conn.cursor()
cursor.execute("""
CREATE TABLE IF NOT EXISTS camera_data (
    camera_id INTEGER,
    car INTEGER,
    truck INTEGER,
    bus INTEGER,
    timestamp DATETIME,
    weightedTrafficDensity FLOAT,
    relativeTrafficDensity FLOAT
)
""")

conn_mtd = sqlite3.connect("camera_mtd.db")
cursor_mtd = conn_mtd.cursor()
cursor_mtd.execute("""
CREATE TABLE IF NOT EXISTS camera_mtd (
    camera_id INTEGER,
    maxTrafficDensity FLOAT
)
""")

query_max = """
SELECT camera_id,
       MAX(weightedTrafficDensity) AS max_traffic_density
FROM camera_data
GROUP BY camera_id
"""

def add_to_database(id_list: list, duration: int, interval: int, weights):
    for i in range(duration):
        timestamp = datetime.now().strftime("%m_%d_%H%M")
        for camera_id in id_list:
            file_name = f"{camera_id}.jpg"
            car, truck, bus = count_image(url.replace("id", camera_id), folder_name, file_name)
            car_weight, truck_weight, bus_weight = weights
            weighted_average = (car * car_weight + truck * truck_weight + bus * bus_weight)
            cursor.execute("""
            INSERT INTO camera_data (camera_id, car, truck, bus, timestamp, weightedTrafficDensity, relativeTrafficDensity)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (camera_id, car, truck, bus, timestamp, weighted_average, 0.0))
            conn.commit()

            # Get the Max WeightedTrafficDensities per Traffic Camera
            mtd = cursor.execute(query_max)
            cursor_mtd.execute("""
            REPLACE INTO camera_mtd (camera_id, maxTrafficDensity)
            VALUES (?, ?)
            """, (camera_id, mtd))
            
            # Calculate the Relative Traffic Density 
            relative_traffic_density = weighted_average / mtd
            cursor_mtd.execute("""
            INSERT INTO camera_data (camera_id, weightedTrafficDensity)
            VALUES (?, ?)
            """, (camera_id, relative_traffic_density))
            
        time.sleep(interval * 60) 
    conn.close()


# add_to_database
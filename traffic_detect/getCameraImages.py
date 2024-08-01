import os
import time
import requests
from datetime import datetime
import cv2

url = "https://tdcctv.data.one.gov.hk/AID04105.JPG"
folder_name = "Camera_Images"

with open("CameraList.txt") as file:
    id_list = [line.strip() for line in file]

if not os.path.exists(folder_name):
    os.makedirs(folder_name)

def download_image(url, folder_name, file_name):
    response = requests.get(url)
    if response.status_code == 200:
        with open(os.path.join(folder_name, file_name), 'wb') as f:
            f.write(response.content)
        print(f"Image saved as {os.path.join(folder_name, file_name)}")
    else:
        print(f"Failed to download image. Status code: {response.status_code}")

def convert_to_greyscale(image_path):
    image = cv2.imread(image_path)
    grey_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    cv2.imwrite(image_path, grey_image)

for i in range(1):
    for camera_id in id_list:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_name = f"{camera_id}_{timestamp}.JPG"
        download_image(url.replace("AID04105", camera_id), folder_name, file_name)
        
        convert_to_greyscale(os.path.join(folder_name, file_name))

    # time.sleep(3 * 60) 

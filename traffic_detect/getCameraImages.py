import os
import time
import requests
from datetime import datetime

url = "https://tdcctv.data.one.gov.hk/AID04105.JPG"
folder_name = "Camera_Images"

# Read the list of IDs from the file
with open("CameraList.txt") as file:
    id_list = [line.strip() for line in file]

# Create the folder if it doesn't exist
if not os.path.exists(folder_name):
    os.makedirs(folder_name)

# Define a function to download and save the image with a unique name
def download_image(url, folder_name, file_name):
    response = requests.get(url)
    if response.status_code == 200:
        with open(os.path.join(folder_name, file_name), 'wb') as f:
            f.write(response.content)
        print(f"Image saved as {os.path.join(folder_name, file_name)}")
    else:
        print(f"Failed to download image. Status code: {response.status_code}")

# Download the image every 3 minutes for 10 times
for i in range(10):
    for camera_id in id_list:
        # Create a unique filename using the camera ID and current timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_name = f"{camera_id}_{timestamp}.JPG"
        download_image(url.replace("AID04105", camera_id), folder_name, file_name)
    time.sleep(3 * 60)  # Sleep for 3 minutes

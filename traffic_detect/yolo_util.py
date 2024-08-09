import os
import requests
import cv2
from yolov10.detect import run_inference


def download_image(url, folder_name, filename):
    # Ensure the folder exists
    os.makedirs(folder_name, exist_ok=True)
    filepath = os.path.join(folder_name, filename)
    
    # Attempt to download the image
    response = requests.get(url)
    if response.status_code == 200:
        with open(filepath, 'wb') as f:
            f.write(response.content)
        # print(f"Image successfully downloaded as {filepath}")
        return filepath
    else:
        print(f"Failed to load image. Status code: {response.status_code}")
        return None

def count_image(url, folder_name, filename):
    filepath = download_image(url, folder_name, filename)
    
    if filepath:
        summary = run_inference(filepath, "", "yolov10m", 320, 0.5, 'Image')
        if summary:
            print(summary)
            car, truck, bus = count_objects(summary)
            return car, truck, bus
        else:
            return "N/A", "N/A", "N/A"
    else:
        print("Counting failed due to download failure")


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





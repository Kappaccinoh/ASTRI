import cv2
import numpy as np
import math

def rgb_to_hsv(r, g, b):
    color = np.uint8([[[b, g, r]]])
    hsv_color = cv2.cvtColor(color, cv2.COLOR_BGR2HSV)
    return hsv_color[0][0]

rgb_green = (0, 255, 0)
hsv_green = rgb_to_hsv(*rgb_green)
lower_green = np.array([hsv_green[0] - 10, 100, 100])
upper_green = np.array([hsv_green[0] + 10, 255, 255])

rgb_red = (30, 255, 255)
hsv_red = rgb_to_hsv(*rgb_red)
lower_red = np.array([hsv_red[0] - 10, 100, 100])
upper_red = np.array([hsv_red[0] + 10, 255, 255])


url = "Camera_Images_Label2/AID03101_20240801_111812_colour_mask.JPG"

def measure_line_length(image_path, lower_color, upper_color):
    image = cv2.imread(image_path)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    mask = cv2.inRange(hsv, lower_color, upper_color)
    image_url = url.replace("colour", "green")
    cv2.imwrite(image_url, mask)
    
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return 0
    
    largest_contour = max(contours, key=cv2.contourArea)
    length = cv2.arcLength(largest_contour, True)
    
    return length

image_path = "Camera_Images_Label2/AID03101_20240801_111812.JPG"
green_line_length = measure_line_length(image_path, lower_green, upper_green)

print(f"Green line length: {green_line_length} pixels")


import cv2

def measure_red_line_lengths(image_path, lower_color, upper_color):
    image = cv2.imread(image_path)
    
    if image is None:
        print(f"Failed to load image: {image_path}")
        return [], []
    
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower_color, upper_color)
    
    cv2.imwrite("masked_image.png", mask)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    lengths = [cv2.arcLength(contour, True) for contour in contours]
    start_points = [contour[0][0] for contour in contours]

    return lengths, start_points


red_line_lengths = measure_red_line_lengths(image_path, lower_red, upper_red)
print(f"Start of contour XY Coordinates with TOP LEFT as (0,0): {red_line_lengths[1]}")
print(f"Red line lengths: {red_line_lengths[0]} pixels")


# Assume 4 metre lines

# def getRoadLength(target_line, ref_list):
#     grad = 0
#     ref_list = [l / 4 for l in ref_list]
#     for i in range(len(ref_list)):
#         temp = ref_list[i]
#         print(temp)
#         grad += temp / 2
#     grad = grad / (len(ref_list)-1)
#     print(grad)
#     total_length = target_line / grad
#     return total_length


# road_length = getRoadLength(green_line_length, red_line_lengths[0])
# print(f"Road is: {road_length}m")


def linear_extrapolation(Y_coords, Y_grads, Y_max):
    m = 0
    for i in range(len(Y_coords)-1):
        Y1, m1 = Y_coords[i], Y_grads[i]
        Y2, m2 = Y_coords[i+1], Y_grads[i+1]
        m += (m1-m2) / (Y1 - Y2)
        print(m)
    m = m / (len(Y_coords) -1)    
    grad_Y_max = Y_grads[0] + m * (Y_max - Y_coords[0])
    
    return grad_Y_max


# def getRoadLength(target_line, ref_list):
#     grad = 0
#     ref_list = [l / 4 for l in ref_list]
#     for i in range(len(ref_list)):
#         temp = ref_list[i]
#         print(temp)
#         grad += temp / 2
#     grad = grad / (len(ref_list)-1)
#     print(grad)
#     total_length = target_line / grad
    
#     return length



Y_max = 240  

Y_grads = [l/4 for l in red_line_lengths[0]]
print(Y_grads)
Y_coords = [y for _, y in red_line_lengths[1]]
print(Y_coords)
grad_Y_max = linear_extrapolation(Y_coords, Y_grads, Y_max)
print(f"Gradient at Y_max: {grad_Y_max:.2f}")

def length_from_grad(grad, Y_max):
    res = 0
    dec = grad/Y_max
    pixel_metre_ratio = grad
    for _ in range(Y_max):
        res += 1/pixel_metre_ratio
        pixel_metre_ratio -= dec
    return res

road_len = length_from_grad(grad_Y_max, Y_max)
print(f"Road has length: {road_len:.2f}m")


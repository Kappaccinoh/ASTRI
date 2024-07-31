import cv2
import numpy as np

# Load the image
image = cv2.imread('Camera_Images/AID03101_20240731_135511.JPG')

# Convert to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply edge detection
edges = cv2.Canny(gray, 50, 150, apertureSize=3)

# Find contours
contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

# Assume the largest contour is the road
road_contour = max(contours, key=cv2.contourArea)

# Calculate the length of the road contour
road_length = cv2.arcLength(road_contour, True)

print(f"Estimated road length: {road_length} pixels")

# Display the image with the detected road
cv2.drawContours(image, [road_contour], -1, (0, 255, 0), 2)
output_filename = 'road_with_contour.jpg'
cv2.imwrite(output_filename, image)

print(f"Image saved as {output_filename}")
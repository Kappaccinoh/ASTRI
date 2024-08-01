import cv2
import numpy as np

# Load the image
image = cv2.imread('Camera_Images/AID03101_20240731_135511.JPG')

# Convert to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply Gaussian blur to reduce noise
blurred = cv2.GaussianBlur(gray, (5, 5), 0)

# Apply edge detection
edges = cv2.Canny(blurred, 30, 100, apertureSize=3)

# Apply morphological operations to close gaps in edges
kernel = np.ones((5, 5), np.uint8)
closed_edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

# Find contours
contours, _ = cv2.findContours(closed_edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

# Filter contours by area (assuming road is one of the largest contours)
min_area = 1000  # Adjust this threshold based on your image
filtered_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_area]

# Assume the largest filtered contour is the road
road_contour = max(filtered_contours, key=cv2.contourArea)

# Calculate the length of the road contour
road_length = cv2.arcLength(road_contour, True)

# Draw the road contour
cv2.drawContours(image, [road_contour], -1, (0, 255, 0), 2)

# Draw a line along the road contour
for i in range(len(road_contour) - 1):
    cv2.line(image, tuple(road_contour[i][0]), tuple(road_contour[i + 1][0]), (255, 0, 0), 2)

# Display the road length on the image
cv2.putText(image, f"Road length: {road_length:.2f} pixels", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

# Save the output image
output_filename = 'road_with_length.jpg'
cv2.imwrite(output_filename, image)

print(f"Estimated road length: {road_length:.2f} pixels")
print(f"Image saved as {output_filename}")

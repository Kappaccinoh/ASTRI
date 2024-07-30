import xml.etree.ElementTree as ET
import json

# Function to parse XML and extract easting and northing for given keys
def extract_coordinates(xml_file, keys_file, output_file):
    tree = ET.parse(xml_file)
    root = tree.getroot()

    # Read keys from the text file
    keys = []
    with open(keys_file, 'r') as f:
        print(f)
        for line in f:
            keys.append(line.strip())
        print(keys)
    # Dictionary to store the results
    results = {}

    # Iterate through each image element in the XML
    for image in root.findall('image'):
        key = image.find('key').text
        if key in keys:
            easting = image.find('easting').text
            northing = image.find('northing').text
            results[key] = {'easting': easting, 'northing': northing}

    # Write the results to a JSON file
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=4)

# Example usage
xml_file = 'Traffic_Camera_Locations_En.xml'
keys_file = 'CameraList.txt'
output_file = 'output.json'
extract_coordinates(xml_file, keys_file, output_file)
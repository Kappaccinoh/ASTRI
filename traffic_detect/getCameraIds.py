import xml.etree.ElementTree as ET

xml_file = 'Traffic_Camera_Locations_En.xml'
keys_file = 'CameraIds.txt'

tree = ET.parse(xml_file)
root = tree.getroot()

keys = [element.text for element in root.findall('.//key')]

with open(keys_file, 'w') as f:
    for key in keys:
        f.write(key + '\n')


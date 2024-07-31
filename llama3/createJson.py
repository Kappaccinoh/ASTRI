import csv
import json


long_lat_key = {}

with open('CameraLocations.csv', 'r')as file:
  csvFile = csv.DictReader(file, delimiter='\t')
  data = list(csvFile)
  print(data)
  for row in data:
    long_lat_key[str((row['latitude'], row['longitude']))] = row['key']

  print(long_lat_key)


with open('locations.json', 'w+') as output_file:
  json.dump(long_lat_key, output_file)
  
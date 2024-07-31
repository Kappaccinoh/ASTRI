import numpy as np
import requests
from scipy.spatial import cKDTree, KDTree
import json
import math

def getStreetNodes(street1, street2):
    if len(street1) == 0:
        return (1,0)
    elif len(street2) == 0:
        return (0,1)

    # Construct the Overpass API query for the first street
    query1 = f"""
    [out:json];
    area["name:en"="Hong Kong"]['admin_level'='2']->.country;
    area["name:en"="Hong Kong"](area.country)->.city;
    way["name:en"="{street1}"](area.city);
    out body;
    relation["name:en"="{street1}"](area.city);
    out body;
    node["name:en"="{street1}"](area.city);
    out body;
    way["name:zh"="{street1}"](area.city);
    out body;
    relation["name:zh"="{street1}"](area.city);
    out body;
    node["name:zh"="{street1}"](area.city);
    out body;
    """

    # Construct the Overpass API query for the second street
    query2 = f"""
    [out:json];
    area["name:en"="Hong Kong"]['admin_level'='2']->.country;
    area["name:en"="Hong Kong"](area.country)->.city;
    way["name:en"="{street2}"](area.city);
    out body;
    relation["name:en"="{street2}"](area.city);
    out body;
    node["name:en"="{street2}"](area.city);
    out body;
    way["name:zh"="{street2}"](area.city);
    out body;
    relation["name:zh"="{street2}"](area.city);
    out body;
    node["name:zh"="{street2}"](area.city);
    out body;
    """

    # Send the requests to the Overpass API
    response1 = requests.post("https://overpass-api.de/api/interpreter",
                              data={"data": query1})
    response2 = requests.post("https://overpass-api.de/api/interpreter",
                              data={"data": query2})

    # Parse the responses
    print("Responses collected, jsonifying (rtree method)")
    data1 = response1.json()
    data2 = response2.json()

    # Extract the node information for each street
    nodes1 = []
    for element in data1['elements']:
        if element['type'] == 'node':
            nodes1.append(element['id'])
        try:
            for node in element['nodes']:
                nodes1.append(node)
        except:
            continue

    # Extract the node information for the second street
    nodes2 = []
    for element in data2['elements']:
        if element['type'] == 'node':
            nodes2.append(element['id'])
        try:
            for node in element['nodes']:
                nodes2.append(node)
        except:
            continue

    nodes1 = list(set(nodes1))
    nodes2 = list(set(nodes2))

    if len(nodes1) == 0:
        return getStreetNodes(street1[1:], street2)
    elif len(nodes2) == 0:
        return getStreetNodes(street1, street2[1:])
    elif set(nodes1) == set(nodes2):
        print("SAME SET")
        return (0,0)
    
    return (nodes1, nodes2)



def approx_coordinates(streetNodes):
    print("No junction found, calculating nearest approximate coordinates instead")

    nodes_of_street1 = streetNodes[0]
    nodes_of_street2 = streetNodes[1]
    # Initialize variables to keep track of the closest nodes and their distances
    closest_node1 = None
    closest_node2 = None
    closest_distance = float('inf')

    num = 0
    street1 = len(nodes_of_street1) + len(nodes_of_street2)
    node1_coords = []
    for node in nodes_of_street1:
        if num % 10 == 0:
            print(f"loop1 {num}/{street1}")

        query3 = f"""
        [out:json];
        node({node});
        out body;
        relation(bn);
        out;
        """
        r = requests.get("https://overpass-api.de/api/interpreter",
             params={'data': query3})
        r = r.json()
        
        lat = r['elements'][0]['lat']
        lon = r['elements'][0]['lon']
        node1_coords.append((lat,lon))

        num += 1
    
    node2_coords = []
    for node in nodes_of_street2:
        if num % 10 == 0:
            print(f"loop2 {num}/{street1}")
        
        query3 = f"""
        [out:json];
        node({node});
        out body;
        relation(bn);
        out;
        """
        r = requests.get("https://overpass-api.de/api/interpreter",
             params={'data': query3})
        r = r.json()

        lat = r['elements'][0]['lat']
        lon = r['elements'][0]['lon']
        node2_coords.append((lat,lon))

        num += 1

    if (node1_coords == [] or node2_coords == []):
        print("Addresses don't exist on Open Street Map")
        return (0,0)
        

    # use KD Tree Method
    # Create KDTree for each set of coordinates
    tree1 = cKDTree(node1_coords)
    tree2 = cKDTree(node2_coords)

    # Find the closest pair of coordinates
    distances, closest_pairs = tree1.query(node2_coords, k=1)

    # Find the index of the closest pair
    closest_index = np.argmin(distances)

    # Get the closest pair of coordinates
    closest_pair_coords1 = node1_coords[closest_pairs[closest_index]]
    closest_pair_coords2 = node2_coords[closest_index]

    print(f"coord of address 1 - {closest_pair_coords1}")
    print(f"coord of address 2 - {closest_pair_coords2}")

    # Calculate the average of the closest pair
    ave_lat_coords = (closest_pair_coords1[0] + closest_pair_coords2[0]) / 2
    ave_lon_coords = (closest_pair_coords1[1] + closest_pair_coords2[1]) / 2

    ave_coords = (ave_lat_coords, ave_lon_coords)

    # Print the average values
    print(f"Average of the closest pair of coordinates: {ave_coords}")
    
    return ave_coords

def junction_coords(streetNodes, street1, street2):

    nodes1 = streetNodes[0]
    nodes2 = streetNodes[1]
    
    # Choose the shorter list for faster access time and performance
    shortNode = nodes2
    if len(nodes1) < len(nodes2):
        shortNode = nodes1

    found_coordinates = (0, 0)

    curr = 0
    total = len(shortNode)
    for node in shortNode:
        if (curr % 10) == 0:
            print(f"looping through list of nodes {curr}/{total}")

        r = requests.get(f'https://www.openstreetmap.org/node/{node}')
        r_data = r.text
        if street1 in r_data and street2 in r_data:

            query3 = f"""
            [out:json];
            node({node});
            out body;
            relation(bn);
            out;
            """

            response3 = requests.get("https://overpass-api.de/api/interpreter",
                                     params={'data': query3})

            res3 = response3.json()
            print(res3['elements'][0]['lat'])
            print(res3['elements'][0]['lon'])
            found_coordinates = (res3['elements'][0]['lat'],
                                 res3['elements'][0]['lon'])
            break

        curr = curr + 1

    return found_coordinates



def addressToGPS(street1, street2):
    # Search OpenMapsAPI for the existance of junctions between the two streets
    streetNodes = getStreetNodes(street1, street2)
    if streetNodes == (0,0) or streetNodes == (1,0) or streetNodes == (0,1):
        return streetNodes
    closest_coords = junction_coords(streetNodes, street1, street2)

    # If the two junctions dont exist, use the approximate method
    if closest_coords == (0,0):
        closest_coords = approx_coordinates(streetNodes)

    print(
        f"The coordinates for the point of intersection between {street1} and {street2} are:"
    )
    
    print(f"Lat,Lon = {closest_coords[0]}, {closest_coords[1]}")

    return closest_coords

# Find nearest camera location
def nearestCam(lookup):

    with open('locations.json', 'r') as file:
        hash_map = json.load(file)


    coordinates = [tuple(map(float, eval(coord))) for coord in hash_map.keys()]
    kdtree = KDTree(coordinates)


    # _, nearest_index = kdtree.query(lookup)
    # nearest_location = coordinates[nearest_index]
    # lat, lon = nearest_location
    # position = (str(lat), str(lon))
    # nearest_code = hash_map[str(position)]

    _, nearest_indices = kdtree.query(lookup, k=5)

    nearest_cameras = []
    for index in nearest_indices:
        nearest_location = coordinates[index]
        lat, lon = nearest_location
        position = str(lat), str(lon)
        nearest_code = hash_map[str(position)]
        nearest_cameras.append((nearest_code, nearest_location))
    print(nearest_cameras)
    for i, (code, location) in enumerate(nearest_cameras, start=1):
        print(f"{i}. Nearest camera {code} (at coordinates {location})")
    # print(f"Nearest location to {lookup} is camera {nearest_code} (at coordinates {nearest_location})")
    return nearest_location

    
# Calculates distance between lat, long coordinates in metres
def haversine(lat1, lon1, lat2, lon2):
    R = 6371000.0
    
    lat1_rad = math.radians(lat1)
    lon1_rad = math.radians(lon1)
    lat2_rad = math.radians(lat2)
    lon2_rad = math.radians(lon2)
    
    dlon = lon2_rad - lon1_rad
    dlat = lat2_rad - lat1_rad
    
    a = math.sin(dlat / 2)**2 + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon / 2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    distance = R * c
    
    return distance
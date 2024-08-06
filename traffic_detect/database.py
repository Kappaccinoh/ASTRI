import sqlite3
import pandas as pd
from datetime import datetime, timedelta

conn = sqlite3.connect("camera_data.db")
cursor = conn.cursor()



cursor.execute("""
CREATE TABLE IF NOT EXISTS camera_data (
    camera_id INTEGER,
    car INTEGER,
    truck INTEGER,
    bus INTEGER,
    timestamp DATETIME,
    weightedTrafficDensity FLOAT
)
""")


df = pd.read_csv("sorted_output.csv", delimiter=" ")
df.to_sql("camera_data", conn, if_exists="replace", index=False)

# Calculate the maximum traffic density per camera by querying all rows of a specific camera_id
query = """
SELECT camera_id,
       MAX(car) AS max_car,
       MAX(truck) AS max_truck,
       MAX(bus) AS max_bus
FROM camera_data
GROUP BY camera_id
"""

result = cursor.execute(query)

for row in result:
    camera_id, max_car, max_truck, max_bus = row
    total_vehicles = max_car + max_truck + max_bus
    max_traffic_density = (max_car * 1 + max_truck * 2 + max_bus * 2.5)
    print(f"Camera ID {camera_id}:")
    print(f"Car - Max: {max_car}")
    print(f"Truck - Max: {max_truck}")
    print(f"Bus - Max: {max_bus}")
    print(f"Max Traffic Density: {max_traffic_density:.2f}")
    print("-" * 40)

# Populate the Max Traffic Density Camera with the calculated value per camera
conn.cursor().execute("""
UPDATE camera_data
SET maxTrafficDensity = (
    SELECT (MAX(car) * 1 + MAX(truck) * 2 + MAX(bus) * 2.5)
    FROM camera_data
    WHERE camera_data.camera_id = camera_data.camera_id
)
""")

# Calculate for each row in the database, the relative traffic density against the populated max traffic density for each camera
query = """
SELECT camera_id,
       car,
       truck,
       bus,
       maxTrafficDensity
FROM camera_data
GROUP BY camera_id
"""

result = cursor.execute(query)

for row in result:
    camera_id, car, truck, bus, maxTrafficDensity = row
    total_vehicles = car + truck + bus
    relative_traffic_density = (car * 1 + truck * 2 + bus * 2.5) / maxTrafficDensity
    print(f"Camera ID {camera_id}:")
    print(f"Car: {car}")
    print(f"Truck: {truck}")
    print(f"Bus: {bus}")
    print(f"Max Traffic Density: {max_traffic_density:.2f}")
    print(f"Relative Traffic Density: {relative_traffic_density:.2f}")
    print("-" * 40)

# Populate the Relative Traffic Density Camera with the calculated value per camera
conn.cursor().execute("""
UPDATE camera_data
SET relativeTrafficDensity = (
    SELECT (car * 1 + truck * 2 + bus * 2.5) / (maxTrafficDensity)
    FROM camera_data
    WHERE camera_data.camera_id = camera_data.camera_id
)
""")

conn.commit()
conn.close()


# def remove_old_records(database):
#     """
#     Queries and removes records with timestamps more than 4 weeks before the current date.
#     """
#     conn = sqlite3.connect(database)
#     cursor = conn.cursor()

#     four_weeks_ago = datetime.now() - timedelta(weeks=4)
#     query = """
#     SELECT * FROM camera_data
#     WHERE timestamp < ?
#     """
#     cursor.execute(query, (four_weeks_ago,))
#     old_records = cursor.fetchall()

#     print("Old records (more than 4 weeks ago):")
#     for record in old_records:
#         print(record)

#     delete_query = """
#     DELETE FROM camera_data
#     WHERE timestamp < ?
#     """
#     cursor.execute(delete_query, (four_weeks_ago,))
#     conn.commit()
#     conn.close()

import pandas as pd
import uuid

# # Load the CSV file into a DataFrame
# df = pd.read_csv('Camera Min Max Vehicles.csv', header=None, sep=' ', names=['camera ID', 'number of cars', 'truck', 'bus', 'timestamp', 'test'])

# # Drop the 'test' column
# df = df.drop('test', axis=1)

# # Sort the DataFrame by 'camera ID' and 'timestamp'
# df = df.sort_values(['camera ID', 'timestamp'])

# # Print the sorted DataFrame
# print(df)

# # Save the sorted data to a new CSV file
# df.to_csv('sorted_output.csv', index=False, header=False, sep=' ')

df = pd.read_csv("sorted_output.csv", header=None, sep=' ', names=['camera_id', 'num cars', 'num trucks', 'num buses', 'timestamp'])

# Group the data by 'camera ID' and calculate the max and min values
grouped = df.groupby('camera_id')
min_max = grouped[['num cars', 'num trucks', 'num buses']].agg(['min', 'max'])

# Flatten the column names
min_max.columns = ['_'.join(col).strip() for col in min_max.columns]

# Merge the min/max values back into the original DataFrame
df = df.join(min_max, on='camera_id')

# Drop the original columns and rename the new columns
df = df.drop(['num cars', 'num trucks', 'num buses'], axis=1)
df = df.rename(columns={
    'number of cars_min': 'min_cars',
    'number of cars_max': 'max_cars',
    'truck_min': 'min_trucks',
    'truck_max': 'max_trucks',
    'bus_min': 'min_buses',
    'bus_max': 'max_buses'
})

print(df)

# Calculate the minimum and maximum total vehicles
df['min_total_vehicles'] = df[['num cars_min', 'num trucks_min', 'num buses_min']].sum(axis=1)
df['max_total_vehicles'] = df[['num cars_max', 'num trucks_max', 'num buses_max']].sum(axis=1)

# Drop duplicate 'camera ID' rows, keeping the first occurrence
df = df.drop_duplicates('camera_id', keep='first')

# Print the final DataFrame
print(df)


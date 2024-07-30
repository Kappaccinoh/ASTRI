import geopandas as gpd
import fiona
import geopandas as gpd
import matplotlib.pyplot as plt
import json




gdb_file = 'RdNet_IRNP.gdb'
# gdb_file = '11-SW-A.gdb'
# with fiona.open(gdb_file) as src:
#     print(src.meta)
# layers = fiona.listlayers(gdb_file, driver='FileGDB')

# # gdf = gpd.read_file('path/to/your/file.gml', driver='GML')
# gdf = gpd.read_file(gdb_file)
# gdf.plot()
# plt.show()
# # for layer in layers:
# #     gdf = gpd.read_file('RdNet_IRNP.gdb.zip', layer = layer)
# #     print(gdf)





layers = fiona.listlayers(gdb_file)
fig, ax = plt.subplots()
for layer_name in layers:
    print(layer_name)
    if layer_name == 'HydrographyPoly' or layer_name == 'ContourPoly':
        continue
    layer_name = gpd.read_file(gdb_file, layer=layer_name)
    layer_name.plot(ax=ax)
    


# gdf1 = gpd.read_file(gdb_file, layer='RoadPoly')
# gdf2 = gpd.read_file(gdb_file, layer='RoadLine')
# gdf1.plot(ax=ax, color='#E568F1', linewidth = 0.5, zorder=0)
# gdf2.plot(ax=ax, color='black', linewidth = 0.5, zorder= 5)

# with open('output.json', 'r') as f:
#     coordinates = json.load(f)

# eastings = [int(data["easting"]) for data in coordinates.values()]
# northings = [int(data["northing"]) for data in coordinates.values()]

# plt.scatter(eastings, northings, marker='.', color='green', s = 5, zorder=10)


plt.savefig('test_plot.png', dpi = 1200)




from addressToGPS import nearestCam, haversine
import fire

def distance_to_cam(lookup):
    # lookup = (22.255, 114.165)
    lat1, lon1 = lookup

    nearest_cam_pos = nearestCam(lookup)
    print(nearest_cam_pos)
    lat2, lon2 = nearest_cam_pos

    distance_m = haversine(lat1, lon1, lat2, lon2)

    print(f"The distance between camera and position is approximately {distance_m:.2f} metres.")


# if __name__ == "__main__":
#     fire.Fire(main)

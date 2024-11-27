import numpy as np

def euclidean_distance(point1, point2):
    return np.sqrt(np.sum((np.array(point1) - np.array(point2)) ** 2))

def calculate_player_centres(player_detections):
    center_coordinates = {}
    for player in player_detections.keys():
        x1, y1, x2, y2 = player_detections[player]
        cc = [x1+ ((x2-x1)/2), y1+ ((y2-y1)/2)]
        center_coordinates[player] = cc
        
    return center_coordinates

def calculate_min_distance(kps, player_center):
    d_min = 100000
    for i in range(0, len(kps), 2):
        pair = kps[i:i+2]  
        pair = [float(pair[0]), float(pair[1])]
        if d_min > euclidean_distance(pair, player_center):
            d_min = euclidean_distance(pair, player_center)
        
    return d_min


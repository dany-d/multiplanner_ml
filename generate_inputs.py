# The goal of this file is to take in an image (or multiple) and build a csv of all inputs for the neural net
import argparse
import glob
import os
from pandas import DataFrame
import pandas as pd
from map_runner import init_collision_checking, get_scrub_map, get_color_mask, collision_check
import math
import numpy as np

# all the features we will generate...
FEATURES = ["filename",
            "start_x", "start_y", "start_theta", "goal_x", "goal_y", "goal_theta", "dist_start_to_goal",
            "num_cars", "num_trucks", "num_bikes", "percent_empty", "percent_occupied",
            "X_intersection", "T_intersection", "C_intersection", "S_intersection", "R_intersection", "L_intersection", "U_intersection",
            "percent_straight_path_collision","dist_to_nearest_obs", "start_region_obs_density", "diff_theta"
            ]

# Color definitions
C_CAR =   [0  ,0  ,255]
C_BIKE =  [255,0  ,255]
C_TRUCK = [255,255,0  ]

C_WHITE = [255,255,255]
WHITE = 255


# Size definitions
S_CAR =   [21, 11]
S_BIKE =  [9, 7 ]
S_TRUCK = [41, 21]


# Area definitions
A_CAR = S_CAR[0] * S_CAR[1]
A_BIKE = S_BIKE[0] * S_BIKE[1]
A_TRUCK = S_TRUCK[0] * S_TRUCK[1]


# This will set all the start and goal info into a df at row_index
def set_start_goal_features(df, row_index, png_path):
    start_q, end_q = init_collision_checking(png_path)
    start_x, start_y, start_theta = start_q
    goal_x, goal_y, goal_theta = end_q
    dist_start_to_goal = math.sqrt(((start_x - goal_x) * (start_x - goal_x)) + ((start_y - goal_y) * (start_y - goal_y)))

    df.loc[row_index]["start_x", "start_y", "start_theta", "goal_x", "goal_y", "goal_theta", "dist_start_to_goal"] = [
        start_x, start_y, start_theta, goal_x, goal_y, goal_theta, dist_start_to_goal]


def straightpathcollision(df,row_index,png_path):
    start,end = init_collision_checking(png_path)
    theta = np.arctan2(end[1] - start[1], end[0] - start[0])
    delta = 1
    count = 0
    count_total = 0
    while math.sqrt(((start[0] - end[0]) ** 2 + (start[1] - end[1]) ** 2)) > 1:
        start = (start[0] + delta * np.cos(theta), start[1] + delta * np.sin(theta), start[2])
        count_total = count_total + 1
        if collision_check(start[0], start[1], start[2]):
            count = count + 1
    df.loc[row_index]["percent_straight_path_collision"] = count/count_total

def dist_to_nearest_obs(df,row_index,png_path):
    start, end = init_collision_checking(png_path)
    delta = 0
    x1,x2,y1,y2,d1,d2,d3,d4 = start,start,start,start,start,start,start,start
    while not (collision_check(x1[0],x1[1],x1[2]) or collision_check(x2[0],x2[1],x2[2]) or collision_check(y1[0],y1[1],y1[2]) or collision_check(y2[0],y2[1],y2[2])
        or collision_check(d1[0],d1[1],d1[2]) or collision_check(d2[0],d2[1],d2[2]) or collision_check(d3[0],d3[1],d3[2]) or collision_check(d4[0],d4[1],d4[2])):
        x1 = (start[0] + delta, start[1],start[2])
        x2 = (start[0] - delta, start[1],start[2])
        d1 = (start[0] + delta, start[1]+ delta,start[2])
        d2 = (start[0] + delta, start[1]- delta,start[2])
        d3 = (start[0] - delta, start[1]- delta,start[2])
        d4 = (start[0] - delta, start[1]+ delta,start[2])
        y1 = (start[0] , start[1]+ delta,start[2])
        y2 = (start[0] , start[1]- delta,start[2])
        delta = delta+1
    df.loc[row_index]["dist_to_nearest_obs"] = delta


def start_region_obs_density(df,row_index,png_path):
    start, end = init_collision_checking(png_path)
    delta = 11
    count = 0
    x1 = (start[0] + delta, start[1], start[2])
    x2 = (start[0] - delta, start[1], start[2])
    d1 = (start[0] + delta, start[1] + delta, start[2])
    d2 = (start[0] + delta, start[1] - delta, start[2])
    d3 = (start[0] - delta, start[1] - delta, start[2])
    d4 = (start[0] - delta, start[1] + delta, start[2])
    y1 = (start[0], start[1] + delta, start[2])
    y2 = (start[0], start[1] - delta, start[2])
    set = [x1,x2,d1,d2,d3,d4,y1,y2]
    count_total = 0
    for i in set:
        count_total = count_total + 1
        if collision_check(i[0],i[1],i[2]):
            count  = count + 1
    df.loc[row_index]["start_region_obs_density"] = count/count_total

def diff_theta(df,row_index,png_path):
    start, end = init_collision_checking(png_path)
    diff = end[2]-start[2]
    df.loc[row_index]["diff_theta"] = diff


def set_map_features(df, row_index):
    # get scrubed map (map without start and goal)
    map = get_scrub_map()
    map_shape = map.shape
    map_area = map_shape[0] * map_shape[1]

    car_mask = get_color_mask(C_CAR, map)
    car_pixel_count = np.sum(car_mask)
    num_cars = int(round(float(car_pixel_count)/float(A_CAR)))

    truck_mask = get_color_mask(C_TRUCK, map)
    truck_pixel_count = np.sum(truck_mask)
    num_trucks = int(round(float(truck_pixel_count)/float(A_TRUCK)))

    bike_mask = get_color_mask(C_BIKE, map)
    bike_pixel_count = np.sum(bike_mask)
    num_bikes = int(round(float(bike_pixel_count)/float(A_BIKE)))

    empty_mask = get_color_mask(C_WHITE, map)
    empty_pixel_count = np.sum(empty_mask)
    percent_empty = float(empty_pixel_count)/float(map_area)

    percent_occupied = 1.0 - percent_empty

    df.loc[row_index]["num_cars", "num_trucks", "num_bikes", "percent_empty", "percent_occupied"] = [
        num_cars, num_trucks, num_bikes, percent_empty, percent_occupied]


def set_map_type(df, row_index, png_path):
    filename = str(os.path.basename(png_path))

    map_type = filename.split("_")[1] # convention: MD_X_.png where X is type

    # start by setting all to false
    df.loc[row_index]["X_intersection", "T_intersection", "C_intersection", "S_intersection",
                      "R_intersection", "L_intersection", "U_intersection"] = [0,0,0,0,0,0,0]

    map_intersection = map_type + "_intersection"

    if map_intersection in FEATURES:
        df.loc[row_index][map_intersection] = 1
    else:
        print("ERROR improper intersection type in file: " + str(png_path) + "set all to false, continuing...")


# pass this function a directory containing png files
def gen_inputs_directory(png_directory):
    png_files = []
    for filename in glob.iglob(png_directory + '/**/*.png', recursive=True):
        png_files.append(os.path.abspath(filename))

    # init dataframe
    df = DataFrame(index=range(len(png_files)), columns= FEATURES)


    for row_index, png_path in enumerate(png_files):

        # set filename
        df.loc[row_index]["filename"] = os.path.basename(png_path)

        # set start and goal features
        set_start_goal_features(df, row_index, png_path)

        # set map features
        set_map_features(df, row_index)

        # set map type
        set_map_type(df, row_index, png_path)

        # percent_straight_path_collision and other inputs
        straightpathcollision(df,row_index,png_path)

        dist_to_nearest_obs(df, row_index, png_path)

        start_region_obs_density(df, row_index, png_path)

        diff_theta(df, row_index, png_path)



    # write the csv file
    df.to_csv(png_directory + "/inputs.csv")


if __name__== "__main__":
    # pd.DataFrame(np_array).to_csv("path/to/file.csv")
    parser = argparse.ArgumentParser(description='This takes directory with a map png in it, generates a csv for it'
                                                 'the inputs for the neural net and saves it in the same directory'
                                                 'Usage: $python generate_inputs.py "sample_maps"')
    parser.add_argument('png_data_directory', action='store', type=str,
                        help='path to the directory with png images (will work if directories are nested)')
    args = parser.parse_args()
    gen_inputs_directory(args.png_data_directory)

# This file will gennerate random maps for use
from map_runner import init_collision_checking_base_map, collision_check, \
                       view_path, get_collision_mask, get_raw_map, get_scrub_map, update_collision_mask, set_raw_map,\
                       set_raw_map_range, set_scrub_map, set_scrub_map_range,\
                       start_center, start_nose, start_fill, goal_center, goal_nose, goal_fill, \
                       change_collision_checker_vehcile_size, find_start_goal_scrub,\
                       set_raw_map_array, set_scrub_map_array, set_collision_mask_array
from generate_inputs import S_BIKE, S_CAR, C_BIKE, C_CAR
from generate_inputs import S_TRUCK as S_BUS
from generate_inputs import C_TRUCK as C_BUS
from A_star_steer_2 import run as Astar2Runner
from random import randint
import numpy as np
from math import sqrt
from matplotlib import pyplot as plt
from PIL import Image


MAX_BUS_COUNT = 1
MAX_MOPED_COUNT = 4
MAX_CAR_COUNT = 3

S_path = "base_maps/S.png"
X_path = "base_maps/X.png"
T_path = "base_maps/T.png"

S_count = 20
X_count = 20
T_count = 20

EMPTY_SCRUB_MAP = None
EMPTY_RAW_MAP = None
EMPTY_COLLISION_MASK = None

path_and_count = [(S_path, S_count), (X_path, X_count), (T_path, T_count)]

OPEN_POSITION_IDXS = []

save_count = 0

def save_map(basemap_path):
    global save_count
    map_type = "Q"
    if "S.png" in basemap_path:
        map_type = "S"
    if "X.png" in basemap_path:
        map_type = "X"
    if "T.png" in basemap_path:
        map_type = "T"

    img = Image.fromarray(np.flip(get_raw_map(), 1).swapaxes(0, 1), 'RGB')
    im_arr = np.flip(get_raw_map(), 1).swapaxes(0, 1)
    # plt.imshow(img)
    plt.imsave("autoMaps_steer/2/AG_"+map_type+"_"+str(save_count), im_arr)
    save_count += 1


def refresh_open_positions():
    global OPEN_POSITION_IDXS
    collision_mask = get_collision_mask()
    Xs, Ys = np.where(collision_mask == False)
    idxs = list(zip(Xs,Ys))
    OPEN_POSITION_IDXS = idxs
def check_dist(x0,y0,x1,y1):
    return sqrt(((x0-x1)**2) + ((y0-y1)**2))

def get_rand_open_position():
    rand_limit = len(OPEN_POSITION_IDXS) - 1
    rand_idx = randint(0, rand_limit)
    return OPEN_POSITION_IDXS[rand_idx]

def rand_theta():
    thetas = [0, np.pi/2, -np.pi/2, np.pi]
    return thetas[randint(0,3)]

def place_start_on_map(x,y,theta):
    nose_xy = (0,0)
    x_range = (0,0)
    y_range = (0,0)
    if theta == 0:
        nose_xy = (x+10,y)
        x_range = (x-10,x+10)
        y_range = (y-5,y+5)
    elif theta == np.pi/2:
        nose_xy = (x, y-10)
        x_range = (x-5,x+5)
        y_range = (y-10,y+10)
    elif theta == -np.pi/2:
        nose_xy = (x, y+10)
        x_range = (x-5,x+5)
        y_range = (y-10,y+10)
    elif theta == np.pi:
        nose_xy = (x-10, y)
        x_range = (x-10,x+10)
        y_range = (y-5,y+5)
    else:
        print("ERROR BAD THETA ON START CAR")

    # set the start fill
    set_raw_map_range(x_range, y_range, start_fill)
    # set the start nose
    set_raw_map(nose_xy[0], nose_xy[1], start_nose)
    # set the start center
    set_raw_map(x, y, start_center)

def place_goal_on_map(x,y,theta):
    nose_xy = (0,0)
    x_range = (0,0)
    y_range = (0,0)
    if theta == 0:
        nose_xy = (x+10,y)
        x_range = (x-10,x+10)
        y_range = (y-5,y+5)
    elif theta == np.pi/2:
        nose_xy = (x, y-10)
        x_range = (x-5,x+5)
        y_range = (y-10,y+10)
    elif theta == -np.pi/2:
        nose_xy = (x, y+10)
        x_range = (x-5,x+5)
        y_range = (y-10,y+10)
    elif theta == np.pi:
        nose_xy = (x-10, y)
        x_range = (x-10,x+10)
        y_range = (y-5,y+5)
    else:
        print("ERROR BAD THETA ON START CAR")

    # set the goal fill
    set_raw_map_range(x_range, y_range, goal_fill)
    # set the goal nose
    set_raw_map(nose_xy[0], nose_xy[1], goal_nose)
    # set the goal center
    set_raw_map(x, y, goal_center)

def place_bus_on_map(x,y,theta):
    x_range = (0,0)
    y_range = (0,0)
    x_half = int((S_BUS[0] - 1) / 2)
    y_half = int((S_BUS[1] - 1) / 2)
    if theta == 0 or theta == np.pi:
        x_range = (x-x_half,x+x_half)
        y_range = (y-y_half,y+y_half)
    elif theta == np.pi/2 or theta == -np.pi/2:
        x_range = (x-y_half,x+y_half)
        y_range = (y-x_half,y+x_half)
    else:
        print("ERROR BAD THETA ON BUS")

    # set the bus fill
    set_raw_map_range(x_range, y_range, C_BUS)
    #remember to do the scrub map as well
    set_scrub_map_range(x_range, y_range, C_BUS)

def place_moped_on_map(x,y,theta):
    x_range = (0,0)
    y_range = (0,0)
    x_half = int((S_BIKE[0] - 1) / 2)
    y_half = int((S_BIKE[1] - 1) / 2)
    if theta == 0 or theta == np.pi:
        x_range = (x-x_half,x+x_half)
        y_range = (y-y_half,y+y_half)
    elif theta == np.pi/2 or theta == -np.pi/2:
        x_range = (x-y_half,x+y_half)
        y_range = (y-x_half,y+x_half)
    else:
        print("ERROR BAD THETA ON MOPED")

    # set the moped fill
    set_raw_map_range(x_range, y_range, C_BIKE)
    #remember to do the scrub map as well
    set_scrub_map_range(x_range, y_range, C_BIKE)

def place_car_on_map(x,y,theta):
    x_range = (0,0)
    y_range = (0,0)
    x_half = int((S_CAR[0] - 1) / 2)
    y_half = int((S_CAR[1] - 1) / 2)
    if theta == 0 or theta == np.pi:
        x_range = (x-x_half,x+x_half)
        y_range = (y-y_half,y+y_half)
    elif theta == np.pi/2 or theta == -np.pi/2:
        x_range = (x-y_half,x+y_half)
        y_range = (y-x_half,y+x_half)
    else:
        print("ERROR BAD THETA ON CAR")

    # set the car fill
    set_raw_map_range(x_range, y_range, C_CAR)
    #remember to do the scrub map as well
    set_scrub_map_range(x_range, y_range, C_CAR)





def mapGen():
    for base_map_path, map_count in path_and_count:
        # load the base map
        init_collision_checking_base_map(base_map_path)

        # save off the empty map
        global EMPTY_COLLISION_MASK
        global EMPTY_RAW_MAP
        global EMPTY_SCRUB_MAP
        EMPTY_COLLISION_MASK = np.copy(get_collision_mask())
        EMPTY_RAW_MAP = np.copy(get_raw_map())
        EMPTY_SCRUB_MAP = np.copy(get_scrub_map())

        # refresh your open positions array
        refresh_open_positions()

        map_num = 0
        while map_num < map_count:
            #TODO: RESET THE MAPS HERE
            set_collision_mask_array(EMPTY_COLLISION_MASK)
            set_raw_map_array(EMPTY_RAW_MAP)
            set_scrub_map_array(EMPTY_SCRUB_MAP)


            #place your vehicles

            # place busses
            # change the collision checker size
            change_collision_checker_vehcile_size(S_BUS[0],S_BUS[1])
            bus_num = randint(0,MAX_BUS_COUNT)
            bus_count = 0
            while(bus_count < bus_num):
                # place a bus
                x,y = get_rand_open_position()
                theta = rand_theta()
                if not collision_check(x,y,theta):
                    # place this bus on the maps
                    place_bus_on_map(x,y,theta)
                    # update the collision checker
                    update_collision_mask()
                    refresh_open_positions()
                    bus_count += 1


            # place moped
            # change the collision checker size
            change_collision_checker_vehcile_size(S_BIKE[0],S_BIKE[1])
            moped_num = randint(0,MAX_MOPED_COUNT)
            moped_count = 0
            while(moped_count < moped_num):
                # place a moped
                x,y = get_rand_open_position()
                theta = rand_theta()
                if not collision_check(x,y,theta):
                    # place this bus on the maps
                    place_moped_on_map(x,y,theta)
                    # update the collision checker
                    update_collision_mask()
                    refresh_open_positions()
                    moped_count += 1


            # place cars
            # change the collision checker size
            change_collision_checker_vehcile_size(S_CAR[0],S_CAR[1])
            car_num = randint(0,MAX_CAR_COUNT)
            car_count = 0
            while(car_count < car_num):
                # place a car
                x,y = get_rand_open_position()
                theta = rand_theta()
                if not collision_check(x,y,theta):
                    # place this bus on the maps
                    place_car_on_map(x,y,theta)
                    # update the collision checker
                    update_collision_mask()
                    refresh_open_positions()
                    car_count += 1

            change_collision_checker_vehcile_size(21, 11)
            # place start and goal last
            # place start
            start_valid = False
            start_xy = (0,0)
            while not start_valid:
                # find an open position
                x,y = get_rand_open_position()
                # check if its valid
                theta = rand_theta()
                if not collision_check(x,y,theta):
                    start_valid = True
                    # place start
                    place_start_on_map(x,y,theta)
                    start_xy = (x,y)

            # place goal
            goal_valid = False
            while not goal_valid:
                # find an open position
                x,y = get_rand_open_position()
                # check if its valid
                theta = rand_theta()
                # remember to check that your goal is placed at least a certain distance from start center
                # THIS IS IMPORTANT your collision checker wont considder the goal car
                if ( (not collision_check(x,y,theta)) and (check_dist(x,y,start_xy[0],start_xy[1]) > 40)) :
                    goal_valid = True
                    # place goal
                    place_goal_on_map(x,y,theta)



            # remember to check if this config can complete
            # find out where the start and goal are
            try:
                (start_q, goal_q, _) = find_start_goal_scrub(get_raw_map())
            except:
                continue
            #check if path ==none or path is empty or path is only 2 or it straight fails
            path = None
            path_failed = False
            try:
                path = Astar2Runner(start_q, goal_q)
            except:
                pass
            if path is None or len(path) <= 2:
                path_failed = True

            if not path_failed:
                map_num += 1
                #remember to save this map if it works
                save_map(base_map_path)


if __name__== "__main__":
    mapGen()
import numpy as np
from PIL import Image
import math
from matplotlib import pyplot as plt
from skimage.draw import polygon,polygon_perimeter,line,bezier_curve
import time
from angle_calc import AngDiff
from matplotlib.colors import NoNorm

#global defines
start_center = [0  ,255,255]
start_nose =   [100,100,100]
start_fill =   [255,0  ,0  ]
goal_center =  [0  ,255,255]
goal_nose =    [50 ,50 ,50 ]
goal_fill =    [0  ,255,0  ]

path_start_color = [0  ,255,0  ]
path_color =       [150,150,150]
path_end_color =   [255,0  ,0  ]

WHITE = 255

CAR_X = 21 # car size in x
CAR_Y = 11 # car size in y

COLLISION_CAR_X = CAR_X
COLLISION_CAR_Y = CAR_Y

start_or_goal_longest_dimension = 21

#global variables
GLOBAL_COLLISION_MASK = None
GLOBAL_SCRUB_MAP = None # map with start and goal removed
GLOBAL_RAW_MAP = None
GLOBAL_MAP_PATH = None

GLOBAL_MAP_OF_COLLISION = None

TRAILER_SAVE_COLLISION_MASK = None

TRAILER_SAVE_COUNT = None
TRAILER_SAVE_MOD = 8

def dist2d(q0,q1):
    dist = math.sqrt((q0[0] - q1[0])**2 + (q0[1] - q1[1])**2 )
    return dist

# This will return a numpy array with the image data
def import_map(png_path):
    img = Image.open(png_path)
    img = img.convert("RGB")
    data = np.array(img, dtype='uint8')
    data = data[:,:,0:3]

    # We quietly make x rows and y cols to make the math easier to think about using the swapaxes(1,2)
    # This has consequences, now whenever we present visual data to the user we need to remeber to flip it back
    # so the photos are not transposed
    data = data.swapaxes(0,1)
    data = np.flip(data, 1) #flip y

    return data

# This will return a numpy array with the image data BLACK AND WHITE
def import_map_BW(png_path):
    img = Image.open(png_path)
    img = img.convert("RGB")
    data = np.array(img, dtype='uint8')
    data = data[:,:,0:3]

    # We quietly make x rows and y cols to make the math easier to think about using the swapaxes(1,2)
    # This has consequences, now whenever we present visual data to the user we need to remeber to flip it back
    # so the photos are not transposed
    data = data.swapaxes(0,1)
    data = np.flip(data, 1) #flip y

    return data


# This function searches outward from an xy point looking for a point that
# matches the color_match. (it makes a squares of the radius and searches that)
# This function will return the (x,y) position of the color match or None if not found
def search_outward(x,y,radius,color_match, map):
    x_map_max, y_map_max, _ = map.shape
    xmin = x-radius
    xmax = x+radius
    ymin = y-radius
    ymax = y+radius
    if xmin < 0: xmin = 0
    if xmax >= x_map_max: xmax = x_map_max-1
    if ymin < 0: ymin = 0
    if ymax >= y_map_max: ymax = y_map_max-1

    for i in range(xmin, xmax):
        for j in range(ymin, ymax):
            if (map[i,j,:] == color_match).all():
                return (i,j)

    return None


# This function will check a list of possible position coordinates to see if it is the start position
# if it is not it will return none, otherwise it will return the (x,y,theta) of the start position
# also returns the location of the start nose
def is_start(coords,raw_map):
    for (x,y) in coords:
        if (raw_map[x,y,:] == start_center).all():
            # this could be the center of the start or the goal lets check some neighbors
            if (raw_map[x+1,y,:] == start_fill).all():
                # we got it this is the start

                # Now lest find the nose of the start
                try:
                    (start_nose_x,start_nose_y) = search_outward(x,y,start_or_goal_longest_dimension,start_nose,raw_map)
                except TypeError as e:
                    import sys
                    raise type(e)(str(e) +
                      ' UNABLE TO FIND START IN MAP: %s MAKE SURE THE COLORS ARE RIGHT' % GLOBAL_MAP_PATH).with_traceback(sys.exc_info()[2])
                delta_x = start_nose_x - x
                delta_y = start_nose_y - y
                theta_radians = math.atan2(delta_y, delta_x)

                return (x,y,theta_radians),(start_nose_x, start_nose_y)
    return None

# This function will check a list of possible position coordinates to see if it is the goal position
# if it is not it will return none, otherwise it will return the (x,y,theta) of the goal position
# also returns the location of the goal nose
def is_goal(coords, raw_map):
    for (x, y) in coords:
        if (raw_map[x, y, :] == goal_center).all():
            # this could be the center of the goal or the goal lets check some neighbors
            if (raw_map[x + 1, y, :] == goal_fill).all():
                # we got it this is the goal

                # Now lest find the nose of the goal
                try:
                    (goal_nose_x, goal_nose_y) = search_outward(x, y, start_or_goal_longest_dimension, goal_nose, raw_map)
                except TypeError as e:
                    import sys
                    raise type(e)(str(e) +
                      ' UNABLE TO FIND GOAL IN MAP: %s MAKE SURE THE COLORS ARE RIGHT' % GLOBAL_MAP_PATH).with_traceback(sys.exc_info()[2])
                delta_x = goal_nose_x - x
                delta_y = goal_nose_y - y
                theta_radians = math.atan2(delta_y, delta_x)

                return (x, y, theta_radians),(goal_nose_x, goal_nose_y)
    return None

# takes in a map and returns a list of arrays which have the x,y locations of that color
# also returns a mask with 1s at that colors location
def get_color_loc(color, map):
    mask = (map[:,:,0] == color[0]) & (map[:,:,1] == color[1]) & (map[:,:,2] == color[2])
    coords_np = np.where(mask)
    (x_s, y_s) = coords_np
    coords = tuple(zip(x_s, y_s))
    return coords, mask


# just returns a mask for a certain color
def get_color_mask(color, map):
    mask = (map[:,:,0] == color[0]) & (map[:,:,1] == color[1]) & (map[:,:,2] == color[2])
    return mask


# This function will return the start configuration and the goal
# Configuration. It will also remove the start and goal configuration
# from the map and return the clean map.
# Returns: (start_x, start_y, start_theta), (start_x, start_y, start_theta), clean_map
def find_start_goal_scrub(raw_map):
    clean_map = np.copy(raw_map)
    row_max, col_max, _ = raw_map.shape
    start = None
    goal = None

    # start point center
    coords, mask_center = get_color_loc(start_center, raw_map)
    (start, start_nose_loc) = is_start(coords, raw_map)
    clean_map[start_nose_loc[0], start_nose_loc[1], :] = WHITE # make the nose white


    # goal point center
    coords, mask = get_color_loc(goal_center, raw_map)
    (goal, goal_nose_loc) = is_goal(coords, raw_map)
    clean_map[goal_nose_loc[0], goal_nose_loc[1], :] = WHITE # make the nose white
    clean_map[..., :][mask] = WHITE # make the goal center white
    clean_map[... , :][mask_center] = WHITE # make the start center white

    # clean the start and goal cars from the map
    coords, mask = get_color_loc(start_fill, raw_map)
    clean_map[..., :][mask] = WHITE  # make the start fill white

    coords, mask = get_color_loc(goal_fill, raw_map)
    clean_map[..., :][mask] = WHITE  # make the start fill white



    if (start == None) or (goal == None):
        print("ERROR: Unable to find start or goal!")
    return (start, goal, clean_map)

# provides a boolean mask matrix of places where the car would colide
def collision_mask_from_map(map):
    mask_inv = get_color_mask([WHITE,WHITE,WHITE],map)
    mask = np.invert(mask_inv)
    return mask

# show the map if save_path is not None then save to that location and do not show
def view_map(map, save_path=None):
    # The swapaxes in the next line is to make up for our x=row y=col convention
    img = Image.fromarray(np.flip(map, 1).swapaxes(0,1), 'RGB')
    plt.imshow(img)
    if save_path is None:
        plt.show()
        print("showing map...")
    else:
        plt.imsave(save_path, img)

# this function saves off a trailer path that was built using the functions given
def save_trailer_path(save_path):
    save_path = save_path.split('.png')[0]+'-TRAILER.png'
    img = np.array(Image.fromarray(np.flip(TRAILER_SAVE_COLLISION_MASK, 1).swapaxes(0,1)))
    plt.imshow(img)
    cmap = plt.cm.jet
    norm = plt.Normalize(vmin=img.min(), vmax=img.max())

    # map the normalized data to colors
    # image is now RGBA (512x512x4)
    image = cmap(norm(img))
    # cmap = plt.cm.viridis
    plt.imsave(save_path, image)


# show the map
def view_collision_mask(collision_mask):

    # The swapaxes in the next line is to make up for our x=row y=col convention
    plt.imshow(np.flip(collision_mask, 1).swapaxes(0,1))
    print("showing collision mask...")
    plt.show()


# show a path or save it to save_filepath if the argument is passed
def view_path(path, map = None, save_filepath = None, isTrailer=False):
    # go get the global path if you were not given one
    if map is None:
        map = GLOBAL_RAW_MAP
        if map is None:
            raise ValueError("map is None, Did you remember to call init_collision_checking()?")

    if save_filepath is None:
        for p in path:
            if isTrailer:
                collision_check(p[0], p[1], p[2], show_colison_bool=True, trailerTheta=p[3])
            else:
                collision_check(p[0], p[1], p[2], show_colison_bool=True)
    if save_filepath is not None and isTrailer:
        trailer_first_bool = True
        for p in path:
            collision_check(p[0], p[1], p[2], show_colison_bool=True, trailerTheta=p[3], trailer_save=True, trailer_save_first=trailer_first_bool)
            trailer_first_bool = False
        save_trailer_path(save_filepath)



    map_with_path = np.copy(map)
    for i in range(len(path)-1):
        p = path[i]
        px0=int(round(p[0]))
        py0=int(round(p[1]))
        pt0=p[2]
        p = path[i+1]
        px1=int(round(p[0]))
        py1=int(round(p[1]))
        pt1=p[2]

        # #draw line from center to center
        # rr, cc = line(px0, py0, px1, py1)
        # map_with_path[rr, cc, :] = path_color
        #
        # #draw line from top to top
        # rr, cc = line(px0, py0-1, px1, py1-1)
        # map_with_path[rr, cc, :] = path_color
        #
        # #draw line from bottom to bottom
        # rr, cc = line(px0, py0+1, px1, py1+1)
        # map_with_path[rr, cc, :] = path_color
        #
        # #draw line from left to left
        # rr, cc = line(px0-1, py0, px1-1, py1)
        # map_with_path[rr, cc, :] = path_color
        #
        # #draw line from right to right
        # rr, cc = line(px0+1, py0, px1+1, py1)
        # map_with_path[rr, cc, :] = path_color

        # find a center point ahead of the current bot: go direction of theta 0 for step size
        L = 10 #stepsize
        cx = int(round(L * math.cos(pt0)))+px0
        cy = int(round(L * math.sin(pt0)))+py0

        rr,cc = bezier_curve(px0,py0,cx,cy,px1,py1,1)
        map_with_path[rr, cc, :] = path_color


    #set start color
    px = int(round(path[0][0]))
    py = int(round(path[0][1]))
    map_with_path[px-1:px+2, py-1:py+2, :] = path_start_color
    #set end color
    px = int(round(path[-1][0]))
    py = int(round(path[-1][1]))

    map_with_path[px-1:px+2, py-1:py+2, :] = path_end_color

    view_map(map_with_path, save_filepath)

def change_collision_checker_vehcile_size(x,y):
    global COLLISION_CAR_X
    global COLLISION_CAR_Y

    COLLISION_CAR_X = x
    COLLISION_CAR_Y = y


# this is the collison checker for the trailer it is only called when there is no collision on the car
def collision_check_trailer(x_center, y_center, theta, show_colison_bool=False, collision_mask = None, trailerTheta = None, trailer_save=False, trailer_save_first=False):


    # we need to find the x_center and y_center and theta of the trailer
    # move the car to the back of the car then swivvle on nose
    x_offset_tail = (CAR_X/2)*math.cos(theta+np.pi)
    y_offset_tail = (CAR_X/2)*math.sin(theta+np.pi)

    #do this again to get to the middle of the trailer using trailer theta

    x_offset_tCenter = x_offset_tail + (CAR_X/2)*math.cos(trailerTheta+np.pi)
    y_offset_tCenter = y_offset_tail + (CAR_X/2)*math.sin(trailerTheta+np.pi)

    tr_x = x_center + x_offset_tCenter
    tr_y = y_center + y_offset_tCenter
    tr_theta = trailerTheta



    # # this is the offset to put the nose at the tail of the car
    # # now we spin the car on its nose
    # rotation_matrix = np.array([[math.cos(trailerTheta), -1*math.sin(trailerTheta)],
    #                             [math.sin(trailerTheta), math.cos(trailerTheta)]])
    # # apply the rotation to the trailer from zero theta
    # point_rel = np.array([[x_offset_nose], [y_offset_nose]])
    #
    # point_rel_rot = np.matmul(rotation_matrix, point_rel)
    # x_rel_rot = point_rel_rot[0, 0]
    # y_rel_rot = point_rel_rot[1, 0]
    #
    # tr_x = x_center + x_rel_rot + x_offset_nose
    # tr_y = y_center + y_rel_rot + y_offset_nose
    #
    # tr_theta = theta + trailerTheta

    return collision_check(tr_x,tr_y,tr_theta, collision_mask=collision_mask, show_colison_bool=show_colison_bool, isTrailer=True, trailer_save=trailer_save, trailer_save_first=trailer_save_first)









# This is the collision checker
# the show_colison_bool will show you a pic of the collision or non-collision
# if you have not run
def collision_check(x_center, y_center, theta, show_colison_bool=False, collision_mask = None, trailerTheta = None, isTrailer = False, trailer_save=False, trailer_save_first=False):

    if trailerTheta is not None:
        # check for jack-kniffing
        if abs(AngDiff(trailerTheta, theta)) > np.deg2rad(110):
            return True

    if collision_mask is None:
        collision_mask = GLOBAL_COLLISION_MASK
        if collision_mask is None:
            raise ValueError("collision_mask is None, Did you remember to call init_collision_checking()?")

    in_collision = False

    x_map_max, y_map_max = collision_mask.shape




    # start by placing the upper left pixel of the car
    start_x = int(x_center - ((COLLISION_CAR_X - 1) / 2)) #- 1
                                        # The plus one below is to accomidate the draw polygon function
    end_x =   int(x_center + ((COLLISION_CAR_X - 1) / 2)) #+ 1 # TODO: This may make the check at the end of the for loop invalid

    start_y = int(y_center - ((COLLISION_CAR_Y - 1) / 2)) #- 1
                                        # The plus one below is to accomidate the draw polygon function
    end_y   = int(y_center + ((COLLISION_CAR_Y - 1) / 2)) #+ 1 # TODO: This may make the check at the end of the for loop invalid

    if isTrailer:
        # start by placing the upper left pixel of the car
        start_x = int(x_center - ((COLLISION_CAR_X - 1) / 2))  # - 1
        # The plus one below is to accomidate the draw polygon function
        end_x = int(x_center)  # + 1 # TODO: This may make the check at the end of the for loop invalid

        start_y = int(y_center - ((COLLISION_CAR_Y - 1) / 2))  # - 1
        # The plus one below is to accomidate the draw polygon function
        end_y = int(y_center + ((COLLISION_CAR_Y - 1) / 2))  # + 1 # TODO: This may make the check at the end of the for loop invalid



    # we need to draw the shape of the car so we get the corners of the car

    x0y0 = np.array([[start_x],[start_y]])
    x0y1 = np.array([[start_x],[end_y]])
    x1y0 = np.array([[end_x],[start_y]])
    x1y1 = np.array([[end_x],[end_y]])

    # we need to transform these points by rotating about the center

    rotation_matrix = np.array([[math.cos(theta), -1*math.sin(theta)],
                                [math.sin(theta), math.cos(theta)]])

    rect_x = []
    rect_y = []

    for pnt in [x0y0, x0y1, x1y1, x1y0]:
        # pull out the points
        x = pnt[0,0]
        y = pnt[1,0]

        # position relative to the car center
        x_rel = float(x - x_center)
        y_rel = float(y - y_center)

        point_rel = np.array([[x_rel],[y_rel]])

        point_rel_rot = np.matmul(rotation_matrix, point_rel)
        x_rel_rot = point_rel_rot[0,0]
        y_rel_rot = point_rel_rot[1,0]

        # now we have rotated this point relative to the center of the car
        # time to translate it back to the point in space
        x_rot = x_rel_rot + x_center
        y_rot = y_rel_rot + y_center

        x_rot_round = round(x_rot)
        y_rot_round = round(y_rot)

        if ((x_rot_round < 0) or (x_rot_round >= x_map_max) or
            (y_rot_round < 0) or (y_rot_round >= y_map_max)):
            #print("Car Placed Outside map!")
            return True

        # now we know where this point will end up lets check the collision_matrix at this point
        if (collision_mask[int(x_rot_round),int(y_rot_round)] == True):
            # This just checks to see if the corners are hitting in which case it returns faster
            if not show_colison_bool:
                return True # normally it should exit here but if we want to see the car we need to continue
            else:
                in_collision = True

        rect_x.append(x_rot)
        rect_y.append(y_rot)

    # before we send the rect_x and rect_y values to the drawing function we need to take a look at them
    # what we are looking for is cases where the start_x and end_x or start_y and end_y have been swapped
    # this can happen when there was a full rotation of the car, this will result in an off-by-one drawing
    # x0y0, x0y1, x1y1, x1y0
    # lowest_x = 1000000
    # lowest_x_ind = 0
    # lowest_y = 1000000
    # lowest_y_ind = 0
    # for i,(x,y) in enumerate(zip(rect_x,rect_y)):
    #     if x < lowest_x:
    #         lowest_x = x
    #         lowest_x_ind = i
    #     if y < lowest_y:
    #         lowest_y = y
    #         lowest_y_ind = i
    # rect_x[lowest_x_ind] -= 1
    # rect_y[lowest_y_ind] -= 1


    # From: http://scikit-image.org/docs/dev/api/skimage.draw.html#skimage.draw.polygon
    rect_x_fill, rect_y_fill = polygon(np.array(rect_x), np.array(rect_y))
    rect_x_edge, rect_y_edge = polygon_perimeter(np.array(rect_x), np.array(rect_y))

    if isTrailer:
        #create the arm of the trailer as a line
        # we need to find where the nose of the trailer exists
        point_nose = np.array([[((CAR_X - 1) / 2)], [0.0]])
        point_nose_rot = np.matmul(rotation_matrix, point_nose)
        # map_of_collision[int(round(point_nose_rot[0, 0] + x_center)), int(round(point_nose_rot[1, 0] + y_center))] = 5
        trailer_arm_x, trailer_arm_y = line(int(round(x_center)),
                                            int(round(y_center)),
                                            int(round(point_nose_rot[0, 0] + x_center)),
                                            int(round(point_nose_rot[1, 0] + y_center)))
        rect_x_edge = np.concatenate((rect_x_edge,trailer_arm_x))
        rect_y_edge = np.concatenate((rect_y_edge,trailer_arm_y))




    if not show_colison_bool:
        if True in collision_mask[rect_x_edge, rect_y_edge]:
            return True
        if True in collision_mask[rect_x_fill, rect_y_fill]:
            return True
        if trailerTheta is not None:
            trailer_result = collision_check_trailer(x_center,y_center,theta, show_colison_bool=show_colison_bool,collision_mask=collision_mask,trailerTheta=trailerTheta)
            return trailer_result or False
        return False

        # for x_, y_ in zip(rect_x_edge, rect_y_edge):
        #     if collision_mask[x_,y_]:
        #         return True
        # for x_, y_ in zip(rect_x_fill, rect_y_fill):
        #     if collision_mask[x_,y_]:
        #         return True
        # return False

    if show_colison_bool:
        map_of_collision = np.zeros([x_map_max, y_map_max], dtype=np.int8)

        # The map_of_collision has ones everywhere that there is an object so we make it 2 and check for 3s
        map_of_collision[rect_x_fill, rect_y_fill] = 2
        map_of_collision[rect_x_edge, rect_y_edge] = 2

        # add the colision bask (which is all ones) so now 3 represents collision
        map_of_collision = map_of_collision + collision_mask



        # Check for 3s
        if 3 in map_of_collision:
            in_collision = True

        if show_colison_bool:
            if in_collision:
                plt.title("BAD! THIS CAR IS IN COLLISION")
            else:
                plt.title("this car is NOT in collision")

            # let us add the nose to the car just to see it
            point_nose = np.array([[((CAR_X - 1) / 2)], [0.0]])
            point_nose_rot = np.matmul(rotation_matrix, point_nose)
            map_of_collision[int(round(point_nose_rot[0,0] + x_center)), int(round(point_nose_rot[1,0] + y_center))] = 5

            point_nose = np.array([[0.0], [0.0]])
            point_nose_rot = np.matmul(rotation_matrix, point_nose)
            map_of_collision[int(round(point_nose_rot[0,0] + x_center)), int(round(point_nose_rot[1,0] + y_center))] = 6

            if trailerTheta is not None:
                global GLOBAL_MAP_OF_COLLISION
                GLOBAL_MAP_OF_COLLISION = map_of_collision



            elif isTrailer:
                map_of_collision = np.add(map_of_collision, GLOBAL_MAP_OF_COLLISION)
                global TRAILER_SAVE_COUNT
                global TRAILER_SAVE_COLLISION_MASK

                if trailer_save_first:
                    # we need to save the full map of this trailer motion and this is our first iteration

                    TRAILER_SAVE_COLLISION_MASK = map_of_collision
                    TRAILER_SAVE_COUNT = 0

                elif trailer_save:

                    if TRAILER_SAVE_COUNT is None:
                        TRAILER_SAVE_COUNT = 0
                    TRAILER_SAVE_COUNT += 1
                    if TRAILER_SAVE_COUNT % TRAILER_SAVE_MOD == 0:
                        TRAILER_SAVE_COLLISION_MASK = np.subtract(TRAILER_SAVE_COLLISION_MASK, GLOBAL_COLLISION_MASK)
                        TRAILER_SAVE_COLLISION_MASK = np.add(TRAILER_SAVE_COLLISION_MASK, map_of_collision)

                else:
                    view_collision_mask(map_of_collision)



            else:
                view_collision_mask(map_of_collision)

        if trailerTheta is not None:
            trailer_result = collision_check_trailer(x_center,y_center,theta, show_colison_bool=show_colison_bool,trailerTheta=trailerTheta,trailer_save=trailer_save,trailer_save_first=trailer_save_first)
            return trailer_result or in_collision
        return in_collision

def get_collision_mask():
    return GLOBAL_COLLISION_MASK

def get_raw_map():
    return GLOBAL_RAW_MAP

def set_raw_map(x,y,color_RGB):
    global GLOBAL_RAW_MAP
    GLOBAL_RAW_MAP[x,y,:] = color_RGB

# sets a range of indexes to the same color
def set_raw_map_range(x_range,y_range,color_RGB):
    global GLOBAL_RAW_MAP
    GLOBAL_RAW_MAP[x_range[0]:x_range[1]+1,y_range[0]:y_range[1]+1,:] = color_RGB

def set_raw_map_array(numpyArr):
    global GLOBAL_RAW_MAP
    GLOBAL_RAW_MAP = np.copy(numpyArr)

def get_scrub_map():
    return GLOBAL_SCRUB_MAP

def set_scrub_map(x,y,color_RGB):
    global GLOBAL_SCRUB_MAP
    GLOBAL_SCRUB_MAP[x,y,:] = color_RGB

def set_scrub_map_range(x_range,y_range,color_RGB):
    global GLOBAL_SCRUB_MAP
    GLOBAL_SCRUB_MAP[x_range[0]:x_range[1]+1,y_range[0]:y_range[1]+1,:] = color_RGB

def set_scrub_map_array(numpyArr):
    global GLOBAL_SCRUB_MAP
    GLOBAL_SCRUB_MAP = np.copy(numpyArr)

# If the scrub map has changed you need to update the collision mask
def update_collision_mask():
    global GLOBAL_COLLISION_MASK
    GLOBAL_COLLISION_MASK = collision_mask_from_map(GLOBAL_SCRUB_MAP)

def set_collision_mask_array(numpyArr):
    global GLOBAL_COLLISION_MASK
    GLOBAL_COLLISION_MASK = np.copy(numpyArr)



# This function intializes the global collision_mask and the global raw_map  returns the start and end qs
def init_collision_checking(path_to_map_png):
    global GLOBAL_COLLISION_MASK
    global GLOBAL_RAW_MAP
    global GLOBAL_SCRUB_MAP
    global GLOBAL_MAP_PATH
    GLOBAL_MAP_PATH = path_to_map_png
    raw_map = import_map(path_to_map_png)
    (start_q, goal_q, map) = find_start_goal_scrub(raw_map)

    print("Start: ", start_q)
    print("Goal: ", goal_q)

    GLOBAL_COLLISION_MASK = collision_mask_from_map(map)
    GLOBAL_RAW_MAP = raw_map
    GLOBAL_SCRUB_MAP = map


    return start_q, goal_q


def init_collision_checking_base_map(path_to_map_png):
    global GLOBAL_COLLISION_MASK
    global GLOBAL_RAW_MAP
    global GLOBAL_SCRUB_MAP
    global GLOBAL_MAP_PATH
    GLOBAL_MAP_PATH = path_to_map_png
    raw_map = import_map_BW(path_to_map_png)

    GLOBAL_COLLISION_MASK = collision_mask_from_map(raw_map)
    GLOBAL_RAW_MAP = raw_map
    GLOBAL_SCRUB_MAP = raw_map


def d2r(degree):
    return np.deg2rad(degree)

def steering_fun(x,y,th):
    phi = [-2 * np.pi / 10, -np.pi / 10, 0, np.pi / 10, 2 * np.pi / 10]
    v = 10
    L = 10
    nodes = []
    for i in phi:
        dx = v*math.cos(i)*math.cos(th)
        dy = v*math.cos(i)*math.sin(th)
        dth = v/L*math.sin(i)
        nodes.append((x+dx,y+dy,th+dth))
    return nodes

def run_on_map(path_to_map_png):
    init_collision_checking(path_to_map_png)

    # test a path
    #path = [(1,1,0),(1,2,0),(1,3,0),(1,4,0),(1,5,0),(1,6,0),(2,6,0),(3,6,0),(4,6,0),(5,6,0)]
    # path = [(100,100,np.pi/2),(103,110,d2r(80)),(106,120,d2r(80)),(124,190,d2r(0))]
    # view_path(path)

    # path = [(75,250,np.pi/2)]
    # for i in range(5):
    #     steer = steering_fun(path[i][0],path[i][1],path[i][2])
    #     p = steer[4]
    #     path.append(p)
    # view_path(path)

    # returnBool = collision_check(85, 248, 3.141592653589793, show_colison_bool=True)
    # print("This should NOT be in collision: ",returnBool)
    #
    # # test the start position of the car and check that its orientation is correct
    # returnBool = collision_check(10, 5, np.pi, show_colison_bool=True)
    # print("This should NOT be in collision: ",returnBool)


    # test a config of the car
    T0 = time.time()
    returnBool = collision_check(219, 242, 0, show_colison_bool=True, trailerTheta=-np.pi/4)
    print("Check for False collision1 got: " + str(returnBool) + " and took: ", time.time() - T0, " sec")

    T0 = time.time()
    returnBool = collision_check(219, 242, np.pi/4, show_colison_bool=False)
    print("Check for False collision2 got: " + str(returnBool) + " and took: ", time.time() - T0, " sec")

    T0 = time.time()
    returnBool = collision_check(219, 242, np.pi/5, show_colison_bool=False)
    print("Check for False collision3 got: " + str(returnBool) + " and took: ", time.time() - T0, " sec")

    T0 = time.time()
    returnBool = collision_check(250, 250, 0, show_colison_bool=False)
    print("Check for True collision got: " + str(returnBool) + " and took: ", time.time() - T0, " sec")

    T0 = time.time()
    returnBool = collision_check(100, 100, 0, show_colison_bool=False)
    print("Check for True EASY collision got: " + str(returnBool) + " and took: ", time.time() - T0, " sec")

    returnBool = collision_check(219, 242, 0, show_colison_bool=True)
    print("This should NOT be in collision: ",returnBool)

    returnBool = collision_check(219, 242, np.pi/5, show_colison_bool=True)
    print("This should NOT be in collision: ",returnBool)

    returnBool = collision_check(219, 242, np.pi/4, show_colison_bool=True)
    print("This should NOT be in collision: ",returnBool)


    T0 = time.time()
    returnBool = colision_check(219, 242, np.pi / 4, colision_mask, show_colison_bool=False)
    print("Time for the old version to find no colision: ", time.time() - T0, "sec")
    print("This should NOT be in colision: ", returnBool)




    returnBool = collision_check(250, 250, 0, show_colison_bool=True)
    print("This should be in collision: ",returnBool)




    print("breakpoint")



if __name__ == "__main__":
    run_on_map("sample_maps/1/Trial map_5.png")

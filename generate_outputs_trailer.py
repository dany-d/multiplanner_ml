# this file will run a bunch of path planners on a bunch of maps and compile their runtime+cost in a csv

import argparse
import glob
import os
import time
import sys
import traceback
from pandas import DataFrame
from map_runner import init_collision_checking, view_path
import timeout_decorator
from angle_calc import AngDiff
import math

# import planners (remember to add them to PLANNERS and PLANNER_NAMES
# from A_star import astar as A_star_runner
# from A_star_1_1 import astar as A_star_1_1_runner
# from A_star_1_5 import astar as A_star_1_5_runner
# from A_star_2 import astar as A_star_2_runner

# import A_star
import A_star_trailer_1_5
import A_star_trailer_2
# import matt_astar
import birrt_trailer

# A_star_runner = A_star.astar
A_star_1_5_runner_trailer = A_star_trailer_1_5.run
A_star_2_runner_trailer = A_star_trailer_2.run
birrt_runner_trailer=birrt_trailer.birrt




PLANNERS = [birrt_runner_trailer, A_star_1_5_runner_trailer, A_star_2_runner_trailer]
PLANNER_NAMES = ["BiRRT_trailer","A_star_1_5_trailer","A_star_2_trailer"]


OUTPUTS = []
for planner_name in PLANNER_NAMES:
    OUTPUTS.append(planner_name + "_runtime")
    OUTPUTS.append(planner_name + "_cost")


PATH_TO_DIRECTORY_CONTAINING_PNG_MAPS = None

TIMEOUT = 200 #seconds TODO: ENABLE TIMEOUT (uncomment below) (when you uncomment it wont run in pycharm)

# builds a solutions directory if none exists, returns the directory path
def build_save_solutions_directory(png_directory, planner_name):
    global PATH_TO_DIRECTORY_CONTAINING_PNG_MAPS
    path_split = os.path.split(os.path.abspath(png_directory))
    PATH_TO_DIRECTORY_CONTAINING_PNG_MAPS = path_split[0] + '/' + path_split[1]
    solutions_directory = os.path.abspath(path_split[0] + '/' + path_split[1] + "_" + str(planner_name) + "_solutions")
    if not os.path.exists(solutions_directory):
        os.makedirs(solutions_directory)
    return solutions_directory

def build_a_save_solution_filepath(planner_name, original_png_path):
    solutions_directory = os.path.abspath(PATH_TO_DIRECTORY_CONTAINING_PNG_MAPS + "_" + str(planner_name) + "_solutions")
    save_image_name = str(os.path.basename(original_png_path))
    save_image_name = save_image_name.split(".png")[0]
    save_image_name = save_image_name + "-" + planner_name + "-SOLUTION.png"
    full_save_image_path = solutions_directory + "/" + save_image_name
    return full_save_image_path

def calc_cost_from_path(path):
    total_dist = 0
    for i in range(len(path) - 1):
        q0 = path[i]
        x0 = q0[0]
        y0 = q0[1]
        t0 = q0[2]

        q1 = path[i+1]
        x1 = q1[0]
        y1 = q1[1]
        t1 = q1[2]

        dt = AngDiff(t0,t1)

        dist = math.sqrt((x0-x1)*(x0-x1) + (y0-y1)*(y0-y1) + dt*dt)
        total_dist += dist


    return total_dist

# @timeout_decorator.timeout(200)
def run_planner_helper(planner, start_q, goal_q):
    return planner(start_q, goal_q)

# takes a function pointer and evaluates it, returns runtime and cost and path
def evaluate_planner(planner, start_q, goal_q):

    start_time = time.time() # start timer
    path = run_planner_helper(planner, start_q, goal_q) # run alg with timeout
    end_time = time.time() # stop timer

    runtime = end_time - start_time

    cost = calc_cost_from_path(path)

    return runtime, cost, path



# If save_solutions = True will generate a png for each solution and save it to the system in a respective directory
def gen_outputs_directory(png_directory, save_solutions = False):
    png_files = []
    for filename in glob.iglob(png_directory + '/**/*.png', recursive=True):
        png_files.append(os.path.abspath(filename))

    # init dataframe
    cols = OUTPUTS
    cols.append("filename")
    df = DataFrame(index=range(len(png_files)), columns= cols)

    if save_solutions:
        for planner_name in PLANNER_NAMES:
            # init solutions directory
            build_save_solutions_directory(png_directory, planner_name)

    for row_index, png_path in enumerate(png_files):

        df.loc[row_index]["filename"] = os.path.basename(png_path)
        # Load the image into the collision checker
        start_q, goal_q = init_collision_checking(png_path)

        # evaluate over each path planner
        for planner, planner_name in zip(PLANNERS, PLANNER_NAMES):
            print("Running " + planner_name + " on " + str(os.path.basename(png_path)))

            try:
                runtime, cost, path = evaluate_planner(planner, start_q, goal_q)
            except Exception as e:
                # raise e
                print("Failed to run map: " + str(png_path) + " on planner: " + str(planner_name) +
                      "\nThe output data will be filled with -1. Here is the specific error:" +
                      "\n\nERROR:\n" + ''.join(traceback.format_stack()))
                print(str(type(e)) + " " + str(e) + "\n\nEND OF ERROR\n\n\n")
                runtime = -1
                cost = -1
                path = [[50,50,0,0],[50,50,0,0],[60,60,0,0]]

            df.loc[row_index][planner_name + "_runtime", planner_name + "_cost"] = [runtime, cost]

            print("took: " + str(runtime) + " sec")

            if save_solutions:
                view_path(path, save_filepath=build_a_save_solution_filepath(planner_name, png_path), isTrailer=True)

        # write the csv file
        try:
            df.to_csv(png_directory + "/outputs.csv")
        except:
            while(1):
                input("CLOSE THE OUTPUTS file and hit enter!")
                try:
                    df.to_csv(png_directory + "/outputs.csv")
                except:
                    pass
                else:
                    break

    print("Done.")


if __name__== "__main__":
    parser = argparse.ArgumentParser(description='This takes a directory with map pngs in it, generates a csv of all '
                                                 'the outputs for the neural net and saves it in the same directory\n'
                                                 'Usage: $python generate_outputs.py "sample_maps" --save_solutions')
    parser.add_argument('png_data_directory', action='store', type=str,
                        help='path to the directory with png images (will work if directories are nested)')
    parser.add_argument('--save_solutions', dest='save_solutions', action='store_true',
                        help="add this arg to save the solutions, (this will run slower)")
    parser.set_defaults(save_solutions=False)
    args = parser.parse_args()
    gen_outputs_directory(args.png_data_directory, args.save_solutions)
# This will create our time vs cost graph
import argparse
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np

def time_cost_graph(png_dir):
    # load the outputs to see what the time and cost will look like for each version
    outputs_df = pd.read_csv(png_dir + "/outputs.csv")
    algs = ["BiRRT_steer", "A_star_1_5_steer"]

    runtimes = []
    costs = []

    for alg in algs:
        runtime = outputs_df[alg+"_runtime"].values
        cost = outputs_df[alg+"_cost"].values

        runtimes.append(np.sum(runtime))
        costs.append(np.sum(cost))

    fig, ax = plt.subplots()
    ax.scatter(np.array(runtimes), np.array(costs))

    for i, txt in enumerate(algs):
        ax.annotate(txt, (runtimes[i], costs[i]))

    plt.xlabel('Runtime (s)', fontsize=12)
    plt.ylabel('Cost', fontsize=12)

    predictions_df = pd.read_csv(png_dir + "/Random_Forest_Prediction.csv")

    total_runtime_fastest = 0
    total_runtime_fastest_true = 0
    total_runtime_fastest_sum_cost_true = 0
    total_cost_lowest = 0
    total_cost_lowest_true = 0
    total_cost_lowest_sum_runtime_true = 0
    # get fastest alg

    count_correct_best_time_predicitions = 0
    count_correct_best_time_predicitions2 = 0


    num_rows = predictions_df.shape[0]
    for row_idx in range(num_rows):
        row = predictions_df.iloc[row_idx]
        lowest_cost = 1000000000
        lowest_cost_true = 0
        lowest_cost_runtime = 0
        fastest_runtime = 1000000000
        fastest_runtime_true = 0
        fastest_runtime_cost = 0
        TRUE_lowest_runtime_list = []

        TRUE_lowest_runtime = 100000000000000
        TRUE_lowest_runtime2 = 100000000000000
        TRUE_lowest_runtime_alg = ""
        TRUE_lowest_runtime_alg2 = ""
        PREDICTED_lowest_runtime_alg = ""

        for alg in algs:
            runtime = row[alg + "_runtime"]
            cost = row[alg + "_cost"]
            if runtime < fastest_runtime:
                fastest_runtime = runtime
                fastest_runtime_true = outputs_df[alg+"_runtime"].values[row_idx]
                fastest_runtime_cost = outputs_df[alg+"_cost"].values[row_idx]
                PREDICTED_lowest_runtime_alg = alg
            if cost < lowest_cost:
                lowest_cost = cost
                lowest_cost_true = outputs_df[alg+"_cost"].values[row_idx]
                lowest_cost_runtime = outputs_df[alg+"_runtime"].values[row_idx]

            #find the best one on true data and see if that matches up outside this for loop
            if outputs_df[alg+"_runtime"].values[row_idx] < TRUE_lowest_runtime:
                TRUE_lowest_runtime = outputs_df[alg+"_runtime"].values[row_idx]
                TRUE_lowest_runtime_alg = alg

            TRUE_lowest_runtime_list.append(outputs_df[alg+"_runtime"].values[row_idx])


        sortedList = sorted(range(len(TRUE_lowest_runtime_list)), key=lambda k: TRUE_lowest_runtime_list[k])

        assert(TRUE_lowest_runtime_alg == algs[sortedList[0]])

        TRUE_lowest_runtime_alg2 = algs[sortedList[1]]

        if TRUE_lowest_runtime_alg == PREDICTED_lowest_runtime_alg:
            count_correct_best_time_predicitions += 1

        if (TRUE_lowest_runtime_alg == PREDICTED_lowest_runtime_alg) or (TRUE_lowest_runtime_alg2 == PREDICTED_lowest_runtime_alg):
            count_correct_best_time_predicitions2 += 1

        total_runtime_fastest += fastest_runtime
        total_runtime_fastest_true += fastest_runtime_true
        total_runtime_fastest_sum_cost_true += fastest_runtime_cost
        total_cost_lowest += lowest_cost
        total_cost_lowest_true += lowest_cost_true
        total_cost_lowest_sum_runtime_true += lowest_cost_runtime

    print("out of " + str((num_rows)) + " maps, " + str(count_correct_best_time_predicitions) + " were predicted correctly for lowest runtime")
    print("This means we have an accuracy of: " + str(count_correct_best_time_predicitions/ (num_rows)))
    print("\n\n")

    print("for seccond choice accuracy we have predictied " + str(count_correct_best_time_predicitions2) + " correct predictions")
    print("This is an accuracy of: " + str(count_correct_best_time_predicitions2/num_rows))

    plt.scatter(total_runtime_fastest_true, total_runtime_fastest_sum_cost_true, c='red')
    ax.annotate("Automated Framework", (total_runtime_fastest_true, total_runtime_fastest_sum_cost_true))

    # plt.scatter(total_cost_lowest_sum_runtime_true, total_cost_lowest_true, c='green')
    # ax.annotate("Lowest Cost", (total_cost_lowest_sum_runtime_true, total_cost_lowest_true))

    plt.show()






if __name__== "__main__":
    parser = argparse.ArgumentParser(description="This will create the cost vs time graph")
    parser.add_argument('png_data_directory', action='store', type=str,
                        help='path to the directory with png images (will work if directories are nested)')
    args = parser.parse_args()
    time_cost_graph(args.png_data_directory)
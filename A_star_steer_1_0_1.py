from A_star_steer import astar

def run(start_q,goal_q):
    path = astar(start_q,goal_q,weight=1.01)
    return path


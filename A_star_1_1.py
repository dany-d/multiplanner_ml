from A_star import astar

def run(start_q,goal_q):
    path = astar(start_q,goal_q,weight=1.1)
    return path


from A_star_trailer import astar

def run(start_q,goal_q):
    start_q2 = (start_q[0],start_q[1],start_q[2],start_q[2])
    goal_q2 = (goal_q[0],goal_q[1],goal_q[2],goal_q[2])
    path = astar(start_q2,goal_q2,weight=1.5)
    return path

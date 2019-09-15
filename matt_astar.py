from heapq import heappush, heappop, heapify
from math import sqrt, sin, cos, radians, degrees
import math

from map_runner import init_collision_checking, collision_check, view_path

import numpy as np
pi = np.pi

def matt_astar(start_q, goal_q):
    goalconfig = goal_q

    CONNECTEDNESS = 4
    DISTANCE_HEURISTIC = "manhattan"  # manhattan or euclidean


    def euclidean_dist_to_goal(DOF):
        diff_theta = abs(math.atan2(sin(goalconfig[2] - DOF[2]), goalconfig[2] - DOF[2]))

        result = sqrt(((goalconfig[0] - DOF[0]) * (goalconfig[0] - DOF[0])) +
                      ((goalconfig[1] - DOF[1]) * (goalconfig[1] - DOF[1])) +
                      (diff_theta * diff_theta))
        return result
        # sqrt(pow(goalconfig[0] - DOF[0], 2.0) + pow(goalconfig[1] - DOF[1], 2.0))# +
        # pow(math.atan2(sin(goalconfig[2]-DOF[2]), goalconfig[2]-DOF[2]), 2))


    def manhattan_dist_to_goal(DOF):
        diff_theta = abs(math.atan2(sin(goalconfig[2] - DOF[2]), goalconfig[2] - DOF[2]))
        result = abs(goalconfig[0] - DOF[0]) + abs(goalconfig[1] - DOF[1]) + diff_theta
        return result


    class node:
        def __init__(self, DOF_list, dist, parent, visited):
            self.DOF_list = DOF_list
            self.dist = dist
            self.parent = parent
            self.visited = visited

            self.f_score = self.calc_f_score(dist, DISTANCE_HEURISTIC)

        def calc_f_score(self, d, method="manhattan"):
            if method == "euclidean":
                f_score = d + euclidean_dist_to_goal(self.DOF_list)
            elif method == "manhattan":
                f_score = d + manhattan_dist_to_goal(self.DOF_list)

            return f_score  # TODO: change this for eyulclydian and the other type

        def q(self):
            # This just formats nicely if you want to put it on a heapq
            return self#(self.f_score, self)

        def s(self):
            # This just formats nicely if you want to put it in a set
            return (self.DOF_list)

        def __eq__(self, other):
            # overriding equality
            if isinstance(other, node):
                return self.DOF_list == other.DOF_list
            return False

        def __ne__(self, other):
            # just overriding inequality
            return not self.__eq__(other)

        def __lt__(self, other):
            # just overriding inequality
            return self.f_score < other.f_score


    def not_visited_adjacent(c_node, visited_list, eps, theta_eps_deg, connectedness=8):
        # returns a list of adjacent nodes that have not been visited
        nodes = []

        # start with changing the theta at the current node
        theta_plus = float(radians((int(degrees(c_node.DOF_list[2])) + theta_eps_deg) % 360))
        if theta_plus >= pi:
            theta_plus -= 2 * pi

        theta_minus = float(radians((int(degrees(c_node.DOF_list[2])) - theta_eps_deg) % 360))
        if theta_minus >= pi:
            theta_minus -= 2 * pi

        DOF_theta_plus = [c_node.DOF_list[0], c_node.DOF_list[1], theta_plus]
        DOF_theta_minus = [c_node.DOF_list[0], c_node.DOF_list[1], theta_minus]
        if not DOF_theta_plus in visited_list:
            nodes.append(node(DOF_theta_plus, c_node.dist + eps, c_node,
                              False))  # TODO: THIS IS WRONG YOU NEED TO ACCOUNT FOR NEW DISTANCE FACTORING IN TURN BETTER
        if not DOF_theta_minus in visited_list:
            nodes.append(node(DOF_theta_minus, c_node.dist + eps, c_node, False))

        # now lets move in x and y
        if connectedness == 4:
            go = [[0.0, eps, eps],  # north
                  [eps, 0.0, eps],  # east
                  [0.0, -eps, eps],  # south
                  [-eps, 0.0, eps]]  # west
        if connectedness == 8:
            diag_dist = sqrt((eps * eps) + (eps * eps))
            go = [[0.0, eps, eps],  # north
                  [eps, eps, diag_dist],  # north-east
                  [eps, 0.0, eps],  # east
                  [eps, -eps, diag_dist],  # south-east
                  [0.0, -eps, eps],  # south
                  [-eps, -eps, diag_dist],  # south-west
                  [-eps, 0.0, eps],  # west
                  [-eps, eps, diag_dist]]  # north-west

        for dir in go:
            DOF = [c_node.DOF_list[0] + dir[0], c_node.DOF_list[1] + dir[1], c_node.DOF_list[2]]
            if not DOF in visited_list:
                nodes.append(node(DOF, c_node.dist + dir[2], c_node, False))

        return nodes


    def is_this_node_in_my_pq(test_node, visit_q):
        # returns the index of where it is in the pq or none if it does not exist
        for i, pq_entry in enumerate(visit_q):
            if (test_node.DOF_list == pq_entry.DOF_list):
                return i
        return None


    visit_q = []
    visited_list = []
    eps = 10
    theta_eps_deg = 45  # pi*(1.0/8.0)
    start_config = start_q
    start_node = node(start_q, dist=0.0, parent=None, visited=True)
    visited_list.append(start_node.s())

    current_node = start_node

    handles = []

    # heappush(visit_q, start_node.q())
    heappush(visit_q, start_node.q())

    while ((len(visit_q) != 0) and (euclidean_dist_to_goal(current_node.DOF_list) > (eps))):
        heapify(visit_q)
        current_node = heappop(visit_q)#[1]
        current_node.visited = True
        visited_list.append(current_node.s())

        for neibr in not_visited_adjacent(current_node, visited_list, eps, theta_eps_deg, CONNECTEDNESS):
            # check for collision
            x_,y_,theta_ = neibr.DOF_list
            if collision_check(x_,y_,theta_):
                # draw red spot
                # handles.append(
                #     env.plot3(points=array(((neibr.DOF_list[0], neibr.DOF_list[1], 0.1))),  # neibr.DOF_list[2]))),
                #               pointsize=4.0,
                #               colors=array(((1, 0, 0, 1)))))
                continue
            # handles.append(env.plot3(points=array(((neibr.DOF_list[0], neibr.DOF_list[1], 0.1))),  # neibr.DOF_list[2]))),
            #                          pointsize=4.0,
            #                          colors=array(((0, 0, 1, 1)))))
            iter_of_node_in_pq = is_this_node_in_my_pq(neibr, visit_q)
            if (iter_of_node_in_pq is not None):
                if (neibr.dist < visit_q[iter_of_node_in_pq].dist):  # TODO: I THINK THE PROBELM IS HERE WITH THE DIRECTION OF THIS SIGN!!!!!!
                    # keep the old node the same but swap out all of its values
                    old_node = visit_q[iter_of_node_in_pq]
                    old_node.dist = neibr.dist
                    old_node.parent = neibr.parent
                    old_node.f_score = neibr.f_score
                    visit_q[iter_of_node_in_pq] = old_node.q()

                    # also you need to re do its entry in the pq
                    # must heapify now
                    heapify(visit_q)
            else:
                # this is a new node we have not seen before, put it on the heap
                heappush(visit_q, neibr.q())

    path = [goalconfig]  # put your final path in this variable

    cost = current_node.dist

    while current_node.parent != None:
        path.append(current_node.DOF_list)
        current_node = current_node.parent

    path.append(current_node.DOF_list)
    path.reverse()
    return path


def main():
    # start = (253, 389, -1.5707963267948966)
    # Goal = (251, 85, -1.5707963267948966)
    start_q, goal_q = init_collision_checking("sample_maps/Trial map_5.png")
    goal_q = [50, start_q[1] + 3, start_q[2]]
    path = matt_astar(start_q, goal_q)
    print(path)
    view_path(path)


if __name__ == '__main__':
    main()

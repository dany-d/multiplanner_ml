import numpy
from map_runner import init_collision_checking, collision_check, view_path
import time
from math import cos,sin
from angle_calc import AngDiff


class Node():
    """A node class for A* Pathfinding"""

    def __init__(self, parent=None, position=None):
        self.parent = parent
        self.position = position

        self.g = 0
        self.h = 0
        self.f = 0

    def __eq__(self, other):
        return self.position == other.position


def euclideanDistance(config1, config2):
    distance = 0
    for x in range(2):
        distance += (((config1.position[0] - config2.position[0]) ** 2) +
                     ((config1.position[1] - config2.position[1]) ** 2) +
                     (AngDiff(config1.position[2], config2.position[2]))**2)

                     # 10*(config1.position[2] - config2.position[2]) ** 2
                     # +(config1.position[2] - config2.position[2] + 2*numpy.pi) ** 2
                     # +(config1.position[2] - config2.position[2] - 2*numpy.pi) ** 2)

    return distance ** (1 / 2)


def contains(list, position):
    for x in list:

        if x.position == position:
            return True

    return False


def steering_fun(x,y,th):
    curve = 5
    phi = [-2 * numpy.pi /curve, -numpy.pi /curve, 0, numpy.pi /curve, 2 * numpy.pi /curve]
    v = 10
    L = 10
    nodes = []
    for i in phi:
        dx = v*cos(i)*cos(th)
        dy = v*cos(i)*sin(th)
        dth = v/L*sin(i)
        nodes.append([x+dx,y+dy,th+dth])
    v = -v
    for i in phi:
        dx = v * cos(i) * cos(th)
        dy = v * cos(i) * sin(th)
        dth = v / L * sin(i)
        nodes.append([x + dx, y + dy, th + dth])
    return nodes


def astar(start, end,weight=1):
    T0 = time.time()
    """Returns a list of tuples as a path from the given start to the given end in the given maze"""

    # Create start and end node
    start_node = Node(None, start)
    start_node.g = start_node.h = start_node.f = 0
    end_node = Node(None, end)
    end_node.g = end_node.h = end_node.f = 0

    # Initialize both open and closed list
    open_list = []
    closed_list = []

    # Add the start node
    open_list.append(start_node)

    # Loop until you find the end
    while len(open_list) > 0:
        if time.time()-T0 > 15: #fifteen secs
            return None

        # Get the current node
        current_node = min(open_list, key=lambda o: o.f)
        # print(current_node.position)

        # for index, item in enumerate(open_list):
        #     if item.f < current_node.f:
        #         current_node = item
        #         current_index = index
        # print(current_node.position, current_node.f)
        # remove current off open list, add to closed list
        open_list.remove(current_node)
        closed_list.append(current_node)

        step = 10
        # Found the goal
        if euclideanDistance(current_node, end_node) < 20:
            path = []
            current = current_node
            while current is not None:
                path.append(current.position)
                current = current.parent
            path = path[::-1]
            path.append(end)
            return path # Return reversed path



        # Generate children
        children = []
        new_nodes = steering_fun(current_node.position[0],current_node.position[1],current_node.position[2])
        for node_position in new_nodes:  # Adjacent squares

            if node_position[2] > numpy.pi:
                node_position[2] = 3.141592653589793
            elif node_position[2] < -numpy.pi:
                node_position[2] = -3.141592653589793

            # # Make sure within range
            # if node_position[0] > (len(maze) - 1) or node_position[0] < 0 or node_position[1] > (len(maze[len(maze)-1]) -1) or node_position[1] < 0:
            #     continue

            # Check for collision!!
            if collision_check(node_position[0], node_position[1], node_position[2]) != 0:
                # collision_check(node_position[0], node_position[1], node_position[2], show_colison_bool=True)
                continue

            # Create new node
            new_node = Node(current_node, node_position)

            # Append
            children.append(new_node)

        # Loop through children
        for child in children:
            # Child is on the closed list
            if contains(closed_list, child.position):
                continue

            # Create the f, g, and h values
            # todo NEED TO ADD PENALIZING FOR THETA
            child.g = current_node.g + euclideanDistance(current_node, child)
            child.h = euclideanDistance(child, end_node)
            child.f = child.g + weight * child.h

            # Child is already in the open list
            if contains(open_list, child.position):
                continue

            # for open_node in open_list:
            #     if open_node.position == child.position and child.g > open_node.g:
            #         continue



            # Add the child to the open list
            open_list.append(child)





def main():
    # start = (253, 389, -1.5707963267948966)
    # Goal = (251, 85, -1.5707963267948966)
    start_q, goal_q = init_collision_checking("Automap_data/autoMaps/AG_S_146.png")
    path = astar(start_q, goal_q)
    print(path)
    view_path(path)


if __name__ == '__main__':
    main()

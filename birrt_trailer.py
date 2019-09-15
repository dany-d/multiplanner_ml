
# Importing requires files
from collections import namedtuple
from random import randint, uniform
from math import sqrt, cos, sin
from map_runner import init_collision_checking, collision_check, view_path
import numpy
import time
from angle_calc import AngDiff


Point = namedtuple('Point', 'x y th th2')
Node = namedtuple('Node', 'id pos parent')
Region = namedtuple('Region', 'pos width height')  # bottom left x,y

# Define the world size and goal bias
WORLD_SIZE = 500
# GOAL_CHANCE = 0.5 # Don't need for BiRRT
width=1
height=1
velocity = 5# This is step size
step=10
step2=10
velocity_theta=numpy.pi/4;
step2_theta=numpy.pi/4;

# Using Euclidean distance for heuristic
def dist(config1, config2):
    distance = 0
    angle=AngDiff(config1[2],config2[2])
    angleth = AngDiff(config1[3], config2[3])
    for x in range(2):
        distance += (((config1[0] - config2[0]) ** 2) +
                     ((config1[1] - config2[1]) ** 2) +
                     20*(angle) ** 2 +
                     10 * (angleth) ** 2)

    return distance ** (1 / 2)

# Function to check if the sampled node is in Goal region
def in_region(point, region):
    return region[0] <= point[0] <= region[0] + width and region[1] <= point[1] <= region[1] + width and region[2] <= point[2] <= region[2] + height

# Function to find the tree node closest to sampled node
def get_closest(nodes, point):
    return min(nodes, key=lambda x: dist(x.pos, point))

# Function to step from nearest node to sampled node
def steer(point1, point2):
    """Return an intermediate point on the line between point1 and point2"""
    total_offset = abs(point2.x - point1.x) + abs(point2.y - point1.y) + abs(point2.th - point1.th)
    x = point1.x + velocity * ((point2.x - point1.x) / total_offset)
    y = point1.y + velocity * ((point2.y - point1.y) / total_offset)
    th = point1.th + velocity_theta * ((point2.th - point1.th) / total_offset)
    return Point(x, y, th)

def steer_smooth(point1, point2):
    """
    Return an intermediate point on the line between point1 and point2
    """
    total_offset = abs(point2[0] - point1[0]) + abs(point2[1] - point1[1]) + abs(point2[2] - point1[2])
    x = point1[0] + step2 * ((point2[0] - point1[0]) / total_offset)
    y = point1[1] + step2 * ((point2[1] - point1[1]) / total_offset)
    th = point1[2] + step2_theta * ((point2[2] - point1[2]) / total_offset)
    return Point(x, y, th)

def smooth(path,n):
    index1=100
    index2 = 100
    local_flag=1
    nodes_smooth = [Node(0, (1,2,3), 0)]
    for k in range(n):
        while index1 == index2:
            index1=randint(0, len(path)-1)
            index2=randint(0, len(path)-1)

            # WORLD_SIZE) % (len(path)

        if(index2<index1):
            temp=index1
            index1=index2
            index2=temp

        # print("Iteration Number is",k+1)
        # print("Length of path is",len(path))
        # print("Index 1 is", index1)
        # print("Index 2 is", index2)
        node1=Point(path[index1][0],path[index1][1],path[index1][2])
        node2=Point(path[index2][0],path[index2][1],path[index2][2])
        node3=node1

        nodes_smooth.clear()
        nodes_smooth = [Node(0, node1, 0)]

        while dist(node1,node2)>step:
            mid_node=steering_fun(node1[0], node1[1],node1[2],node2)
            if mid_node is None:
                local_flag=0
                break
            else:
                nodes_smooth.append(Node(len(nodes_smooth), mid_node, len(nodes_smooth)-1))
                node1=mid_node
                local_flag=1
            if dist(node1, node2) <= step:
                if collision_check(node2[0], node2[1], node2[2]):
                    local_flag = 0
                    break
                else:
                    nodes_smooth.append(Node(len(nodes_smooth), Point(node2[0],node2[1],node2[2]), len(nodes_smooth)-1))
                    local_flag=1
                    break

        if local_flag == 1:
            del path[index1:index2]
            path.insert(index1, [node3[0],node3[1],node3[2]])
            for l in range(1,len(nodes_smooth)):
                item=[nodes_smooth[l].pos.x,nodes_smooth[l].pos.y,nodes_smooth[l].pos.th]
                path.insert(l+index1,item)

        index1=index2
    return path

def steering_fun(x,y,th,th2,z_randA):
    curve = 5
    phi = [-2 * numpy.pi / curve, -numpy.pi / curve, 0, numpy.pi / curve, 2 * numpy.pi / curve]
    v = 7
    L = 10
    L2 = 10
    nodes = []
    for i in phi:
        dx = v * cos(i) * cos(th)
        dy = v * cos(i) * sin(th)
        dth = v / L * sin(i)
        dth2 = v / L2 * cos(i) * sin(th - th2)
        nodes.append((x + dx, y + dy, th + dth, th2 + dth2))
    v = -v
    for i in phi:
        dx = v * cos(i) * cos(th)
        dy = v * cos(i) * sin(th)
        dth = v / L2 * sin(i)
        dth2 = v / L2 * cos(i) * sin(th - th2)
        nodes.append((x + dx, y + dy, th + dth, th2 + dth2))

    nodes.sort(key=lambda x: dist(x, z_randA))

    for i in range(len(nodes)):
        closest = nodes[i]
        if collision_check(closest[0],closest[1], closest[2],trailerTheta=closest[3]):
            continue
        else:
            return Point(closest[0],closest[1], closest[2],closest[3])

def birrt(start, Goal):
    T0 = time.time()
    start_pos = Point(start[0], start[1], start[2],start[3])
    end_pos = Point(Goal[0], Goal[1], Goal[2],Goal[3])

    nodes_start = [Node(0, start_pos, 0)]
    nodes_goal = [Node(0, end_pos, 0)]

    flag=0
    i=0
    j=0

    while True:
        if time.time() - T0 > 15:  # fifteen secs
            print("The code took more than 15 sec")
            return None

        #### FIRST TREE
        while i<=j: # To Ensure Both Trees are Balanced
            if time.time() - T0 > 15:  # fifteen secs
                return None

            z_randA = Point(randint(0, WORLD_SIZE), randint(0, WORLD_SIZE),uniform(-numpy.pi,numpy.pi),uniform(-numpy.pi,numpy.pi))

            if collision_check(z_randA[0], z_randA[1],z_randA[2],trailerTheta=z_randA[3]):
                continue

            nearestA = get_closest(nodes_start, z_randA)

            if z_randA == nearestA.pos:
                continue

            # Moving 1 step in direction of the random node

            new_posA=steering_fun(nearestA.pos[0], nearestA.pos[1],nearestA.pos[2],nearestA.pos[3],z_randA)
            if new_posA is None:
                continue

            # new_posA = steer(nearestA.pos, z_randA)

            # if collision_check(new_posA[0], new_posA[1], new_posA[2]):
            #     continue

            nodes_start.append(Node(len(nodes_start), new_posA, nearestA.id))

            # Find the node in tree B closes to new_posA
            nearestinB = get_closest(nodes_goal, new_posA)


            # Use RRT Connect to reach this node in tree B
            while dist(new_posA,nearestinB.pos)>step:
                newer_posA = steering_fun(new_posA[0], new_posA[1],new_posA[2],new_posA[3],nearestinB.pos)
                    # steer(new_posA,nearestinB.pos)
                if newer_posA is None:
                    break
                else:
                    # nearestA = get_closest(nodes_start, newer_posA)
                    new_posA = newer_posA
                    nodes_start.append(Node(len(nodes_start), newer_posA, len(nodes_start)-1))
                    # path.append([nodes_start.pos.x, nodes_start.pos.y, nodes_start.pos.th, nodes_start.pos.th2])
                    # view_path(nodes_start.pos, isTrailer=True)
                if dist(nearestinB.pos,newer_posA)<=step+10:
                    # nearestA = get_closest(nodes_start, nearestinB.pos)
                    # nodes_start.append(Node(len(nodes_start), nearestinB.pos, len(nodes_start)-1))
                    flag=1
                    break

            i = i + 1

            # if len(nodes_start) % 100 == 0:
            # print("{} Nodes Searched in Tree A".format(len(nodes_start)))

            if flag==1:
                nodes_g = nodes_goal
                path = []
                set_goal=0 # Defining a variable to add goal to the path
                nodes_g.sort(key=lambda n: n.id)
                # Highlight path from goal back to start position

                current_node_g=nearestinB
                # current_node_g = nodes_g[-1]
                if current_node_g.id == 0:
                    path.append([current_node_g.pos.x, current_node_g.pos.y, current_node_g.pos.th,current_node_g.pos.th2])
                    set_goal=0

                while current_node_g.id != 0:
                    parent = nodes_g[current_node_g.parent]
                    path.append([current_node_g.pos.x, current_node_g.pos.y, current_node_g.pos.th,current_node_g.pos.th2])
                    current_node_g = parent
                    set_goal=1

                if set_goal==1:
                    path.append([current_node_g.pos.x, current_node_g.pos.y, current_node_g.pos.th,current_node_g.pos.th2])

                path.reverse()
                nodes_s = nodes_start
                nodes_s.sort(key=lambda n: n.id)

                current_node_s = nodes_s[-1]
                while current_node_s.id != 0:
                    parent = nodes_s[current_node_s.parent]
                    path.append([current_node_s.pos.x, current_node_s.pos.y, current_node_s.pos.th, current_node_s.pos.th2])
                    current_node_s = parent

                path.append([current_node_s.pos.x, current_node_s.pos.y, current_node_s.pos.th, current_node_s.pos.th2]) # Add start to the que

                # view_path(path)
                # path = smooth(path, 100)
                path.reverse()
                view_path(path,isTrailer=True)
                return path
                # return nodes_start # Need to return both path

            # if len(nodes_start)>=10:

        #### SECOND TREE

        while j <= i:
            if time.time() - T0 > 15:  # fifteen secs
                return None
            z_randB = Point(randint(0, WORLD_SIZE), randint(0, WORLD_SIZE),uniform(-numpy.pi,numpy.pi),uniform(-numpy.pi,numpy.pi))

            if collision_check(z_randB[0], z_randB[1],z_randB[2],trailerTheta=z_randB[3]):
                continue

            nearestB = get_closest(nodes_goal, z_randB)

            if z_randB == nearestB.pos:
                continue

            # Moving 1 step in direction of the random node

            new_posB = steering_fun(nearestB.pos[0], nearestB.pos[1], nearestB.pos[2],nearestB.pos[3], z_randB)
            if new_posB is None:
                continue

            # new_posB = steer(nearestB.pos, z_randB)

            # if collision_check(new_posB[0], new_posB[1], new_posB[2]):
            #     continue

            nodes_goal.append(Node(len(nodes_goal), new_posB, nearestB.id))

            # Find the node in tree B closes to new_posA
            nearestinA = get_closest(nodes_start, new_posB)

            # Use RRT Connect to reach this node in tree B
            while dist(new_posB, nearestinA.pos) > step:
                newer_posB = steering_fun(new_posB[0], new_posB[1],new_posB[2],new_posB[3],nearestinA.pos)
                if newer_posB is None:
                    break
                else:
                    # nearestB = get_closest(nodes_goal, newer_posB)
                    new_posB = newer_posB
                    nodes_goal.append(Node(len(nodes_goal), newer_posB, len(nodes_goal)-1))
                if dist(nearestinA.pos, newer_posB) <= step+10:
                    # nearestB = get_closest(nodes_goal, nearestinA.pos)
                    # nodes_goal.append(Node(len(nodes_goal), nearestinA.pos, len(nodes_goal)-1))
                    flag=1
                    break
                # new_posA=Node(len(nodes_start)-1,new_posA,nearestA.id)
            j = j + 1

            # if len(nodes_goal) % 100 == 0:
            #     print("{} Nodes Searched in Tree B".format(len(nodes_goal)))

            # if len(nodes_goal)==4000:
            #     return nodes_goal# Need to return both path

            if flag==1:
                nodes_g = nodes_goal
                path = []
                set_goal=0

                nodes_g.sort(key=lambda n: n.id)
                # Highlight path from goal back to start position

                current_node_g = nodes_g[-1]
                if current_node_g.id == 0:
                    path.append([current_node_g.pos.x, current_node_g.pos.y, current_node_g.pos.th, current_node_g.pos.th2])
                    set_goal=0

                while current_node_g.id != 0:
                    parent = nodes_g[current_node_g.parent]
                    path.append([current_node_g.pos.x, current_node_g.pos.y, current_node_g.pos.th, current_node_g.pos.th2])
                    current_node_g = parent
                    set_goal=1

                if set_goal==1:
                    path.append([current_node_g.pos.x, current_node_g.pos.y, current_node_g.pos.th, current_node_g.pos.th2])

                path.reverse()
                nodes_s = nodes_start
                nodes_s.sort(key=lambda n: n.id)

                current_node_s = nearestinA
                # current_node_s = nodes_s[-1]
                while current_node_s.id != 0:
                    parent = nodes_s[current_node_s.parent]
                    path.append([current_node_s.pos.x, current_node_s.pos.y, current_node_s.pos.th, current_node_s.pos.th2])
                    current_node_s = parent

                path.append([current_node_s.pos.x, current_node_s.pos.y, current_node_s.pos.th, current_node_s.pos.th2])

                # view_path(path)
                # path = smooth(path, 100)
                path.reverse()
                view_path(path, isTrailer=True)
                return path

if __name__ == '__main__':
    start, Goal=init_collision_checking("Automap_data/autoMaps/AG_X_266.png")

    start_pos = Point(start[0], start[1],start[2],start[2])
    end_pos = Point(Goal[0], Goal[1],Goal[2],Goal[2])


    birrt(start_pos, end_pos)
    # print("test")
    # smooth_path=nodes
    # node_count = len(nodes)
    # print("Nodes Calculated {}".format(node_count))
    #
    #
    # nodes.sort(key=lambda n: n.id)
    #
    # path=[]
    # # Highlight path from goal back to start position
    #
    # current_node = nodes[-1]
    # while current_node.id != 0:
    #     parent = nodes[current_node.parent]
    #     path.append([current_node.pos.x,current_node.pos.y,current_node.pos.th])
    #     current_node = parent
    # print("Path found!")
    # smooth_path=smooth(nodes, 200)
    # view_path(smooth_path)
    #print("abc")
    # plt.show()

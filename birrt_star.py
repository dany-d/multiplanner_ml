import sys, random, math
from math import sqrt,cos,sin,atan2
import numpy
from map_runner import init_collision_checking, collision_check, view_path
from random import randint, uniform
from collections import namedtuple
import operator
import time


class Node:
    x = 0
    y = 0
    th = 0
    cost=0
    parent=None
    def __init__(self,x, y,th):
         self.x = x
         self.y = y
         self.th= th



def dist(p1,p2): #give nodes
    return sqrt((p1.x-p2.x)**2 + (p1.y-p2.y)**2 + 10*(p1.th - p2.th)**2)

def steer(point1, point2):
    """
    Return an intermediate point on the line between point1 and point2
    """
    total_offset = abs(point2.x - point1.x) + abs(point2.y - point1.y) + abs(point2.th - point1.th)
    if total_offset==0:
        print(point1,point2)
    x = point1.x + velocity * ((point2.x - point1.x) / total_offset)
    y = point1.y + velocity * ((point2.y - point1.y) / total_offset)
    th = point1.th + velocity * ((point2.th - point1.th) / total_offset)
    return x, y, th

def printnode(node):
    print(node.x,node.y,node.th)

# def printbranch(start,end):
#     while dist(start,end)<1:
#         start = (start[0] + delta * cos(theta), start[1] + delta * sin(theta),start[2])

def contains(list, node):
    for x in list:
        if (x.x,x.y,x.th) == (node.x,node.y,node.th):
            return True
    return False

def straightpathcollision(start,end):
    theta = atan2(end[1] - start[1], end[0] - start[0])
    delta = 1
    count = 0
    while dist(start,end)<1:
        start = (start[0] + delta * cos(theta), start[1] + delta * sin(theta),start[2])
        if collision_check(point):
            count = count + 1
    return count


def get_closest(nodes, node): # (nodelist,node)
    return min(nodes, key=lambda x: dist(x, node))


Region = namedtuple('Region', 'pos width height')

def chooseParent(nn,newnode,nodes):
    for p in nodes:
       if dist(p,newnode) < RADIUS and p.cost+dist(p,newnode) < nn.cost+dist(nn,newnode):
           nn = p
           newnode.cost=nn.cost+dist(nn,newnode)
           newnode.parent = nn
           continue
    return nn,newnode

def reWire(nodes,newnode):
    for i in range(len(nodes)):
        p = nodes[i]
        if p!=newnode.parent and dist(p,newnode)<RADIUS and newnode.cost+dist(p,newnode) < p.cost:
            p.parent = newnode
            p.cost=newnode.cost+dist(p,newnode)
            nodes[i]=p
        if i >10:
            continue
    return nodes

def getnewnode(nodes,othernode,start,end):
    if uniform(0, 1) < Goal_bias:
        rand_point = min(othernode, key=lambda x: dist(x, nodes[0]))
    else:
        rand_point = Node(math.ceil(randint(0, WORLD_SIZE)), math.ceil(randint(0, WORLD_SIZE)), math.ceil(uniform(start[2], end[2])))


    if collision_check(rand_point.x, rand_point.y, rand_point.th):
        return nodes

    nearest_node = get_closest(nodes, rand_point)
    new_node = Node(steer(nearest_node, rand_point)[0], steer(nearest_node, rand_point)[1],
                    steer(nearest_node, rand_point)[2])  # todo check if need rrt-connect

    if collision_check(new_node.x, new_node.y, new_node.th):
        return nodes

    if len(nodes) % 100 == 0:
        print("{} Nodes Searched".format(len(nodes)))

    new_node.cost = nearest_node.cost + dist(nearest_node, new_node)
    new_node.parent = nearest_node
    nn, new_node = chooseParent(nearest_node, new_node, nodes)

    nodes.append(new_node)
    nodes.sort(key=operator.attrgetter('cost'))
    nodes = reWire(nodes, new_node)
    return nodes




WORLD_SIZE = 499
Goal_bias = 0.1
RADIUS = 9
velocity = 5



#[rand_point.x,rand_point.y,rand_point.th],[nearest_node.x,nearest_node.y,nearest_node.th]

closedlist = []

def rrtstar(start, end):
    nodes_start= []
    nodes_goal = []
    nodes_start.append(Node(start[0],start[1],start[2]))
    nodes_goal.append(Node(end[0], end[1], end[2]))

    # goal = Node(end[0],end[1],end[2])
    # step_th = (end[2]-start[2])/10
    # end_region = Region(Point(Goal[0], Goal[1],Goal[2]), 2, 1)
    i = len(nodes_goal)
    j = len(nodes_goal)
    while True:


        if i<=j:
            nodes_start = getnewnode(nodes_start,nodes_goal,start,end)
            i = len(nodes_start)
        else:
            nodes_goal = getnewnode(nodes_goal,nodes_start,start,end)
            j = len(nodes_goal)


        if len(nodes_start)>500:
        # if dist(nodes_start[-1],get_closest(nodes_goal,nodes_start[-1]))<0.2:
            path = []
            for i in nodes_start: # plot all nodes
                path.append((i.x,i.y,i.th))
            for i in nodes_goal:
                path.append((i.x, i.y, i.th))

            # view_path(path)

            path = []

            current = nodes_start[-1]
            while current is not None:
                path.append((current.x,current.y,current.th))
                current = current.parent
            path = path[::-1]

            current = get_closest(nodes_goal,nodes_start[-1])
            while current is not None:
                path.append((current.x,current.y,current.th))
                current = current.parent
            return path


# 325 229


def main():

    start, goal = init_collision_checking("sample_maps/2/md_T_6.png")
    # start = (350,39,-numpy.pi/2)
    # goal = (135,224,0)

    starttime = time.clock()
    path = rrtstar(start, goal)
    print(time.clock() - starttime)
    print(path)
    view_path(path)

if __name__ == '__main__':
    main()
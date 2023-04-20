import random
import math
import numpy as np
from numpy.linalg import norm
from numpy.random import rand,uniform
class Node:
    def __init__(self,p,parent=None):
        self.p = p
        self.parent = parent

    # @classmethod
    # def copy(cls,d):
    #     return d

    
class RRTConnect:
    def __init__(self, start, goal, obstacle_list, max_iter=5000, delta=0.01, epsilon=0.5):
        self.start = Node(start)
        self.goal = Node(goal)
        self.obstacle_list = obstacle_list
        self.max_iter = max_iter
        self.delta = delta
        self.epsilon = epsilon
        #strat to end
        self.node_list1 = [self.start]
        #end to start
        self.node_list2 = [self.goal]

    def plan(self):
        for i in range(self.max_iter):
            #random search, generate a point
            if rand(1) < self.epsilon:
                rnd = Node(rand(3))
            else:
                rnd = self.goal

            nearest1 = self.nearest(self.node_list1, rnd)
            nearest2 = self.nearest(self.node_list2, rnd)

            new_node1 = self.steer(nearest1, rnd)
            new_node2 = self.steer(nearest2, rnd)

            if self.check_collision(new_node1, self.obstacle_list):
                self.node_list1.append(new_node1)

            if self.check_collision(new_node2, self.obstacle_list):
                self.node_list2.append(new_node2)

            if self.is_same_node(new_node1, new_node2):
                return self.merge_path(new_node1, new_node2)
            #
            p1 = self.nearest(self.node_list1, self.goal)
            p2 = self.nearest(self.node_list2, self.start)
            print("iteration",i,norm(p1.p-p2.p))

        return None

    def steer(self, from_node, to_node):
        # dist = math.sqrt((to_node.x - from_node.x)**2 + (to_node.y - from_node.y)**2)
        dist = norm(to_node.p-from_node.p)
        if dist > self.delta:
            p_ = from_node.p +self.delta/dist*(to_node.p-from_node.p)
            return Node(p_,from_node)
        else:
            return Node(to_node.p,from_node)

    def nearest(self, node_list, node):
        nearest_node = node_list[0]
        # min_dist = math.sqrt((node.x - nearest_node.x)**2 + (node.y - nearest_node.y)**2)
        min_dist = norm(node.p-nearest_node.p)
        for n in node_list:
            dist = norm(node.p-n.p)
            if dist < min_dist:
                min_dist = dist
                nearest_node = n
        return nearest_node

    def check_collision(self, node, obstacle_list):
        for obs in obstacle_list:
            if norm(node.p-np.array([obs[0],obs[1],obs[2]])) <= obs[3]:
                return False
        return True

    def is_same_node(self, node1, node2):
        dist = norm(node1.p-node2.p)
        return dist < self.delta

    def merge_path(self, node1, node2):
        path1 = self.get_path(node1)
        path1.reverse()
        path2 = self.get_path(node2)
        return path1 + path2

    def get_path(self, node):
        path = []
        while node.parent is not None:
            path.append(node.p)
            node = node.parent
        path.append(node.p)
        return path




import matplotlib.pyplot as plt
def plot_globe(a,b,c,r,dense=10):
    '''
    球心:(a,b,c)
    半径:r
    球面的点:(x,y,z)
    '''
    t1=np.linspace(0,np.pi,dense)
    t2=np.linspace(0,np.pi*2,dense)
    t1,t2=np.meshgrid(t1,t2)
    x=a+r*np.sin(t1)*np.cos(t2)
    y=b+r*np.sin(t1)*np.sin(t2)
    z=c+r*np.cos(t1)
    return x,y,z 


# Example usage
start = np.array((0, 0,0))
goal = np.array((1, 1,1))
obstacle_list = [(0.5, 0.5,0.5, 0.1),
                 (0.3,0.7,0.5,0.1),
                 (0.7,0.3,0.5,0.1)]

rrt = RRTConnect(start, goal, obstacle_list)
path = rrt.plan()


fig=plt.figure()
ax=plt.axes(projection="3d") 

if path is not None:
    
    ax.plot([node[0] for node in path], [node[1] for node in path],[node[2] for node in path], '-r')
    ax.plot(start[0], start[1],start[2], 'bo')
    ax.plot(goal[0], goal[1],goal[2], 'bo')
    for obs in obstacle_list:
        x,y,z =plot_globe(obs[0],obs[1],obs[2],obs[3])
        ax.plot_surface(x,y,z,color="k",alpha=0.5)
    plt.show()

else:
    print("No feasible path found.")

import random
import math
import sys
sys.path.append("/home/holmes/Desktop/graduation/code/graduation_simulation_code")
import utils.plot_utils as plut

class Node:
    def __init__(self, x, y,parent=None):
        self.x = x
        self.y = y
        self.parent = parent

class RRTConnect:
    def __init__(self, start, goal, obstacle_list, max_iter=1000, delta=0.05, epsilon=0.001):
        self.start = Node(start[0], start[1])
        self.goal = Node(goal[0], goal[1])
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
            #random search
            if random.random() > self.epsilon:
                rnd = Node(random.uniform(0, 1), random.uniform(0, 1))
            else:
                rnd = Node(self.goal.x, self.goal.y)

            nearest1 = self.nearest(self.node_list1, rnd)
            nearest2 = self.nearest(self.node_list2, rnd)

            new_node1 = self.steer(nearest1, rnd)
            new_node2 = self.steer(nearest2, rnd)
            N = 10.0
            not_trap = True
            for i in range(0,int(N)+1):
                n =Node((N-i)/N*new_node1.x+i/10.0*nearest1.x,(N-i)/N*new_node1.y+i/10.0*nearest1.y)
                if self.check_collision(n, self.obstacle_list):
                    not_trap =False
                    break
            if not_trap:
                self.node_list1.append(new_node1)
            not_trap = True
            for i in range(0,int(N)+1):
                n =Node((N-i)/N*new_node2.x+i/10.0*nearest2.x,(N-i)/N*new_node2.y+i/10.0*nearest2.y)
                if self.check_collision(n, self.obstacle_list):
                    not_trap =False
                    break
            if not_trap:
                self.node_list2.append(new_node2)

            if self.is_same_node(new_node1, new_node2):
                return self.merge_path(new_node1, new_node2)

        return None

    def steer(self, from_node, to_node):
        dist = math.sqrt((to_node.x - from_node.x)**2 + (to_node.y - from_node.y)**2)
        if dist > self.delta:
            theta = math.atan2(to_node.y - from_node.y, to_node.x - from_node.x)
            x = from_node.x + self.delta * math.cos(theta)
            y = from_node.y + self.delta * math.sin(theta)
            return Node(x, y,from_node)
        else:
            return Node(to_node.x,to_node.y,from_node)

    def nearest(self, node_list, node):
        nearest_node = node_list[0]
        min_dist = math.sqrt((node.x - nearest_node.x)**2 + (node.y - nearest_node.y)**2)
        for n in node_list:
            dist = math.sqrt((node.x - n.x)**2 + (node.y - n.y)**2)
            if dist < min_dist:
                min_dist = dist
                nearest_node = n
        return nearest_node

    def check_collision(self, node, obstacle_list):
        for obs in obstacle_list:
            if math.sqrt((node.x - obs[0])**2 + (node.y - obs[1])**2) <= obs[2]:
                return True
        return False

    def is_same_node(self, node1, node2):
        dist = math.sqrt((node1.x - node2.x)**2 + (node1.y - node2.y)**2)
        return dist < self.delta

    def merge_path(self, node1, node2):
        path1 = self.get_path(node1)
        path1.reverse()
        path2 = self.get_path(node2)
        return self.node_list1,self.node_list2,path1,path2,path1 + path2

    def get_path(self, node):
        path = []
        while node.parent is not None:
            path.append([node.x, node.y])
            node = node.parent
        path.append([node.x, node.y])
        return path




f_path = '/home/holmes/Desktop/graduation/hitsz_paper/pictures/'
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
# Example usage
start = (0, 0)
goal = (1, 1)
obstacle_list = [(0.4, 0.4, 0.2),
                 (0.3,0.8,0.2),
                 (0.8,0.3,0.2)]

rrt = RRTConnect(start, goal, obstacle_list)
t_s,t_e,path_s,path_e,path = rrt.plan()
fig, ax = plt.subplots()
pad = 0.1
low = 0 -pad
up = 1+pad

if path is not None:
    ax.plot([node.x for node in t_s], [node.y for node in t_s], 'oy',label='起点树节点')
    ax.plot([node.x for node in t_e], [node.y for node in t_e], '*g',label='终点树节点')
    ax.plot([node[0] for node in path_s], [node[1] for node in path_s], 'or')
    ax.plot([node[0] for node in path_e], [node[1] for node in path_e], '*r')
    ax.plot([node[0] for node in path], [node[1] for node in path], '-r',label='全局轨迹')
    ax.plot(start[0], start[1], 'bo')
    ax.plot(goal[0], goal[1], 'bo')
    for obs in obstacle_list:
        circle = plt.Circle((obs[0], obs[1]), obs[2]-0.1, color='k')
        plt.gcf().gca().add_artist(circle)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(0.2))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(0.2))
    ax.xaxis.set_minor_locator(ticker.MultipleLocator(0.02))
    ax.yaxis.set_minor_locator(ticker.MultipleLocator(0.02))
    ax.set_xlabel('x方向距离(m)')
    ax.set_ylabel('y方向距离(m)')
    ax.axis([low,up]*2)
    ax.legend(loc=0)
    fig.savefig(f_path+"kd_rrt/Figure_3.pdf",pad_inches=0.005)
    plt.show()

else:
    print("No feasible path found.")

import random
import math

class Node:
    def __init__(self, x, y,parent=None):
        self.x = x
        self.y = y
        self.parent = parent

class RRTConnect:
    def __init__(self, start, goal, obstacle_list, max_iter=1000, delta=0.1, epsilon=0.1):
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

            if self.check_collision(new_node1, self.obstacle_list):
                self.node_list1.append(new_node1)

            if self.check_collision(new_node2, self.obstacle_list):
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
                return False
        return True

    def is_same_node(self, node1, node2):
        dist = math.sqrt((node1.x - node2.x)**2 + (node1.y - node2.y)**2)
        return dist < self.delta

    def merge_path(self, node1, node2):
        path1 = self.get_path(node1)
        path1.reverse()
        path2 = self.get_path(node2)
        return path1 + path2

    def get_path(self, node):
        path = []
        while node.parent is not None:
            path.append([node.x, node.y])
            node = node.parent
        path.append([node.x, node.y])
        return path




import matplotlib.pyplot as plt

# Example usage
start = (0, 0)
goal = (1, 1)
obstacle_list = [(0.5, 0.5, 0.1),
                 (0.3,0.7,0.1),
                 (0.7,0.3,0.1)]

rrt = RRTConnect(start, goal, obstacle_list)
path = rrt.plan()

if path is not None:
    plt.plot([node[0] for node in path], [node[1] for node in path], '-r')
    
    plt.plot(start[0], start[1], 'bo')
    plt.plot(goal[0], goal[1], 'bo')
    for obs in obstacle_list:
        circle = plt.Circle((obs[0], obs[1]), obs[2], color='k')
        plt.gcf().gca().add_artist(circle)
    plt.grid()
    plt.axis('scaled')
    plt.show()

else:
    print("No feasible path found.")

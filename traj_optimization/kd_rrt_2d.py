import numpy as np
from numpy.linalg import norm
from numpy.random import rand,uniform


class Node:
    def __init__(self,p,parent=None):
        self.p = p
        self.parent = parent


class RRTConnect:
    def __init__(self, start, goal, obstacle_list, max_iter=100000, delta=0.01, epsilon=0.5):
        self.start = Node(start)
        self.goal = Node(goal)
        self.obstacle_list = obstacle_list
        self.max_iter = max_iter
        self.delta = delta
        self.epsilon = epsilon
        self.dt = 1
        #strat to end
        self.node_list1 = [self.start]
        #end to start
        self.node_list2 = [self.goal]

    def plan(self):
        for i in range(self.max_iter):
            #random search, generate a point
            if rand(1) < self.epsilon:
                rnd = Node(rand(4))
            else:
                rnd = self.goal
            #the code can be improved
            #just to generate the state
            nearest1 = self.nearest(self.node_list1, rnd)
            new_node1 = self.steer(nearest1, rnd,1)
            if self.check_collision(new_node1, self.obstacle_list):
                self.node_list1.append(new_node1)
                nearest1_ = self.nearest(self.node_list2, new_node1)
                if self.is_same_node(new_node1, nearest1_):
                    return self.merge_path(new_node1, nearest1_)
            #here 
            nearest2 = self.nearest(self.node_list2, rnd)
            new_node2 = self.steer(nearest2, rnd,-1)
            if self.check_collision(new_node2, self.obstacle_list):
                self.node_list2.append(new_node2)
                nearest2_ = self.nearest(self.node_list1,new_node2)
                if self.is_same_node(new_node2, nearest2_):
                    return self.merge_path(new_node2, nearest2_)
            #
            p1 = self.nearest(self.node_list1, self.goal)
            p2 = self.nearest(self.node_list2, self.start)
            # print("iteration",i,min(norm(new_node1.p-nearest1_.p),norm(new_node2.p-nearest2_.p)))
            print("iteration",i,norm(p1.p[:2]-p2.p[:2]),norm(p1.p[2:]-p2.p[2:]))


        return None

    def steer(self, from_node, to_node,direct):
        # dist = math.sqrt((to_node.x - from_node.x)**2 + (to_node.y - from_node.y)**2)
        #加入动力学模型
        ddsit = norm(to_node.p[2:]-from_node.p[2:])/self.dt
        if direct:
            dist = norm(to_node.p[:2]-from_node.p[:2]-from_node.p[2:]*self.dt)
        else:
            dist = norm(from_node.p[:2]-to_node.p[:2]-to_node.p[2:]*self.dt)

        if dist > 1e-6 or ddsit>self.dt*0.1:
            #we have some problems beacuse of the iteration
            if direct:
                p_ = np.hstack([from_node.p[:2] +self.dt*from_node.p[2:],
                                from_node.p[2:] +self.dt*0.1*np.sign(to_node.p[2:]-from_node.p[2:])])
            else:#backward,from_node and to_node is already inverse
                p_ = np.hstack([from_node.p[:2] -self.dt*from_node.p[2:]-pow(self.dt,2)*10*np.sign(to_node.p[2:]-from_node.p[2:]),
                                from_node.p[2:] +self.dt*0.1*np.sign(to_node.p[2:]-from_node.p[2:])])
            return Node(p_,from_node)
        else:
            return Node(to_node.p,from_node)

    def nearest(self, node_list, node):
        nearest_node = node_list[0]
        # min_dist = math.sqrt((node.x - nearest_node.x)**2 + (node.y - nearest_node.y)**2)
        min_dist = norm(node.p-nearest_node.p)
        #travers
        for n in node_list:
            dist = norm(node.p-n.p)
            if dist < min_dist:
                min_dist = dist
                nearest_node = n
        return nearest_node

    def check_collision(self, node, obstacle_list):
        for obs in obstacle_list:
            if norm(node.p[:2]-np.array([obs[0],obs[1]])) <= obs[2]:
                return False
        return True

    def is_same_node(self, node1, node2):
        dist = norm(node1.p[:2]-node2.p[:2])
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

#kd-rrt 2d

import matplotlib.pyplot as plt

# Example usage
# obstacle_list = [(0.5, 0.5,0.5, 0.1), (0.3,0.7,0.5,0.1),(0.7,0.3,0.5,0.1)]
# rrt = RRTConnect(start, goal, obstacle_list)
# path = rrt.plan()
# fig=plt.figure()
# ax=plt.axes(projection="3d") 
# # ax.set_xlabel()
# # ax.set_ylabel()
# if path is not None:
    
#     ax.plot([node[0] for node in path], [node[1] for node in path],[node[2] for node in path], '-r')
#     ax.plot(start[0], start[1],start[2], 'bo')
#     ax.plot(goal[0], goal[1],goal[2], 'bo')
#     for obs in obstacle_list:
#         x,y,z =plot_globe(obs[0],obs[1],obs[2],obs[3])
#         ax.plot_surface(x,y,z,color="k",alpha=0.5)
#     plt.show()

# else:
#     print("No feasible path found.")


# Example usage
start = np.array((0, 0,0,0))
goal = np.array((1, 1,0,0))
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

#pattern generator
#foothold related 
#swing related
#contact related
#moion related
#output the reference of something
#local planner is a litte complicated, However, this part is what make sure the intelligence of the robot
#small brain or main brain?
import numpy as np
import sys
sys.path.append("/home/holmes/Data/python/wbc_controller")
# print(sys.path)
import matplotlib.pyplot as plt
from quadprog import solve_qp
from numpy.linalg import matrix_rank as rank,inv
from traj_optimization.traj import traj_opt,traj_opt_regular,body_traj_show
from math import sqrt
import local_planner_conf as conf
from copy import deepcopy



def nt(t):
    return np.array([pow(t,5),pow(t,4),pow(t,3),pow(t,2),t,1]) 
def dnt(t):
    return np.array([5*pow(t,4),4*pow(t,3),3*pow(t,2),2*t,1,0]) 
def ddnt(t):
    return np.array([20*pow(t,3),12*pow(t,2),6*t,2,0,0]) 


def mnt(t):
    return np.block([
        [nt(t),np.zeros(6),np.zeros(6)],
        [np.zeros(6),nt(t),np.zeros(6)],
        [np.zeros(6),np.zeros(6),nt(t)]])


def mdnt(t):
    return np.block([
        [dnt(t),np.zeros(6),np.zeros(6)],
        [np.zeros(6),dnt(t),np.zeros(6)],
        [np.zeros(6),np.zeros(6),dnt(t)]])

 
def mddnt(t):
    return np.block([
        [ddnt(t),np.zeros(6),np.zeros(6)],
        [np.zeros(6),ddnt(t),np.zeros(6)],
        [np.zeros(6),np.zeros(6),ddnt(t)]])


def mT(t):
    return np.vstack([nt(0),dnt(0),ddnt(0),nt(t),dnt(t),ddnt(t)])

def T_all(t):
    return np.vstack([nt(t),dnt(t),ddnt(t)])
"""
    for swing foot motion, we get some assumption like this
    1. the velocity and accerlation of the start point and end point are zero
    2. the velocity of the apex in the trajectory is set to [1.,0.,0.],suppose the robot just move in x direction
    3. for now the robot can not jump because the z direction
"""

"""
    for swing foot motion, we get some assumption like this
    1. the velocity and accerlation of the start point and end point are zero
    2. the velocity of the apex in the trajectory is set to [1.,0.,0.],suppose the robot just move in x direction
"""
def traj_2seg_spline(p_s,p_e,p_m,T):
    # important for illness or infeasible problems
    Q = np.diag([*[1,1,100000],*[1]*3,*[1]*3,*[1000,1000,1000000],*[1]*3,*[1]*3])
    P_ = np.vstack([mnt(T/2),mdnt(T/2),mddnt(T/2),mnt(T),mdnt(T),mddnt(T)])
    q_ = np.hstack([p_m,np.zeros(3),np.zeros(3),p_e,np.zeros(3),np.zeros(3)])
    # equality constraints for start point
    A = np.vstack([mnt(0),mdnt(0),mddnt(0)],dtype=np.float64)
    b = np.hstack([p_s,np.zeros(3),np.zeros(3)],dtype=np.float64)
    # standard qp form
    P = P_.T@Q@P_  
    q = -P_.T@Q@q_
    xf = solve_qp(P,q,None,None,A=A,b=b,solver='quadprog')
    if xf is None:
        raise ValueError("QP solver failed")
    # print('------sol----------')
    # print(np.linalg.norm(P_@xf-q_))
    # print(np.linalg.norm(A@xf-b))
    return xf

def swing_foot_plan(p_s,p_e,T):
    z = 0.02
    A = mT(T)
    A_inv = inv(A)
    Az = mT(T/2.0)
    Az_inv = inv(Az)                  
    coeff0 = A_inv[:,0]*p_s[0]+A_inv[:,3]*p_e[0]
    coeff1 = A_inv[:,0]*p_s[1]+A_inv[:,3]*p_e[1]
    coeff2_1 = Az_inv[:,3]*z
    coeff2_2 = Az_inv[:,0]*z
    return coeff0,coeff1,np.hstack([coeff2_1,coeff2_2])

def swing_foot_traj_get(t,T,coeff0,coeff1,coeff2):
    x = T_all(t)@coeff0
    y = T_all(t)@coeff1
    z = T_all(t)@coeff2[:6] if t < T/2 else T_all(t-T/2)@coeff2[6:]
    return np.array([x[0],y[0],z[0]]),np.array([x[1],y[1],z[1]]),np.array([x[2],y[2],z[2]])

def plot_convex_shape(vertices_, color='k'):
    """Plot a convex shape given its vertices using matplotlib."""
    vertices = deepcopy(vertices_)
    for i in range(len(vertices)-1,-1,-1):
        if vertices[i] is None:
            vertices.pop(i)
    num_vertices = len(vertices)
    x = [vertices[i][0] for i in range(num_vertices)]
    y = [vertices[i][1] for i in range(num_vertices)]
    # plt.fill(x, y, color=color)
    x.append(vertices[0][0])  # Add the first vertex to close the shape
    y.append(vertices[0][1])  # Add the first vertex to close the shape
    plt.plot(x, y, color=color)

def plot_convex_quiver(vertices,edge=None, color='k'):
    """Plot a convex shape and its normal vector"""
    num_vertices = len(vertices)
    x = [vertices[i][0] for i in range(num_vertices)]
    y = [vertices[i][1] for i in range(num_vertices)]
    x.append(vertices[0][0])  # Add the first vertex to close the shape
    y.append(vertices[0][1])  # Add the first vertex to close the shape
    midx = [(x[i]+x[i+1])/2.0 for i in range(len(x)-1)]
    midy = [(y[i]+y[i+1])/2.0 for i in range(len(y)-1)]
    plt.plot(x, y, color=color)
    if edge is not None:
        plt.quiver(midx,midy,edge[:,0],edge[:,1],color='k')






def reduce_convex(polygon_set,s=0.05,w=0.025):
    """reduce the shape of the support polygons"""
    def calcualte_p(num,modified_vertex,origin_vertex):
        c = np.zeros(num)
        vec_n = np.zeros((num,2))
        for i in range(num):
            direct = origin_vertex[(i + 1) % num] - origin_vertex[i]
            normal = np.array([direct[1], -direct[0]])#inside polygons direct
            normal /= np.linalg.norm(normal)
            modified_vertex[i] +=s*normal
            vec_n[i] = normal
            c[i] = -normal@modified_vertex[i]
            # modified_vertex[(i + 1) % num] +=s*normal
        for i in range(num):
            modified_vertex[(i+1)%num] = -np.linalg.inv(vec_n[[i,(i+1)% num]])@c[[i,(i+1)% num]]
        reduce_polygon.append((modified_vertex,duration))
        edge.append(np.hstack([vec_n,c.reshape(1,-1).T]))

    #
    polygons_ = deepcopy(polygon_set)
    reduce_polygon = []
    edge = []
    #get the normal vector and the bias of each edge
    for [vertex,duration] in polygons_:
        vertex_ = []
        new_vertex = []
        for k in vertex:
            if k is not None:
                new_vertex.append(k.copy())
                vertex_.append(k.copy())
        num_vertices = len(new_vertex)
        if num_vertices>2:
            calcualte_p(num_vertices,new_vertex,vertex_)
        else:
            direct = new_vertex[1] - new_vertex[0]
            direct /=np.linalg.norm(direct)
            normal = np.array([direct[1], -direct[0]])#inside polygons direct
            normal /= np.linalg.norm(normal)
            v1 = new_vertex[0]+(s+w)*normal
            v2 = new_vertex[0]-(s+w)*normal
            v3 = new_vertex[1]-(s+w)*normal
            v4 = new_vertex[1]+(s+w)*normal
            ver_ = [v1,v2,v3,v4]
            calcualte_p(4,deepcopy(ver_),ver_)
    return reduce_polygon,edge


class local_planner:
    contact = [True,True,True,True]# [FL,FR, RL,RR] follow this sequence, True is in contact, False means in swing phase
    start_stand = [True,True,True,True]
    start_swing = [False,False,False,False]
    #change the force by the phase
    def __init__(self,conf,T):
        self.stance_phase = 0.75# the phase of stance 
        self.swing_phase = 0.25# the time of swing 
        # self.lift_off = np.array([0.05,0.55,0.7,0.2])# old
        self.lift_off = np.array([0.7,0.2,0.05,0.55])# the lift off event time!
        self.touch_down = self.lift_off+self.swing_phase
        self.touch_down *=T
        self.lift_off *=T
        self.stance_phase *=T# the phase of stance 
        self.swing_phase *=T# the time of swing 
        self.T_gait = T
        self.conf = conf
        self.cur = 0.0
        self.phase = np.zeros(4)#  when in contact the phase decreasing for 0.75* T_gait to 0, when in swing the phase increasing for 0 to 0.25*T_gait seconds 
        self.next_foot =np.zeros((4,2))
        self.cur_foot = np.zeros((4,2))
        self.coeff0 = np.zeros((4,6))
        self.coeff1 = np.zeros((4,6))
        self.coeff2 = np.zeros((4,12))
        self.dim  = 2   
        return
    
    
    
    #exist bugs about the duration between the head and the end
    def update(self,T,foot,hip,v_ref,v_hip):
        self.cur = T%self.T_gait
        for i in range(4):
            if self.cur > self.lift_off[i] and self.cur < self.touch_down[i]:
                self.contact[i] = False
                self.start_stand[i] = True
                self.phase[i] = (self.cur - self.lift_off[i])#in time (second)
            else:
                self.contact[i] = True
                self.start_swing[i] = True
                self.phase[i] = (self.T_gait-self.cur+self.lift_off[i]) \
                    if self.cur > self.lift_off[i] else (self.lift_off[i]-self.cur)
                #get current foot step
                self.cur_foot[i] = foot[i][:2]
            if self.start_stand[i] :
                #do some thing        
                self.start_stand[i]  = False

            if self.start_swing[i] :
                #do some thing
                self.coeff0[i],self.coeff1[i],self.coeff2[i] = swing_foot_plan(foot[i],self.next_foot[i],self.swing_phase)
                self.start_swing[i]  = False
                self.cur_foot[i] = self.next_foot[i]
                    # predict next foot hold for all foot
        # self.next_foot = self.cur_foot + np.array([hip[0]+v_ref[0]*self.stance_phase/2.0,hip[1]+v_ref[1]*self.stance_phase/2.0])


    def in_contact(self,leg):
        return self.contact[leg]
    

    def swing_foot_traj(self,leg,samle_t):
        t = samle_t%self.T_gait- self.lift_off[leg]
        T = self.swing_phase
        coeff0 = self.coeff0[leg]
        coeff1 = self.coeff1[leg]
        coeff2 = self.coeff2[leg]
        x = T_all(t)@coeff0
        y = T_all(t)@coeff1
        z = T_all(t)@coeff2[:6] if t < T/2 else T_all(t-T/2)@coeff2[6:]
        return np.array([x[0],y[0],z[0]]),np.array([x[1],y[1],z[1]]),np.array([x[2],y[2],z[2]])

        
    #event based
    def get_support_polygon(self,foot):
        #this only used in the first start point
        #foot also is necessary
        t = self.cur
        a = list(self.lift_off)
        b = list(self.touch_down)
        for m in range(len(a)):
            if a[m]<t:
                a[m] +=self.T_gait-t
            else:
                a[m] -=t
            if b[m]<t:
                b[m]+=self.T_gait-t
            else:
                b[m] -=t
        t = 0
        foot_ = []
        for m in range(4):
            if self.in_contact(m):
                foot_.append(foot[m])
            else:
                foot_.append(None)
        origin_a = a.copy()
        origin_b = b.copy()
        support_polygon = []
        #make sure the change of next_foot in other function will not change the value here
        #because it contains all the numpy array data structure
        next_foot_ =self.next_foot.copy()
        #
        while len(a)>0 or len(b)>0:
            if len(a)>0 and len(b)>0:
                a_min = min(a)
                b_min = min(b)
                if a_min < b_min :
                    index_ = origin_a.index(a_min)
                    a.remove(a_min )
                    dt = a_min-t
                    t = a_min
                    support_polygon.append([deepcopy(foot_),dt])
                    foot_[index_] = None
                else:
                    index_ = origin_b.index(b_min)
                    b.remove(b_min)
                    dt = b_min-t
                    t = b_min
                    support_polygon.append([deepcopy(foot_),dt])
                    foot_[index_] = next_foot_[index_,:]
            elif len(a)>0:
                a_min = min(a)
                index_ = origin_a.index(a_min)
                a.remove(a_min )
                dt = a_min-t
                t = a_min
                support_polygon.append([deepcopy(foot_),dt])
                foot_[index_] = None
            elif len(b)>0:
                b_min = min(b)
                index_ = origin_b.index(b_min)
                b.remove(b_min)
                dt = b_min-t
                t = b_min
                support_polygon.append([deepcopy(foot_),dt])
                foot_[index_] = next_foot_[index_,:]
        support_polygon.append([deepcopy(foot_),self.T_gait-t])
        #change the order of the foot sequence inorde to draw the polygons
        for item in support_polygon:
            temp = item[0][2]
            item[0][2] = item[0][3]
            item[0][3] = temp
        return deepcopy(support_polygon)


    def print(self):
        n = self.contact_num()
        print(str(self.cur)+"\n",n,self.contact,self.phase)
    

    def body_traj_plan(self,stp,dstp,ddstp,final_pos,edge,support_polygon,shrink_polygon):
        self.duration=[support_polygon[j][1] for j in range(len(support_polygon))]
        self.traj_tot_time = sum(self.duration)
        self.cum_duration = np.cumsum(self.duration)
        coeff_regular =traj_opt_regular(self.duration,stp,dstp,ddstp,final_pos)
        r_coeff =traj_opt(self.duration,stp,dstp,ddstp,final_pos,edge,coeff_regular)
        self.coeff = coeff_regular
        body_traj_show(self.duration,support_polygon,shrink_polygon,2,self.coeff)

        
    def body_traj_update(self,sample_t):
        dim  = self.dim 
        p = np.zeros(2)
        v = np.zeros(2)
        a = np.zeros(2)
        if sample_t < self.traj_tot_time :
            for i in range(1,len(self.cum_duration)):
                if sample_t < self.cum_duration[i]:
                    time = sample_t- self.cum_duration[i-1]
                    for k in range(2):
                        p[k] =   nt(time)@self.coeff[i*6*dim+k*6:i*6*dim+(k+1)*6]
                        v[k] =  dnt(time)@self.coeff[i*6*dim+k*6:i*6*dim+(k+1)*6]
                        a[k] = ddnt(time)@self.coeff[i*6*dim+k*6:i*6*dim+(k+1)*6]
                    return p,v,a
        else:
            # otherwise the final point
            i = len(self.cum_duration)-1
            time = self.traj_tot_time- self.cum_duration[-2]
            for k in range(2):
                p[k] =   nt(time)@self.coeff[i*6*dim+k*6:i*6*dim+(k+1)*6]
            return p,v,a



def print_each_support_polygon(polys_1,polys_2,edge=None):
        for i in range(len(polys_1)):
            plt.figure()
            plt.title("polygon_"+str(i))
            plot_convex_shape(polys_1[i][0])
            if edge is not None:
                plot_convex_quiver(polys_2[i][0],edge[i],'r')
            else:
                plot_convex_quiver(polys_2[i][0],None,'r')
            plt.grid()
            # plt.xlim([-0.2,0.2])
            # plt.ylim([-0.2,0.2])


def print_all_support_polygon(polys_1,polys_2,edge=None):
        plt.figure()
        for i in range(len(polys_1)):
            plot_convex_shape(polys_1[i][0])
            if edge is not None:
                plot_convex_quiver(polys_2[i][0],edge[i],'r')
            else:
                plot_convex_quiver(polys_2[i][0],None,'r')
            plt.grid()
            # plt.xlim([-0.2,0.2])
            # plt.ylim([-0.2,0.2])

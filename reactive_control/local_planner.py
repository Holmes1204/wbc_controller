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
sys.path.append("/home/holmes/Desktop/graduation/code/graduation_simulation_code")
# print(sys.path)
import matplotlib.pyplot as plt
from quadprog import solve_qp
from numpy.linalg import matrix_rank as rank,inv
from traj_optimization.traj import traj_opt,traj_show,traj_opt_regular
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






def reduce_convex(polygon_set,s=0.025,w=0.025):
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
    contact = [True,True,True,True]# [FL,FR, RR,RL] follow this sequence, True is in contact, False means in swing phase
    first_stand = [True,True,True,True]
    first_swing = [False,False,False,False]
    #change the force by the phase
    def __init__(self,conf,T):
        self.stance_phase = 0.75# the phase of stance 
        self.swing_phase = 0.25# the time of swing 
        self.lift_off = np.array([0.05,0.55,0.2,0.7])# the lift off event time!
        self.touch_down = self.lift_off+self.swing_phase
        self.touch_down *=T
        self.lift_off *=T
        self.stance_phase *=T# the phase of stance 
        self.swing_phase *=T# the time of swing 
        self.T = T
        self.conf = conf
        self.cur = 0.0
        self.phase = np.zeros(4)#  when in contact the phase decreasing for 0.75 to 0, when in swing the phase increasing for 0 to 0.25 seconds have 
        self.next_foot =np.zeros((4,2))
        self.coeff0 = np.zeros((4,6))
        self.coeff1 = np.zeros((4,6))
        self.dim  = 2
        self.coeff2 = np.zeros((4,12))
        self.traj_time = 0
        return
    
    
    
    #exist bugs about the duration between the head and the end
    """
    foot: np.array, shape(3,4)
    """
    def update(self,dt,foot,hip,v_ref,v_hip):
        for i in range(4):
            if self.cur > self.lift_off[i] and self.cur < self.touch_down[i]:
                if self.first_swing[i] :
                    #do some thing
                    #2. get the coefficient of the swing trajectory 
                    #plan
                    self.coeff0[i],self.coeff1[i],self.coeff2[i] = swing_foot_plan(foot[i],self.next_foot[i],self.swing_phase)
                    self.first_swing[i]  = False
                self.contact[i] = False
                self.first_stand[i] = True
                self.phase[i] = (self.cur - self.lift_off[i])#in time (second)

            else:
                self.contact[i] = True
                if self.first_stand[i] :
                    #do some thing        
                    #1. get the next foothold
                    #plan
                    self.first_stand[i]  = False
                self.first_swing[i] = True
                self.next_foot[i] = np.array([hip[i][0]+v_ref[0]*self.stance_phase/2.0,hip[i][1]+v_ref[1]*self.stance_phase/2.0])
                self.phase[i] = (self.T-self.cur+self.lift_off[i]) \
                    if self.cur > self.lift_off[i] else (self.lift_off[i]-self.cur)
        self.cur +=dt
        if self.cur>self.T:
            self.cur = 0.0


    def in_contact(self,leg):
        return self.contact[leg]
    
    def contact_num(self):
        n = 4
        for i in self.contact:
            if not i:
                n-=1
        return n

    def swing_foot_traj(self,leg):
        p,v,a = swing_foot_traj_get(self.phase[leg],self.swing_phase,self.coeff0[leg],self.coeff1[leg],self.coeff2[leg])
        return p,v,a
        
    #event based
    def get_support_polygon(self,foot,next_foot):
        #this only used in the first start point
        t = 0.0
        a = list(self.lift_off)
        b = list(self.touch_down)
        origin_a = a.copy()
        origin_b = b.copy()
        support_polygon = []
        foot_ = list(foot.copy())
        next_foot_ =next_foot.copy()
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
        support_polygon.append([deepcopy(foot_),self.T-t])
        return deepcopy(support_polygon)


    def print(self):
        n = self.contact_num()
        print(str(self.cur)+"\n",n,self.contact,self.phase)
    

    def body_traj_plan(self):
        self.duration =[0.05,0.15,0.10,0.15,0.10,0.15,0.10,0.15,0.05]
        self.cum_duration = np.cumsum(self.duration)
        self.traj_tot_time = sum(self.duration)
        #traj_opt(n_seg,dim,duration,stp,dstp,ddstp,fp,p=None,v=None,a=None):
        self.coeff = traj_opt_regular(self.duration,[0,0],[0,0],[0,0],[0.2,.0])
        self.traj_time = 0
        
    def body_traj_update(self,dt):
        dim  = self.dim 
        p = np.zeros(2)
        v = np.zeros(2)
        a = np.zeros(2)
        if self.traj_time < self.traj_tot_time :
            for i in range(1,len(self.cum_duration)):
                if self.traj_time < self.cum_duration[i]:
                    time = self.traj_time- self.cum_duration[i-1]
                    for k in range(2):
                        p[k] =   nt(time)@self.coeff[i*6*dim+k*6:i*6*dim+(k+1)*6]
                        v[k] =  dnt(time)@self.coeff[i*6*dim+k*6:i*6*dim+(k+1)*6]
                        a[k] = ddnt(time)@self.coeff[i*6*dim+k*6:i*6*dim+(k+1)*6]
                    self.traj_time +=dt
                    return p,v,a
        #follow the time splice and get the reference  

    def body_traj_show(self):
        traj_show(self.duration,self.dim,self.coeff)


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


if __name__ == "__main__":
    import time
    N = int(conf.T_SIMULATION/conf.dt)+1  
    PRINT_N = int(conf.PRINT_T/conf.dt)
    t = 0.0
    #
    T = 1
    local_plan  = local_planner (conf,T)
    foot = None
    hip = None
    v_ref= None
    v_hip= None
    discrete_time = np.array(range(N))*conf.dt
    #
    n_dim = 2
    pb   = np.zeros((n_dim,N))
    dpb  = np.zeros((n_dim,N))
    ddpb = np.zeros((n_dim,N))
    foot_data   = np.zeros((4,n_dim,N))
    #
    p = np.zeros(n_dim)
    v = np.zeros(n_dim)
    a = np.zeros(n_dim)
    p0 = np.array([0.0,0.])
    v0 = np.array([0.5,0.])
    p_end = np.array([1.,0.])
    #
    p = p0
    v = v0
    #
    kp = 100
    kd = 2*sqrt(kp)
    time_start = time.time()
    hip = np.zeros((4,2))
    #FL 0
    #FR 1 
    #RR 3
    #RL 4
    bias = np.array([[0.2,0.1],
                     [0.2,-0.1],
                     [-0.2,-0.1],
                     [-0.2,0.1]])
    
    for i in range(4):
        hip[i,:] = p+bias[i,:]
    
    foot = hip.copy()
    h = 0.3
    g = 9.8
    for ss in range(N):
        #record the data
        pb[:,ss] = p
        dpb[:,ss] = v
        ddpb[:,ss] = a
        foot_data[:,:,ss]=foot.copy()
        #get some point
        for i in range(4):
                hip[i,:] = p+bias[i,:]
        #
        local_plan.update(conf.dt,foot,hip,v,v)
        for j in range(4) :
            if local_plan.in_contact(j):
                pass 
            else:
                foot[j,:] = local_plan.swing_foot_traj(j)[:2]


        #1. plot the convex polygon , done
        if ss == 0:
            support_polygon = local_plan.get_support_polygon(foot,local_plan.next_foot)
            shrink_polygon,edge = reduce_convex(support_polygon)
            #2. add the ZMP dynamic Model and get the coefficients
            #  


        #simplified model,use pd forward controller that can follow the traj
        a = kp*(p_end-p)+kd*(-v)
        p += v*conf.dt
        v += a*conf.dt
        t += conf.dt

        
    print("time eslaped",time.time()-time_start)

    print_each_support_polygon(support_polygon,shrink_polygon,edge)
    print_all_support_polygon(support_polygon,shrink_polygon)
    # figure p
    plt.figure()
    plt.plot(discrete_time,pb[0],label='p_x')
    plt.plot(discrete_time,pb[1],label='p_y')
    plt.grid()
    plt.legend()
    #figure v
    plt.figure()
    plt.plot(discrete_time,dpb[0],label='v_x')
    plt.plot(discrete_time,dpb[1],label='v_y')
    plt.grid()
    plt.legend()
    #figure a
    plt.figure()
    plt.plot(discrete_time,ddpb[0],label='a_x')
    plt.plot(discrete_time,ddpb[1],label='a_y')
    plt.grid()
    plt.legend()
    #figure 2
    plt.figure()
    plt.plot(pb[0],pb[1],label='pos')
    plt.grid()
    plt.legend()
    #foot
    plt.figure()
    plt.plot(discrete_time,foot_data[0,0],label='FL_f_x')
    plt.plot(discrete_time,foot_data[0,1],label='FL_f_y')
    plt.plot(discrete_time,foot_data[1,0],label='FR_f_x')
    plt.plot(discrete_time,foot_data[1,1],label='FR_f_y')
    plt.plot(discrete_time,foot_data[2,0],label='RR_f_x')
    plt.plot(discrete_time,foot_data[2,1],label='RR_f_y')
    plt.plot(discrete_time,foot_data[3,0],label='RL_f_x')
    plt.plot(discrete_time,foot_data[3,1],label='RL_f_y')
    plt.grid()
    plt.legend()
    #
    plt.figure()
    plt.plot(foot_data[0,0],foot_data[0,1],label='FL_foot')
    plt.grid()
    plt.legend()
    plt.show()



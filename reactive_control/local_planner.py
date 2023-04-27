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
sys.path.append("../")
# print(sys.path)
import matplotlib.pyplot as plt
from quadprog import solve_qp
from numpy.linalg import matrix_rank as rank,inv
from traj_optimization.traj import traj_opt,traj_show

def nt(t):
    return np.array([pow(t,5),pow(t,4),pow(t,3),pow(t,2),t,1]) 
def mnt(t):
    return np.block([
        [nt(t),np.zeros(6),np.zeros(6)],
        [np.zeros(6),nt(t),np.zeros(6)],
        [np.zeros(6),np.zeros(6),nt(t)]])

def dnt(t):
    return np.array([5*pow(t,4),4*pow(t,3),3*pow(t,2),2*t,1,0]) 

def mdnt(t):
    return np.block([
        [dnt(t),np.zeros(6),np.zeros(6)],
        [np.zeros(6),dnt(t),np.zeros(6)],
        [np.zeros(6),np.zeros(6),dnt(t)]])


def ddnt(t):
    return np.array([20*pow(t,3),12*pow(t,2),6*t,2,0,0]) 

def mddnt(t):
    return np.block([
        [ddnt(t),np.zeros(6),np.zeros(6)],
        [np.zeros(6),ddnt(t),np.zeros(6)],
        [np.zeros(6),np.zeros(6),ddnt(t)]])

"""
    for swing foot motion, we get some assumption like this
    1. the velocity and accerlation of the start point and end point are zero
    2. the velocity of the apex in the trajectory is set to [1.,0.,0.],suppose the robot just move in x direction
"""
def traj_2seg_spline(p_s,p_e,T,v):
    
    p_m  = 0.5*(p_s + p_e)+np.array([0.,0.,0.03])
    dp_m =np.array([2.0*v[0],0.,0.])

    dp_s =np.zeros(3)
    dp_e =np.zeros(3)

    ddp_e =np.zeros(3)
    ddp_s =np.zeros(3)
    ddp_m =np.zeros(3)
    
    A = np.vstack([mnt(0),mdnt(0),mddnt(0),mnt(T),mdnt(T),mddnt(T)])
    A_ = np.block([[A,np.zeros(A.shape)],[np.zeros(A.shape),A]])
    b = np.hstack([p_s,dp_s,ddp_s,p_m,dp_m,ddp_m,p_m,dp_m,ddp_m,p_e,dp_e,ddp_e])
    G = A_.T@A_
    h = A_.T@b
    xf, f, xu, iters, lagr, iact = solve_qp(G,h)
    return xf


class local_planner:
    time_factor = 1.0 # the time of a period of motion
    stance_phase = 0.75# the phase of stance 
    swing_phase = 0.25# the time of swing 
    contact = [True,True,True,True]# [FL,FR, RL,RR] follow this sequence, True is in contact, False means in swing phase
    phase = [0,0,0,0]#  when in contact the phase decreasing for 0.75 to 0, when in swing the phase increasing for 0 to 0.25 seconds have 
    current_phi = 0#contact schedule phase 
    lift_off = [0.05,0.55,0.7,0.2]# the lift off event time!
    first_stand = [True,True,True,True]
    first_swing = [False,False,False,False]
    next_foot =np.zeros((4,3))
    swing_coeff = np.zeros((4,36))
    #change the force by the phase
    def __init__(self,conf):
        self.touch_down = [self.lift_off[i] + self.swing_phase for i in range(4)]#the touch down event time
        self.conf = conf
        return
    
    
    
    #exist bugs about the duration between the head and the end
    """
    foot: np.array, shape(3,4)
    """
    def update(self,dt,foot,hip,v):
        self.current_phi +=dt/self.time_factor
        if self.current_phi > 1.0:
            self.current_phi = 0.0
        for i in range(4):
            if self.current_phi > self.lift_off[i] and self.current_phi < self.touch_down[i]:
                if self.first_swing[i] :
                    #do some thing
                    #2. get the coefficient of the swing trajectory 
                    self.swing_coeff[i] = traj_2seg_spline(foot[i],self.next_foot[i],self.swing_phase*self.time_factor*0.5,v)
                    #plan
                    self.first_swing[i]  = False
                self.contact[i] = False
                self.first_stand[i] = True
                self.phase[i] = (self.current_phi - self.lift_off[i])*self.time_factor#in time (second)
            else:
                self.contact[i] = True
                if self.first_stand[i] :
                    #do some thing        
                    #1. get the next foothold
                    #plan
                    self.first_stand[i]  = False
                self.first_swing[i] = True
                self.next_foot[i] = np.array([hip[i][0]+1.5*v[0]*self.stance_phase*self.time_factor/2.0,hip[i][1]+v[1]*self.stance_phase*self.time_factor/2.0,foot[i][2]])
                self.phase[i] = (self.time_factor-self.current_phi+self.lift_off[i])*self.time_factor \
                    if self.current_phi > self.lift_off[i] else (self.lift_off[i]-self.current_phi)*self.time_factor

    def in_contact(self,leg):
        return self.contact[leg]
    
    def contact_num(self):
        n = 4
        for i in self.contact:
            if not i:
                n-=1
        return n

    def swing_foot_traj_pos(self,leg):
        if(self.phase[leg]<self.swing_phase*self.time_factor*0.5):
            return mnt(self.phase[leg])@self.swing_coeff[leg,:18]
        else:
            return mnt(self.phase[leg]%(self.swing_phase*self.time_factor*0.5))@self.swing_coeff[leg,18:]
        

    def swing_foot_traj_vel(self,leg):
        if(self.phase[leg]<self.swing_phase*self.time_factor*0.5):
            return mdnt(self.phase[leg])@self.swing_coeff[leg,:18]
        else:
            return mdnt(self.phase[leg]%(self.swing_phase*self.time_factor*0.5))@self.swing_coeff[leg,18:]
        


    def swing_foot_traj_acc(self,leg):
        if(self.phase[leg]<self.swing_phase*self.time_factor*0.5):
            return mddnt(self.phase[leg])@self.swing_coeff[leg,:18]
        else:
            return mddnt(self.phase[leg]%(self.swing_phase*self.time_factor*0.5))@self.swing_coeff[leg,18:]
        


    def print(self):
        n = self.contact_num()
        print(str(self.current_phi)+"\n",n,self.contact,self.phase)
    

    def body_traj_plan(self):
        self.duration =10*[0.05,0.15,0.10,0.15,0.10,0.15,0.10,0.15,0.05]
        self.cum_duration = np.cumsum(self.duration)
        self.traj_tot_time = sum(self.duration)
        self.dim  = 2
        #traj_opt(n_seg,dim,duration,stp,dstp,ddstp,fp,p=None,dp=None,ddp=None):
        self.coeff = traj_opt(self.duration,[0,0],[0,0],[0,0],[2.5,.0])
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



if __name__ == "__main__":
    contact = local_planner()
    dt = 0.001
    foot = np.zeros((4,3))
    for i in range(1000):
        # contact.update(dt,foot)
        # contact.swing_foot_traj_pos(0)
        # contact.swing_foot_traj_vel(0)
        # contact.print()
        pass

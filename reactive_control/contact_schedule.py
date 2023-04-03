import numpy as np
from numpy import nan
from numpy.linalg import inv
from numpy.linalg import pinv
from numpy.linalg import matrix_rank as rank
from numpy.linalg import norm
import matplotlib.pyplot as plt
import time
from math import sqrt

#pattern generator
class contact_schedule:
    time_factor = 1.0 # the time of a period of motion
    stance_phase = 0.75# the phase of stance 
    swing_phase = 0.25# the time of swing 
    contact = [True,True,True,True]# [FL,FR, RL,RR] follow this sequence, True is in contact, False means in swing phase
    phase = [0,0,0,0]#  when in contact the phase decreasing for 0.75 to 0, when in swing the phase increasing for 0 to 0.25 
    current_phi = 0#contact schedule tiem 
    lift_off = [0.05,0.2,0.55,0.7]# the lift off event time

    #change the force by the phase
    def __init__(self):
        self.touch_down = [self.lift_off[i] + self.swing_phase for i in range(4)]#the touch down event time
        return
    #exist bugs about the duration between the head and the end
    def update(self,dt):
        self.current_phi +=dt/self.time_factor
        if self.current_phi > 1.0:
            self.current_phi = 0.0
        for i in range(4):
            if self.current_phi > self.lift_off[i] and self.current_phi < self.touch_down[i]:
                self.contact[i] = False
                self.phase[i] = (self.current_phi - self.lift_off[i])/self.time_factor
            else:
                self.contact[i] = True
                self.phase[i] = (self.time_factor-self.current_phi+self.lift_off[i])/self.time_factor \
                    if self.current_phi > self.lift_off[i] else (self.lift_off[i]-self.current_phi)/self.time_factor

    def in_contact(self,leg):
        return self.contact[leg]
    
    def contact_num(self):
        n = 4
        for i in self.contact:
            if not i:
                n-=1
        return n
    
    def print(self):
        n = self.contact_num()
        print(str(self.current_phi)+"\n",n,self.contact,self.phase)
    
if __name__ == "__main__":

    contact = contact_schedule()
    dt = 0.001
    for i in range(1000):
        contact.update(dt)
        contact.print()
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 18 09:47:07 2019

@author: student
"""

import numpy as np
import os
from math import sqrt

np.set_printoptions(precision=3, linewidth=200, suppress=True)
LINE_WIDTH = 60

# q0 = np.array([0.0, 0.67, -1.3, -0.0, 0.67, -1.3, 0.0, 0.67, -1.3, -0.0, 0.67, -1.3])# fixed body frame
q0 = np.array([0.0,0.0,0.31836983483860026,0.0,0.0,0.0,1.0,0.0, 0.67, -1.3, -0.0, 0.67, -1.3, 0.0, 0.67, -1.3, -0.0, 0.67, -1.3])# floating base
T_SIMULATION = 2           # simulation time 
dt = 0.001                   # simulate the controller,control time 1khz
ndt = 20                      # number of integration steps for each control loop dt/ndt is the real  simulation time,20 times sampling

kp = 10                # proportional gain of end effector task
kd = 2*sqrt(kp)         # derivative gain of end effector task
kp_j = 1                 # proportional gain of end effector task
kd_j = 2*sqrt(kp)         # derivative gain of end effector task
FL_foot = 'FL_foot'    
FR_foot = 'FR_foot'    
RL_foot = 'RL_foot'    
RR_foot = 'RR_foot'    
Foot_frame = [FL_foot,FR_foot,RL_foot,RR_foot]
FL_hip = 'FL_hip'    
FR_hip = 'FR_hip'    
RL_hip = 'RL_hip'    
RR_hip = 'RR_hip'  
Hip_frame = [FL_hip,FR_hip,RL_hip,RR_hip]
# PARAMETERS OF REFERENCE SINUSOIDAL TRAJECTORY

simulate_coulomb_friction = 0    # flag specifying whether coulomb friction is simulated
simulation_type = 'euler' # either 'timestepping' or 'euler'
# tau_coulomb_max = 0*np.ones(12)   # expressed as percentage of torque max for fixed trunk
tau_coulomb_max = 0*np.ones(18)   # expressed as percentage of torque max for floating trunk

ground_pos = np.array([0,0,0])
ground_normal = np.array([0,0,1])
g_kp = 1e7
g_kd = 2*sqrt(g_kp)
ground_Kp = np.diag(np.ones(3)*g_kp)
ground_Kd = np.diag(np.ones(3)*g_kd)
ground_mu = 0.5

randomize_robot_model = 0
model_variation = 30.0

use_viewer = 2#0 None, 1 Gepetto, 2 Rviz
simulate_real_time = False          # flag specifying whether simulation should be real time or as fast as possible
show_floor = False
PRINT_T = 0.01                  # print some info every PRINT_T seconds
DISPLAY_T = 0.0005              # update robot configuration in viwewer every DISPLAY_T seconds
CAMERA_TRANSFORM = [2.582354784011841, 1.620774507522583, 1.0674564838409424, 
                    0.2770655155181885, 0.5401807427406311, 0.6969326734542847, 0.3817386031150818]

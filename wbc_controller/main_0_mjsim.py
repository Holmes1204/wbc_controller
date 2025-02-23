import numpy as np
from numpy import nan
from numpy.linalg import inv,pinv,norm,matrix_rank as rank
import matplotlib.pyplot as plt
import time
from math import sqrt
import sys
sys.path.append("/home/holmes/Data/python/wbc_controller")
from utils.robot_wrapper import RobotWrapper
import main_1_conf as conf
import solutions.main_1_solution as solution

import pinocchio as pin
# from local_planner import local_planner,reduce_convex
from solutions.WBC_HO_qp import task,WBC_HO


import mujoco
import mujoco.viewer
mj_model = mujoco.MjModel.from_xml_path('/home/holmes/Data/python/wbc_controller/mujoco_sim/mujoco_menagerie/unitree_a1/scene.xml')
mj_data = mujoco.MjData(mj_model)
mj_model.opt.timestep = conf.dt/conf.ndt
#  heading x positive direction, lifting  left front leg
pin_init_pos = np.array([0.0,0.0,0.32,0.0,0.0,0.0,1.0,0.0,0.67, -1.3, -0.0, 0.67, -1.3, 0.0, 0.67, -1.3, -0.0, 0.67, -1.3])
pin_init_j_pos = pin_init_pos[7:]

import math
print("".center(conf.LINE_WIDTH,'#'))
print(" Quadrupedal Robot".center(conf.LINE_WIDTH, '#'))
print("".center(conf.LINE_WIDTH,'#'), '\n')

PLOT_EE_POS = 0
PLOT_BODY_POS = 0
PLOT_BODY_VEL = 0
PLOT_BODY_ACC = 0
PLOT_DOG_JOINT_POS = 0
PLOT_DOG_TORQUES = 0


rmodel, rcollision_model, rvisual_model = pin.buildModelsFromUrdf("./a1_description/urdf/a1.urdf", ".",pin.JointModelFreeFlyer())
robot = RobotWrapper(rmodel, rcollision_model, rvisual_model)   
# simu = RobotSimulator(conf, robot)
# local_plan  = local_planner(conf,1)

# simu.add_contact_surface("ground",conf.ground_pos,conf.ground_normal, 
#                          conf.ground_Kp,conf.ground_Kd,conf.ground_mu)
# [simu.add_candidate_contact_point(foot) for foot in conf.Foot_frame]
# simu.add_candidate_contact_point("trunk")

##can keep adding frames, to make the robot stand
nx, ndx = 3, 3
N = 10*int(conf.T_SIMULATION/conf.dt)      # number of time steps
tau     = np.empty((robot.na, N))*nan    # joint torques
q       = np.empty((robot.nq, N+1))*nan  # joint angles
v       = np.empty((robot.nv, N+1))*nan  # joint velocities
# dv      = np.empty((robot.nv, N+1))*nan  # joint accelerations
fx   = np.empty((nx,  N))*nan        # end-effector reference position
fx_des = np.empty((nx,  N))*nan        # end-effector reference position
traj_bp  = np.empty((ndx, N))*nan        # end-effector reference velocity
traj_dbp = np.empty((ndx, N))*nan        # end-effector reference acceleration
traj_ddbp = np.empty((ndx, N))*nan        # end-effector desired acceleration



t = 0.0
kp, kd = conf.kp, conf.kd
PRINT_N = int(conf.PRINT_T/conf.dt)


# Frction cone and reaction modulation, Dx <= f
f_max = 50
max_torque = 33.5
foot_mu = conf.ground_mu
FRC_ = np.array([[ 1, 0,-foot_mu],
               [-1, 0,-foot_mu],
               [ 0, 1,-foot_mu],
               [ 0,-1,-foot_mu],
               [ 0, 0,-1],
               [ 0, 0, 1]])
frc_ =np.array([0,0,0,0,0,f_max])

def to_mj(q):
    """
    Convert from mujoco to pinocchio
    pin = [x,y,z,w]
    mj = [w,x,y,z]
    """
    q_ = q.copy()
    quat = q[3:7]# x,y,z,w
    q_[3:7] = quat[[3,0,1,2]] # w,x,y,z
    q_[7:] = q[7:][[3,4,5,0,1,2,9,10,11,6,7,8]]
    return q_


def to_mj_v(v):
    v_ = v.copy()
    v_[6:] = v[6:][[3,4,5,0,1,2,9,10,11,6,7,8]]
    return v_


def to_mj_tau(tau):
    return tau[[3,4,5,0,1,2,9,10,11,6,7,8]]

def to_pin(q):
    """
    Convert from mujoco to pinocchio
    pin = [x,y,z,w]
    mj = [w,x,y,z]
    """
    q_ = q.copy()
    quat = q[3:7]#w,xyz
    q_[3:7] = quat[[1,2,3,0]]
    q_[7:] = q[7:][[3,4,5,0,1,2,9,10,11,6,7,8]]
    return q_


def to_pin_v(v):
    v_ = v.copy()
    v_[6:] = v[6:][[3,4,5,0,1,2,9,10,11,6,7,8]]
    return v_


ss = 0
mj_data.qpos[:] = to_mj(pin_init_pos)
mj_data.qvel[:] = np.zeros(robot.nv)


# gait pattern & trajectory generator, the prdefined contact sequence
# whole body controller tracking the gait related trajectory
data_to_lot1 = np.zeros((3,10000))
data_to_lot2 = np.zeros((3,10000))
n_c = 4 
n_u = 18

with mujoco.viewer.launch_passive(mj_model, mj_data) as viewer:
  start = time.time()
  while viewer.is_running() and time.time() - start < 100:
    time_start = time.time()
    # state feedback and its value  check before simulation
    q = to_pin(mj_data.qpos)
    v = to_pin_v(mj_data.qvel)

    # compute mass matrix M, bias terms h, gravity terms g
    robot.computeAllTerms(q, v)
    M = robot.mass(q)
    h = robot.nle(q, v)#include gravity force
    # g = robot.gravity(q)
    #
    # simulation is really important for all types of robot, by using gazebo and webot and somethign else ss don't konw the dynamics of the robot
    # this is not so good for the real world


    p_h = np.zeros((4,3))
    v_h = np.zeros((4,3))
    J_bp = np.zeros((3,18))
    J_bR = np.zeros((3,18))
    dJdq_bp = np.zeros(3)
    dJdq_bR = np.zeros(3)
    J_c = np.zeros((12,18))
    dJdq_c = np.zeros(12)
    p_f = np.zeros((4,3))
    v_f = np.zeros((4,3))
    zero_acc = np.zeros(18)
    j_contact = 0
    tasks = []
    # state feedback and Jacobian and djacobian@dq

    for j in range(len(conf.Foot_frame)) :
        frame_id = robot.model.getFrameId(conf.Foot_frame[j])
        J = robot.frameJacobian(q, frame_id)[:3,:]
        dJdq = robot.frameAcceleration(q, v, zero_acc, frame_id).linear
        H = robot.framePlacement(q, frame_id)
        v_frame = robot.frameVelocity(q, v, frame_id)
        p_f[j,:] = H.translation
        v_f[j,:] = v_frame.linear
        J_c[3*j:3*j+3,:] = J
        dJdq_c[3*j:3*j+3] = dJdq

    for j in range(len(conf.Hip_frame)) :
        frame_id = robot.model.getFrameId(conf.Hip_frame[j])
        H = robot.framePlacement(q, frame_id)
        v_frame = robot.frameVelocity(q, v, frame_id)
        p_h[j,:] = H.translation+np.array([0.0,(-1)**j*0.084,0.0])
        v_h[j,:] = v_frame.linear


    # feedback of body
    frame_id = robot.model.getFrameId("trunk")
    H = robot.framePlacement(q, frame_id)
    x_bp= H.translation # take the 3d position of the end-effector
    x_bR = H.rotation
    v_b = robot.frameVelocity(q, v, frame_id)
    dx_bp = v_b.linear # take linear part of 6d velocity
    dx_bR = v_b.angular
    J_bp = robot.frameJacobian(q, frame_id, False)[:3,:]
    dJdq_bp = robot.frameAcceleration(q, v, zero_acc, frame_id).linear
    J_bR = robot.frameJacobian(q, frame_id, False)[3:,:]
    dJdq_bR = robot.frameAcceleration(q, v, zero_acc, frame_id).angular
    
    
    
    omega = 2*math.pi/2.0
    amp = 0.01
    phi = math.pi/4
    o1 = amp*math.sin(omega*t+phi)
    o2 = amp*omega*math.cos(omega*t+phi)
    o3 = -amp*omega*omega*math.sin(omega*t+phi)
    target_base = np.array([0.0+o1,0.0+o1,0.30 +o1])
    target_bv   = np.array([0.0+o2,0.0+o2, 0.0 +o2])
    target_ba   = np.array([0.0+o3,0.0+o3, 0.0 +o3])
    
    data_to_lot1[:,ss] = x_bp
    data_to_lot2[:,ss] = target_base
    

    n_c =4
    M_f = M[:6,:]
    h_f = h[:6]
    Jc_f = J_c[:,:6]
    M_j = M[6:,:]
    h_j = h[6:]
    
    Jc_j = J_c[:,6:]    # nc
    A1 = np.hstack((M_f,-Jc_f.T))
    b1 = -h_f
    
    D1_1 = np.block([[M_j,-Jc_j.T],
                   [-M_j,Jc_j.T]])
    f1_1 = np.block([-h_j+max_torque*np.ones(12),h_j+max_torque*np.ones(12)])

    D1_2 = np.hstack([np.zeros((6*n_c,n_u)),np.block([[ FRC_,np.zeros((6,3))     ,np.zeros((6,3))      ,np.zeros((6,3))],
                                                    [ np.zeros((6,3))   ,FRC_,np.zeros((6,3)) ,np.zeros((6,3))],
                                                    [np.zeros((6,3))      ,np.zeros((6,3))      ,FRC_,np.zeros((6,3))],
                                                    [np.zeros((6,3))      ,np.zeros((6,3))      ,np.zeros((6,3)),FRC_]])])
    f1_2 = np.block([frc_]*n_c)
    
    D1 = np.vstack([D1_1,D1_2])
    f1 = np.hstack([f1_1,f1_2])
    
    
    A2 = np.hstack([J_c,np.zeros((3*n_c,3*n_c))])
    b2 = -dJdq_c
    
    Kp_bp = 1000
    Kd_bp = 0.01*sqrt(Kp_bp)
    Kp_bR = 5
    Kd_bR = 0.01*sqrt(Kp_bR)
    
    A3_bp = np.hstack([J_bp,np.zeros((3,3*n_c))])
    b3_bp = (target_ba+Kp_bp*(target_base- x_bp) - Kd_bp*(target_bv- dx_bp) ) - dJdq_bp  

    A3_bR = np.hstack([J_bR,np.zeros((3,3*n_c))])
    b3_bR =(Kp_bR*(pin.log3(x_bR.T)) - Kd_bR*(dx_bR) ) - dJdq_bR

    # swint foot trajecotory 
    target_p_f_1 = np.array([0.18,0.128,0.21])
    Kp_f_1 = 1
    kd_f_1 =0.5*sqrt(Kp_f_1)
    
    A3_f_1 =np.hstack([J_c[:3,:],np.zeros((3,3*n_c))])
    b3_f_1 = -dJdq_c[:3] + Kp_f_1*(target_p_f_1 - p_f[0]) - kd_f_1*v_f[0]
    
    A3 = np.vstack([A3_bp,0.5*A3_bR])
    b3 = np.hstack([b3_bp,0.5*b3_bR])

    
    A4 = np.hstack([np.zeros((3*n_c,n_u)),np.eye(3*n_c)])
    b4 = np.zeros(3*n_c)
    # print(A1.shape,b1.shape,D1.shape,f1.shape)
    # print(A2.shape,b2.shape)
    # print(A3.shape,b3.shape)
    # print(A4.shape,b4.shape)
    # tasks.append(task(A1,b1,D1,f1,0))
    # tasks.append(task(A2,b2,None,None,1))
    # tasks.append(task(A3,b3,None,None,2))
    # tasks.append(task(A4,b4,None,None,3))
    tasks.append(task(0,(A1,b1),None)) # dynamic 
    tasks.append(task(1,None,(D1,f1))) # force and friction cone
    tasks.append(task(2,(A2,b2),None)) # no contact motion
    tasks.append(task(3,(A3,b3),None)) # tracking 
    tasks.append(task(4,(A4,b4),None)) # minimize force
    out = WBC_HO(tasks).solve()
    
    tau = M_j@out[:18] + h_j - Jc_j.T@out[18:]
    # ctrl = kp*(pin_init_j_pos-q[7:]) - kd*(v[6:])
    # print('ctrl',ctrl)
    # print('out',tau)
    mj_data.ctrl = to_mj_tau(tau)
    for i in range(conf.ndt):
        mujoco.mj_step(mj_model, mj_data)
    
    print("--------check--------↓")
    err1 = np.linalg.norm(A1@out-b1)
    err2 = np.linalg.norm(A2@out-b2)
    err3 = np.linalg.norm(A3@out-b3)
    err4 = np.linalg.norm(A4@out-b4)
    print("eval eq 1",np.linalg.norm(A1@out-b1))
    print("eval ie 1",max(0,np.max(D1@out-f1)))
    print("eval eq 2",np.linalg.norm(A2@out-b2))
    print("eval eq 3",np.linalg.norm(A3@out-b3))
    print("eval eq 4",np.linalg.norm(A4@out-b4))
    print('force',out[18:].reshape(-1,3))  
    if np.max(D1@out-f1)>0.5 and ss>1000:
        binary_file_path = 'mj_data.npz'
        np.savez(binary_file_path,A1=A1,b1=b1,D1=D1,f1=f1,A2=A2,b2=b2,A3=A3,b3=b3,A4=A4,b4=b4)
        raise ValueError("Infeasible")
    print("--------check--------↑")
    # local_plan.update_phase(conf.dt)
    t += conf.dt
    print('time',t)

    time_spent = time.time() - time_start
    ss += 1
    if ss % 4 ==0:# 
        viewer.sync()
    if t>8.0:
        break

sim_time = np.array(range(ss))*conf.dt
plt.figure()
plt.plot(sim_time,data_to_lot1[0,:ss],label='x')
plt.plot(sim_time,data_to_lot1[1,:ss],label='y')
plt.plot(sim_time,data_to_lot1[2,:ss],label='z')   
# plt.title('1')
# plt.legend()
# plt.figure()
plt.plot(sim_time,data_to_lot2[0,:ss],'--',label='x_des')
plt.plot(sim_time,data_to_lot2[1,:ss],'--',label='y_des')
plt.plot(sim_time,data_to_lot2[2,:ss],'--',label='z_des')
plt.legend()
plt.title('2')
plt.show()
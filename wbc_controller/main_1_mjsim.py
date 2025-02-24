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
from local_planner import local_planner,reduce_convex
from solutions.WBC_HO_qp import task,WBC_HO


import mujoco
import mujoco.viewer
model = mujoco.MjModel.from_xml_path('/home/holmes/Data/python/wbc_controller/unitree_a1/scene.xml')
mj_data = mujoco.MjData(model)
model.opt.timestep = conf.dt/conf.ndt
#  heading x positive direction, lifting  left front leg
pin_init_pos = np.array([0.0,0.0,0.32,0.0,0.0,0.0,1.0,0.0, 0.67, -1.3, -0.0, 0.67, -1.3, 0.0, 0.67, -1.3, -0.0, 0.67, -1.3])
pin_init_j_pos = pin_init_pos[7:]

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
local_plan  = local_planner(conf,1)

# simu.add_contact_surface("ground",conf.ground_pos,conf.ground_normal, 
#                          conf.ground_Kp,conf.ground_Kd,conf.ground_mu)
# [simu.add_candidate_contact_point(foot) for foot in conf.Foot_frame]
# simu.add_candidate_contact_point("trunk")

##can keep adding frames, to make the robot stand
nx, ndx = 3, 3
N = 4000     # number of time steps
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

S = np.zeros((12,18))
S[:,6:]=np.eye(12)
# Frction cone and reaction modulation, Dx <= f
f_max = 50
tau_max = 33.5
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

with mujoco.viewer.launch_passive(model, mj_data,show_left_ui=False,show_right_ui=False) as viewer:
  viewer.cam.azimuth= 90.0
  viewer.cam.distance= 2.0
  viewer.cam.elevation= -10.0
  viewer.cam.lookat= np.array([0., 0., 0.])
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
 

    p_h = np.zeros((4,3))
    v_h = np.zeros((4,3))
    J_bp = np.zeros((3,18))
    J_bR = np.zeros((3,18))
    dJdq_bp = np.zeros(3)
    dJdq_bR = np.zeros(3)
    J_f = np.zeros((12,18))
    dJdq_f = np.zeros(12)
    p_f = np.zeros((4,3))
    v_f = np.zeros((4,3))
    j_contact = 0
    tasks = []
    # state feedback and Jacobian and djacobian@dq
    for j in range(len(conf.Foot_frame)) :
        frame_id = robot.model.getFrameId(conf.Foot_frame[j])
        J = robot.frameJacobian(q, frame_id, False)[:3,:]
        dJdq = robot.frameAcceleration(q, v, None, frame_id, False).linear
        H = robot.framePlacement(q, frame_id, False)
        v_frame = robot.frameVelocity(q, v, frame_id, False)
        p_f[j,:] = H.translation
        v_f[j,:] = v_frame.linear
        J_f[3*j:3*j+3,:] = J
        dJdq_f[3*j:3*j+3] = dJdq

    for j in range(len(conf.Hip_frame)) :
        frame_id = robot.model.getFrameId(conf.Hip_frame[j])
        H = robot.framePlacement(q, frame_id, False)
        v_frame = robot.frameVelocity(q, v, frame_id, False)
        p_h[j,:] = H.translation+np.array([0.0,(-1)**j*0.084,0.0])
        v_h[j,:] = v_frame.linear


    # feedback of body
    frame_id = robot.model.getFrameId("trunk")
    H = robot.framePlacement(q, frame_id, False)
    x_bp= H.translation # take the 3d position of the end-effector
    x_bR = H.rotation
    v_b = robot.frameVelocity(q, v, frame_id, False)
    dx_bp = v_b.linear # take linear part of 6d velocity
    dx_bR = v_b.angular
    J_bp = robot.frameJacobian(q, frame_id, False)[:3,:]
    dJdq_bp = robot.frameAcceleration(q, v, None, frame_id, False).linear
    J_bR = robot.frameJacobian(q, frame_id, False)[3:,:]
    dJdq_bR = robot.frameAcceleration(q, v, None, frame_id, False).angular


    v_ref = np.array([0.1,0.,0.])
    #foot update
    local_plan.update_foot(p_f,p_h,v_ref,v_ref)
    if ss== 0:
        # local_plan.body_traj_show()
        stp = x_bp[:2]
        dstp =dx_bp[:2]
        ddstp =np.zeros(2)
        fp=x_bp[:2]+v_ref[:2]*1
        support_polygon = local_plan.get_support_polygon(p_f[:,:2])
        shrink_polygon,edge = reduce_convex(support_polygon)
        local_plan.body_traj_plan(stp,dstp,ddstp,fp,edge,support_polygon,shrink_polygon)
        
        #存在bug
        # print_each_support_polygon(support_polygon,shrink_polygon,edge)
        # print_all_support_polygon(support_polygon,shrink_polygon)
        # plt.show()

    #planning
    n_contact = local_plan.contact_num()
    J_st = np.zeros((3*n_contact,18))
    dJdq_st = np.zeros(3*n_contact)
    J_sw = np.zeros((3*(4-n_contact),18))
    dJdq_sw = np.zeros(3*(4-n_contact))
    p_sw = np.zeros(3*(4-n_contact))
    dp_sw = np.zeros(3*(4-n_contact))
    p_sw_des = np.zeros(3*(4-n_contact))
    dp_sw_des = np.zeros(3*(4-n_contact))
    ddp_sw_des = np.zeros(3*(4-n_contact))
    D2 = np.zeros((n_contact*FRC_.shape[0],n_contact*FRC_.shape[1]))
    f2 = np.zeros(n_contact*frc_.shape[0])
    #update the all plan information
    # n_contact =4
    if n_contact == 4:
        #first contact
        J_st=J_f
        dJdq_st=dJdq_f
        B_st = np.block([[FRC_,np.zeros((6,3)),np.zeros((6,3)),np.zeros((6,3))],
                         [np.zeros((6,3)),FRC_,np.zeros((6,3)),np.zeros((6,3))],
                         [np.zeros((6,3)),np.zeros((6,3)),FRC_,np.zeros((6,3))],
                         [np.zeros((6,3)),np.zeros((6,3)),np.zeros((6,3)),FRC_]])
        beta_st = np.block([frc_,frc_,frc_,frc_])
        Q,R = np.linalg.qr(J_st.T,'complete')
        R = R[:R.shape[1],:]
        Q_u = Q[:,3*n_contact:]# when n_contact == 0, something tricky will happen
        Q_c = Q[:,:3*n_contact]
        d_spe = (12,18)
        A1 = Q_u.T@np.hstack([-M,S.T])#so important that this can help imporve the efficiency
        b1 = Q_u.T@h
        D1 = np.block([[np.zeros(d_spe), np.eye(12)],
                       [np.zeros(d_spe),-np.eye(12)]])
        f1 = np.block([np.ones(12)*tau_max,np.ones(12)*-(-tau_max)])
        # kexi = solution.WBC_HO(A1,b1,D1,f1)
        A2 = np.hstack([J_st,np.zeros((3*n_contact,12))])
        b2 = -dJdq_st
        D2 = B_st@inv(R)@Q_c.T@np.hstack([M,-S.T])
        f2 = beta_st - B_st@inv(R)@Q_c.T@h
        tasks.append(task(0,(A1,b1),None))
        tasks.append(task(0.5,None,(D1,f1)))
        tasks.append(task(1,(A2,b2),(D2,f2)))
    elif n_contact < 4 and n_contact >0:
        j_st = 0
        j_sw =0
        B_st = np.zeros((n_contact*FRC_.shape[0],n_contact*FRC_.shape[1]))
        beta_st = np.zeros(n_contact*frc_.shape[0])
        for j in range(len(conf.Foot_frame)) :
            if local_plan.in_contact(j):
                J_st[3*j_st:3*j_st+3,:] = J_f[3*j:3*j+3,:]
                dJdq_st[3*j_st:3*j_st+3] = dJdq_f[3*j:3*j+3]
                #in contact must have the force constrints,FL,FR,RL,RR
                B_st[j_st*FRC_.shape[0]:(j_st+1)*FRC_.shape[0],j_st*FRC_.shape[1]:(j_st+1)*FRC_.shape[1]]= FRC_
                beta_st[j_st*frc_.shape[0]:(j_st+1)*frc_.shape[0]]= frc_
                j_st +=1
            else:
                #calculate the A
                J_sw[3*j_sw:3*j_sw+3,:] = J_f[3*j:3*j+3,:]
                dJdq_sw[3*j_sw:3*j_sw+3] =  dJdq_f[3*j:3*j+3]
                p_sw[3*j_sw:3*j_sw+3]=p_f[j]
                dp_sw[3*j_sw:3*j_sw+3] = v_f[j]
                p_sw_des[3*j_sw:3*j_sw+3],dp_sw_des[3*j_sw:3*j_sw+3],ddp_sw_des[3*j_sw:3*j_sw+3] = local_plan.swing_foot_traj(j)
                j_sw +=1
        #some tricky copied 

        # fx[:] = p_f[0,:]
        # if local_plan.in_contact(0):
        #    fx_des[:] = p_f[0,:]
        # else:
        #    fx_des[:],_,_ = local_plan.swing_foot_traj(0)
        #task1
        Q,R = np.linalg.qr(J_st.T,'complete')
        R = R[:R.shape[1],:]
        Q_u = Q[:,3*n_contact:]# when n_contact == 0, something tricky will happen
        Q_c = Q[:,:3*n_contact]
        d_spe = (12,18)
        A1 = Q_u.T@np.hstack([-M,S.T])#so important that this can help imporve the efficiency
        b1 = Q_u.T@h
        D1 = np.block([[np.zeros(d_spe), np.eye(12)],
                       [np.zeros(d_spe),-np.eye(12)]])
        f1 = np.block([np.ones(12)*33.5,np.ones(12)*33.5])
        #task2
        A2 = np.hstack([J_st,np.zeros((3*n_contact,12))])
        b2 = -dJdq_st
        D2 = B_st@inv(R)@Q_c.T@np.hstack([M,-S.T])
        f2 = beta_st - B_st@inv(R)@Q_c.T@h
        #task3
        Kp_sw = 1000
        Kd_sw = 1*sqrt(Kp_sw)
        A4 = np.hstack([J_sw,np.zeros((3*(4-n_contact),12))])
        b4 = -dJdq_sw+Kp_sw*(p_sw_des-p_sw)+Kd_sw*(dp_sw_des-dp_sw)+ddp_sw_des
        
        tasks.append(task(0,(A1,b1),None))
        tasks.append(task(0.5,None,(D1,f1)))
        tasks.append(task(1,(A2,b2),(D2,f2)))
        tasks.append(task(2,(A4,b4),None))
    else:
        J_sw = J_f
        dJdq_sw = dJdq_sw
        A4 = np.hstack([J_sw,np.zeros((3*(4-n_contact),12))])
        b4 = -dJdq_sw+Kp_sw*(p_sw_des-p_sw)+Kd_sw*(dp_sw_des-dp_sw)
        tasks.append(task(2,(A4,b4),None))
        raise ValueError('n_contact is not valid')
#
    Kp_bp = 1000
    Kd_bp = 1*sqrt(Kp_bp)
    kkp = 1
    Kp_bR = 5
    Kd_bR = 2*sqrt(Kp_bR)
    #
    traj_p,traj_dp,traj_ddp = local_plan.body_traj_update(conf.dt)
    # x_bp_des = np.array([traj_p[0],traj_dp[1],0.32])
    # dx_bp_des = np.array([traj_dp[0],traj_dp[1],0])
    # ddx_bp_des = np.array([traj_ddp[0],traj_ddp[1],0])
    x_bp_des = np.array([0.8*(ss/N)*(ss/N)/2.0,0.0,0.32])
    dx_bp_des = np.array([0.8*ss/N,0.0,0])
    ddx_bp_des = np.array([0.8,0,0])
    print(x_bp_des)
    # traj_bp[:]= x_bp_des
    # traj_dbp[:]= dx_bp_des
    # traj_ddbp[:]= ddx_bp_des
    # attitude
    x_bR_des = np.eye(3)
    dx_bR_des = np.array([0.0,0.0,0.0])
    A3 = np.vstack([np.hstack([J_bp,np.zeros((3,12))]),
                    np.hstack([J_bR,np.zeros((3,12))])])
    b3 = np.hstack([-dJdq_bp+kkp*(ddx_bp_des+Kp_bp*(x_bp_des-x_bp)+Kd_bp*(dx_bp_des-dx_bp)),
                    -dJdq_bR+Kp_bR*(pin.log3(x_bR_des.dot(x_bR.T)))+Kd_bR*(dx_bR_des-dx_bR)])

    tasks.append(task(2,(A3,b3),None))
    out = WBC_HO(tasks).solve()

    F = inv(R)@Q_c.T@(M@out[:18]+h-S.T@out[18:])
    
    tau = out[18:]
    # send joint torques to simulator
    mj_data.ctrl = to_mj_tau(tau)
    for i in range(conf.ndt):
        mujoco.mj_step(model, mj_data)
    # print(tau)


    local_plan.update_phase(conf.dt)
    t += conf.dt
    print('time',t)
    ss += 1
    time_spent = time.time() - time_start
    if ss % 4 ==0:# 
        viewer.sync()
    if t>8.0:
        break

sim_time = np.array(range(ss))*conf.dt
plt.figure()
plt.plot(sim_time,data_to_lot1[0,:ss],label='x')
plt.plot(sim_time,data_to_lot1[1,:ss],label='y')
plt.plot(sim_time,data_to_lot1[2,:ss],label='z')   
plt.title('1')
plt.legend()
plt.figure()
plt.plot(sim_time,data_to_lot2[0,:ss],label='x')
plt.plot(sim_time,data_to_lot2[1,:ss],label='y')
plt.plot(sim_time,data_to_lot2[2,:ss],label='z')
plt.legend()
plt.title('2')
# plt.show()
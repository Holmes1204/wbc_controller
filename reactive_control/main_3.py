import numpy as np
from numpy import nan
from numpy.linalg import inv,pinv,norm,matrix_rank as rank
import matplotlib.pyplot as plt
import time
from math import sqrt
import sys
sys.path.append("/home/holmes/Desktop/graduation/code/graduation_simulation_code")
import utils.plot_utils as plut
from utils.robot_loaders import loadUR
from utils.robot_wrapper import RobotWrapper
from utils.robot_simulator import RobotSimulator
import main_1_conf as conf
import solutions.main_1_solution as solution
from example_robot_data.robots_loader import load
import pinocchio as pin
from local_planner import local_planner,reduce_convex
from solutions.WBC_HO import task,WBC_HO


print("".center(conf.LINE_WIDTH,'#'))
print(" Quadrupedal Robot".center(conf.LINE_WIDTH, '#'))
print("".center(conf.LINE_WIDTH,'#'), '\n')

PLOT_EE_POS = 1
PLOT_BODY_POS = 1
PLOT_BODY_VEL = 1
PLOT_BODY_ACC = 1
PLOT_DOG_JOINT_POS = 0
PLOT_DOG_TORQUES = 0


rmodel, rcollision_model, rvisual_model = pin.buildModelsFromUrdf("./a1_description/urdf/a1.urdf", ".",pin.JointModelFreeFlyer())
robot = RobotWrapper(rmodel, rcollision_model, rvisual_model)   
simu = RobotSimulator(conf, robot)
local_plan  = local_planner(conf,1)

simu.add_contact_surface("ground",conf.ground_pos,conf.ground_normal, 
                         conf.ground_Kp,conf.ground_Kd,conf.ground_mu)
[simu.add_candidate_contact_point(foot) for foot in conf.Foot_frame]
simu.add_candidate_contact_point("trunk")

##can keep adding frames, to make the robot stand
nx, ndx = 3, 3
N = int(conf.T_SIMULATION/conf.dt)      # number of time steps
tau     = np.empty((robot.na, N))*nan    # joint torques
q       = np.empty((robot.nq, N+1))*nan  # joint angles
v       = np.empty((robot.nv, N+1))*nan  # joint velocities
dv      = np.empty((robot.nv, N+1))*nan  # joint accelerations
fx   = np.empty((nx,  N))*nan        # end-effector reference position
fx_des = np.empty((nx,  N))*nan        # end-effector reference position
traj_bp  = np.empty((ndx, N))*nan        # end-effector reference velocity
traj_dbp = np.empty((ndx, N))*nan        # end-effector reference acceleration
traj_ddbp = np.empty((ndx, N))*nan        # end-effector desired acceleration


S = np.zeros((12,18))
S[:,6:]=np.eye(12)

t = 0.0
kp, kd = conf.kp, conf.kd
PRINT_N = int(conf.PRINT_T/conf.dt)
#
foot_mu = conf.ground_mu/sqrt(2)
B_ = np.array([[ 1, 0,-foot_mu],
               [-1, 0,-foot_mu],
               [ 0, 1,-foot_mu],
               [ 0,-1,-foot_mu],
               [ 0, 0,-1],
               [ 0, 0, 1]])
beta_ =np.array([0,0,0,0,0,1000])

for ss in range(0, N):#ss: simualtion step
    time_start = time.time()
    v[:,ss] = simu.v
    q[:,ss] = simu.q
    dv[:,ss] =simu.dv
    # compute mass matrix M, bias terms h, gravity terms g
    robot.computeAllTerms(q[:,ss], v[:,ss])
    M = robot.mass(q[:,ss], False)
    h = robot.nle(q[:,ss], v[:,ss], False)#include gravity force
    g = robot.gravity(q[:,ss])
    #
    # simulation is really important for all types of robot, by using gazebo and webot and somethign else ss don't konw the dynamics of the robot
    # this is not so good for the real world


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
        J = robot.frameJacobian(q[:,ss], frame_id, False)[:3,:]
        dJdq = robot.frameAcceleration(q[:,ss], v[:,ss], None, frame_id, False).linear
        H = robot.framePlacement(q[:,ss], frame_id, False)
        v_frame = robot.frameVelocity(q[:,ss], v[:,ss], frame_id, False)
        p_f[j,:] = H.translation
        v_f[j,:] = v_frame.linear
        J_f[3*j:3*j+3,:] = J
        dJdq_f[3*j:3*j+3] = dJdq

    for j in range(len(conf.Hip_frame)) :
        frame_id = robot.model.getFrameId(conf.Hip_frame[j])
        H = robot.framePlacement(q[:,ss], frame_id, False)
        v_frame = robot.frameVelocity(q[:,ss], v[:,ss], frame_id, False)
        p_h[j,:] = H.translation+np.array([0.0,(-1)**j*0.084,0.0])
        v_h[j,:] = v_frame.linear


    # feedback of body
    frame_id = robot.model.getFrameId("trunk")
    H = robot.framePlacement(q[:,ss], frame_id, False)
    x_bp= H.translation # take the 3d position of the end-effector
    x_bR = H.rotation
    v_b = robot.frameVelocity(q[:,ss], v[:,ss], frame_id, False)
    dx_bp = v_b.linear # take linear part of 6d velocity
    dx_bR = v_b.angular
    J_bp = robot.frameJacobian(q[:,ss], frame_id, False)[:3,:]
    dJdq_bp = robot.frameAcceleration(q[:,ss], v[:,ss], None, frame_id, False).linear
    J_bR = robot.frameJacobian(q[:,ss], frame_id, False)[3:,:]
    dJdq_bR = robot.frameAcceleration(q[:,ss], v[:,ss], None, frame_id, False).angular


    v_ref = np.array([0.1,0.,0.])
    #foot update
    local_plan.update_foot(p_f,p_h,v_ref,v_ref)
    if ss%500== 0:
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
    D2 = np.zeros((n_contact*B_.shape[0],n_contact*B_.shape[1]))
    f2 = np.zeros(n_contact*beta_.shape[0])
    #update the all plan information
    # n_contact =4
    if n_contact == 4:
        #first contact
        J_st=J_f
        dJdq_st=dJdq_f
        B_st = np.block([[B_,np.zeros((6,3)),np.zeros((6,3)),np.zeros((6,3))],
                         [np.zeros((6,3)),B_,np.zeros((6,3)),np.zeros((6,3))],
                         [np.zeros((6,3)),np.zeros((6,3)),B_,np.zeros((6,3))],
                         [np.zeros((6,3)),np.zeros((6,3)),np.zeros((6,3)),B_]])
        beta_st = np.block([beta_,beta_,beta_,beta_])
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
        # kexi = solution.WBC_HO(A1,b1,D1,f1)
        A2 = np.hstack([J_st,np.zeros((3*n_contact,12))])
        b2 = -dJdq_st
        D2 = B_st@inv(R)@Q_c.T@np.hstack([M,-S.T])
        f2 = beta_st - B_st@inv(R)@Q_c.T@h
        tasks.append(task(A1,b1,D1,f1,0))
        tasks.append(task(A2,b2,D2,f2,1))
    elif n_contact < 4 and n_contact >0:
        j_st = 0
        j_sw =0
        B_st = np.zeros((n_contact*B_.shape[0],n_contact*B_.shape[1]))
        beta_st = np.zeros(n_contact*beta_.shape[0])
        for j in range(len(conf.Foot_frame)) :
            if local_plan.in_contact(j):
                J_st[3*j_st:3*j_st+3,:] = J_f[3*j:3*j+3,:]
                dJdq_st[3*j_st:3*j_st+3] = dJdq_f[3*j:3*j+3]
                #in contact must have the force constrints,FL,FR,RL,RR
                B_st[j_st*B_.shape[0]:(j_st+1)*B_.shape[0],j_st*B_.shape[1]:(j_st+1)*B_.shape[1]]= B_
                beta_st[j_st*beta_.shape[0]:(j_st+1)*beta_.shape[0]]= beta_
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

        fx[:,ss] = p_f[0,:]
        if local_plan.in_contact(0):
           fx_des[:,ss] = p_f[0,:]
        else:
           fx_des[:,ss],_,_ = local_plan.swing_foot_traj(0)
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
        Kd_sw = 2*sqrt(Kp_sw)
        A4 = np.hstack([J_sw,np.zeros((3*(4-n_contact),12))])
        b4 = -dJdq_sw+Kp_sw*(p_sw_des-p_sw)+Kd_sw*(dp_sw_des-dp_sw)+ddp_sw_des
        
        tasks.append(task(A1,b1,D1,f1,0))
        tasks.append(task(A2,b2,D2,f2,1))
        tasks.append(task(A4,b4,None,None,2))
    else:
        J_sw = J_f
        dJdq_sw = dJdq_sw
        A4 = np.hstack([J_sw,np.zeros((3*(4-n_contact),12))])
        b4 = -dJdq_sw+Kp_sw*(p_sw_des-p_sw)+Kd_sw*(dp_sw_des-dp_sw)
        tasks.append(task(A4,b4,None,None,2))

#
    Kp_bp = 1000
    Kd_bp = 2*sqrt(Kp_bp)
    Kp_bR = 1000
    Kd_bR = 2*sqrt(Kp_bR)
    #
    traj_p,traj_dp,traj_ddp = local_plan.body_traj_update(conf.dt)
    x_bp_des = np.array([traj_p[0],traj_dp[1],0.32])
    dx_bp_des = np.array([traj_dp[0],traj_dp[1],0])
    ddx_bp_des = np.array([traj_ddp[0],traj_ddp[1],0])
    traj_bp[:,ss]= x_bp_des
    traj_dbp[:,ss]= dx_bp_des
    traj_ddbp[:,ss]= ddx_bp_des
    # attitude
    x_bR_des = np.eye(3)
    dx_bR_des = np.array([0.0,0.0,0.0])
    A3 = np.vstack([np.hstack([J_bp,np.zeros((3,12))]),
                    np.hstack([J_bR,np.zeros((3,12))])])
    b3 = np.hstack([-dJdq_bp+ddx_bp_des+Kp_bp*(x_bp_des-x_bp)+Kd_bp*(dx_bp_des-dx_bp),
                    -dJdq_bR+Kp_bR*(pin.log3(x_bR_des.dot(x_bR.T)))+Kd_bR*(dx_bR_des-dx_bR)])

    tasks.append(task(A3,b3,None,None,3))
    out = WBC_HO(tasks).solve()
    
    F = inv(R)@Q_c.T@(M@out[:18]+h-S.T@out[18:])
    

    tau[:,ss] = np.hstack([np.zeros(6),out[18:]])
    # send joint torques to simulator
    simu.simulate(tau[:,ss], conf.dt, conf.ndt)
    # print(tau[:,ss])

    if ss%PRINT_N == 0:
        print("Time %.3f"%(t))
    local_plan.update_phase(conf.dt)
    t += conf.dt
    time_spent = time.time() - time_start


LABEL={0:'x',1:'y',2:'z'}
# PLOT STUFF
time = np.arange(0.0, N*conf.dt, conf.dt)
if(PLOT_EE_POS):    
    (f, ax) = plut.create_empty_figure(2)
    title = "Foot"
    f.suptitle(title, fontsize=16)
    f.canvas.manager.set_window_title(title)
    ax = ax.reshape(2)
    # for i in range(2):
    ax[0].plot(fx[0,:], fx[2,:])
    ax[0].plot(fx_des[0,:], fx_des[2,:])
    ax[1].plot(fx[1,:], fx[2,:])
    ax[1].plot(fx_des[1,:], fx_des[2,:])
    leg = ax[0].legend()
    leg.get_frame().set_alpha(0.5)
    

if(PLOT_BODY_POS):    
    (f, ax) = plut.create_empty_figure(3)
    ax = ax.reshape(3)
    title = "BODY_POS"
    f.suptitle(title, fontsize=16)
    f.canvas.manager.set_window_title(title)
    for i in range(3):
        ax[i].plot(time, q[i,:-1], label='body_pos')
        ax[i].plot(time, traj_bp[i,:], '--', label='ref')
        ax[i].set_xlabel('Time [s]')
        ax[i].set_ylabel(r''+LABEL[i]+' [m]')
    leg = ax[0].legend()
    leg.get_frame().set_alpha(0.5)

if(PLOT_BODY_VEL):    
    (f, ax) = plut.create_empty_figure(3)
    ax = ax.reshape(3)
    title = "BODY_VEL"
    f.suptitle(title, fontsize=16)
    f.canvas.manager.set_window_title(title)
    for i in range(3):
        ax[i].plot(time, v[i,:-1], label='body_vel')
        ax[i].plot(time, traj_dbp[i,:], '--', label='ref')
        ax[i].set_xlabel('Time [s]')
        ax[i].set_ylabel(r''+LABEL[i]+' [m]')
    leg = ax[0].legend()
    leg.get_frame().set_alpha(0.5)

if(PLOT_BODY_ACC):    
    (f, ax) = plut.create_empty_figure(3)
    ax = ax.reshape(3)
    title = "BODY_ACC"
    f.suptitle(title, fontsize=16)
    f.canvas.manager.set_window_title(title)
    for i in range(3):
        ax[i].plot(time, dv[i,:-1], label='body_acc')
        ax[i].plot(time, traj_ddbp[i,:], '--', label='ref')
        ax[i].set_xlabel('Time [s]')
        ax[i].set_ylabel(r''+LABEL[i]+' [m]')
    leg = ax[0].legend()
    leg.get_frame().set_alpha(0.5)

if(PLOT_DOG_JOINT_POS):    
    (f, ax) = plut.create_empty_figure(6,2)
    ax = ax.reshape(12)
    title = "DOG_JOINT_POS"
    f.suptitle(title, fontsize=16)
    f.canvas.manager.set_window_title(title)
    for i in range(12):
        ax[i].plot(time, q[7+i,:-1], label='q')
        ax[i].set_xlabel('Time [s]')
        ax[i].set_ylabel(r'$q_{'+str(i)+'}$ [rad]')
    leg = ax[0].legend()
    leg.get_frame().set_alpha(0.5)

        

if(PLOT_DOG_TORQUES):    
    (f, ax) = plut.create_empty_figure(6,2)
    ax = ax.reshape(12)
    title = "DOG_TORQUES"
    f.suptitle(title, fontsize=16)
    f.canvas.manager.set_window_title(title)
    for i in range(12):
        ax[i].plot(time, tau[6+i,:], label=r'$\tau$ '+str(i))
        ax[i].set_xlabel('Time [s]')
        ax[i].set_ylabel(r'$\tau_{'+str(i)+'}$ [N/m]')
    leg = ax[0].legend()
    leg.get_frame().set_alpha(0.5)

        
plt.show()

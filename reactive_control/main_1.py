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
from local_planner import local_planner
from solutions.WBC_HO import task,WBC_HO


print("".center(conf.LINE_WIDTH,'#'))
print(" Quadrupedal Robot".center(conf.LINE_WIDTH, '#'))
print("".center(conf.LINE_WIDTH,'#'), '\n')

PLOT_EE_POS = 1
PLOT_BODY_POS = 1
PLOT_DOG_JOINT_POS = 0
PLOT_DOG_TORQUES = 0


#load('ur5')
rmodel, rcollision_model, rvisual_model = pin.buildModelsFromUrdf("./a1_description/urdf/a1.urdf", ".",pin.JointModelFreeFlyer())
robot = RobotWrapper(rmodel, rcollision_model, rvisual_model)   
simu = RobotSimulator(conf, robot)
local_plan  = local_planner()
simu.add_contact_surface("ground",conf.ground_pos,conf.ground_normal,
                         conf.ground_Kp,conf.ground_Kd,conf.ground_mu)
[simu.add_candidate_contact_point(foot) for foot in conf.Foot_frame]
simu.add_candidate_contact_point("trunk")

##can keep adding frames, to make the robot stand
nx, ndx = 3, 3
N = int(conf.T_SIMULATION/conf.dt)      # number of time steps
tau     = np.empty((robot.na, N))*nan    # joint torques
tau_c   = np.empty((robot.na, N))*nan    # joint Coulomb torques
q       = np.empty((robot.nq, N+1))*nan  # joint angles
v       = np.empty((robot.nv, N+1))*nan  # joint velocities
dv      = np.empty((robot.nv, N+1))*nan  # joint accelerations
x       = np.empty((nx,  N))*nan        # end-effector position
dx      = np.empty((ndx, N))*nan        # end-effector velocity
ddx     = np.empty((ndx, N))*nan        # end effector acceleration
x_ref   = np.empty((nx,  N))*nan        # end-effector reference position
dx_ref  = np.empty((ndx, N))*nan        # end-effector reference velocity
ddx_ref = np.empty((ndx, N))*nan        # end-effector reference acceleration
ddx_des = np.empty((ndx, N))*nan        # end-effector desired acceleration

fx   = np.empty((nx,  N))*nan        # end-effector reference position
fx_des = np.empty((nx,  N))*nan        # end-effector reference position

two_pi_f             = 2*np.pi*conf.freq   # frequency (time 2 PI)
two_pi_f_amp         = two_pi_f * conf.amp
two_pi_f_squared_amp = two_pi_f * two_pi_f_amp

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
beta_ =np.array([0,0,0,0,0,100])
for ss in range(0, N):#ss: simualtion step
    time_start = time.time()
    # set reference trajectory
    # x_ref[:,ss]  = conf.x0 +  conf.amp*np.sin(two_pi_f*t + conf.phi)
    # dx_ref[:,ss]  = two_pi_f_amp * np.cos(two_pi_f*t + conf.phi)
    # ddx_ref[:,ss] = - two_pi_f_squared_amp * np.sin(two_pi_f*t + conf.phi)
    # x_ref[:,ss]  =  np.array([0.0,0.10,0.32])  
    # dx_ref[:,ss] = np.array([0.0, 0.0, 0.0])  
    # ddx_ref[:,ss] = np.array([0.0, 0.0, 0.0])  
    # read current state from simulator
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
    #
    #control a frame
    # frame_id = 13
    # #
    # H = robot.framePlacement(q[:,ss], frame_id, False)
    # x[:,ss] = H.translation # take the 3d position of the end-effector
    # v_frame = robot.frameVelocity(q[:,ss], v[:,ss], frame_id, False)
    # dx[:,ss] = v_frame.linear # take linear part of 6d velocity
    # # implement your control law here
    # ddx_des[:,ss] = ddx_ref[:,ss] + kp*(x_ref[:,ss] - x[:,ss]) + kd*(dx_ref[:,ss] - dx[:,ss])



    # J6 = robot.frameJacobian(q[:,ss], frame_id, False)
    # J = J6[:3,:]            # take first 3 rows of J6
    # dJdq = robot.frameAcceleration(q[:,ss], v[:,ss], None, frame_id, False).linear
    # dJ6 = robot.frameJacobianTimeVariation(q[:,ss],v[:,ss],frame_id)
    # dJ = dJ6[:3,:]

    J_f = np.zeros((12,18))
    dJdq_f = np.zeros(12)
    p_f = np.zeros((4,3))
    v_f = np.zeros((4,3))
    j_contact = 0
    tasks = []
    #state feedback and Jacobian and djacobian@dq
    for j in range(len(conf.Foot_frame)) :
        frame_id = robot.model.getFrameId(conf.Foot_frame[j])
        J = robot.frameJacobian(q[:,ss], frame_id, False)[:3,:]
        dJdq = robot.frameAcceleration(q[:,ss], v[:,ss], None, frame_id, False).linear
        dJ = robot.frameJacobianTimeVariation(q[:,ss],v[:,ss],frame_id)[:3,:]
        H = robot.framePlacement(q[:,ss], frame_id, False)
        v_frame = robot.frameVelocity(q[:,ss], v[:,ss], frame_id, False)
        p_f[j,:] = H.translation
        v_f[j,:] = v_frame.linear
        J_f[3*j:3*j+3,:] = J
        dJdq_f[3*j:3*j+3] = dJdq
        # if local_plan.in_contact(j):
        #     J_st[3*j:3*j+3,:] = J
        #     dJdq_st[3*j:3*j+3] = dJdq
        # else:
        #     J_sw[3*j:3*j+3,:] = J
        #     dJdq_sw[3*j:3*j+3] = dJdq
    p_h = np.zeros((4,3))
    v_h = np.zeros((4,3))
    for j in range(len(conf.Hip_frame)) :
        frame_id = robot.model.getFrameId(conf.Hip_frame[j])
        H = robot.framePlacement(q[:,ss], frame_id, False)
        v_frame = robot.frameVelocity(q[:,ss], v[:,ss], frame_id, False)
        p_h[j,:] = H.translation+np.array([0.0,(-1)**j*0.084,0.0])
        v_h[j,:] = v_frame.linear

    ##control body pos and orientation
    J_bp = np.zeros((3,18))
    J_bR = np.zeros((3,18))
    dJdq_bp = np.zeros(3)
    dJdq_bR = np.zeros(3)
    frame_id = robot.model.getFrameId("trunk")
    H = robot.framePlacement(q[:,ss], frame_id, False)
    x_bp= H.translation # take the 3d position of the end-effector
    x_bR = H.rotation
    x_bp_des = np.array([x_bp[0],x_bp[1],0.32])
    x_bR_des = np.eye(3)
    v_b = robot.frameVelocity(q[:,ss], v[:,ss], frame_id, False)
    dx_bp = v_b.linear # take linear part of 6d velocity
    dx_bR = v_b.angular
    dx_bp_des = np.array([0.04,0.04,0])
    dx_bR_des = np.array([0.0,0.0,0.0])
    #here is contact
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
                p_sw_des[3*j_sw:3*j_sw+3] = local_plan.swing_foot_traj_pos(j)
                dp_sw_des[3*j_sw:3*j_sw+3] = local_plan.swing_foot_traj_vel(j)
                ddp_sw_des[3*j_sw:3*j_sw+3] = local_plan.swing_foot_traj_acc(j)
                j_sw +=1
        #some tricky copied 

        fx[:,ss] = p_f[0,:]
        if local_plan.in_contact(0):
           fx_des[:,ss] = p_f[0,:]
        else:
           fx_des[:,ss] = local_plan.swing_foot_traj_pos(0)
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

        #task4
        Kp_sw = 1500
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
        tasks.append(task(A4,b4,None,None,3))


    #jacobian
    J = robot.frameJacobian(q[:,ss], frame_id, False)[:3,:]
    dJdq = robot.frameAcceleration(q[:,ss], v[:,ss], None, frame_id, False).linear
    dJ = robot.frameJacobianTimeVariation(q[:,ss],v[:,ss],frame_id)[:3,:]
    J_bp = J
    dJdq_bp = dJdq
    #
    J = robot.frameJacobian(q[:,ss], frame_id, False)[3:,:]
    dJdq = robot.frameAcceleration(q[:,ss], v[:,ss], None, frame_id, False).angular
    dJ = robot.frameJacobianTimeVariation(q[:,ss],v[:,ss],frame_id)[3:,:]
    J_bR = J
    dJdq_bR = dJdq


    #hierarchical optimization
    # kexi = np.zeros(30)
    # A1 = Q_u.T@np.hstack([-M,S.T])#so important that this can help imporve the efficiency
    # b1 = Q_u.T@h
    # D1 = np.block([[np.zeros(d_spe), np.eye(12)],
    #                 [np.zeros(d_spe),-np.eye(12)]])
    # f1 = np.block([np.ones(12)*33.5,np.ones(12)*33.5])

    # #need to add some inequalities
    # Z1 = np.eye(30) - pinv(A1)@A1
    # A2 = np.hstack([J_st,np.zeros((3*n_contact,12))])
    # b2 = -dJdq_st
    # dkexi =solution.WBC_HO(A4,b4,D1,f1)
    # t2 = task(A2,b2,None,None,1)

    #the differnce of the hierarchical settings
#motion tracking of legs
#
#    which leg
#   
    Kp_bp = 10
    Kd_bp = 2*sqrt(Kp_bp)
    Kp_bR = 10
    Kd_bR = 2*sqrt(Kp_bR)
    # Z2 = Z1@(np.eye(30)-pinv(A2@Z1)@A2@Z1)
    A3 = np.vstack([np.hstack([J_bp,np.zeros((3,12))]),
                    np.hstack([J_bR,np.zeros((3,12))])])
    b3 = np.hstack([-dJdq_bp+Kp_bp*(x_bp_des-x_bp)+Kd_bp*(dx_bp_des-dx_bp),
                    -dJdq_bR+Kp_bR*(pin.log3(x_bR_des.dot(x_bR.T)))+Kd_bR*(dx_bR_des-dx_bR)])

    tasks.append(task(A3,b3,None,None,3))
    out = WBC_HO(tasks).solve()
    #test for motion tracking 
    # Z3 = Z2@(np.eye(30)-pinv(A3@Z2)@A3@Z2)
    # Z4 = Z3@(np.eye(30)-pinv(A4@Z3)@A4@Z3)
    F = inv(R)@Q_c.T@(M@out[:18]+h-S.T@out[18:])
    

    tau[:,ss] = np.hstack([np.zeros(6),out[18:]])
    # send joint torques to simulator
    simu.simulate(tau[:,ss], conf.dt, conf.ndt)
    local_plan.update(conf.dt,p_f,p_h,v_b.linear)
    # print(tau[:,ss])
    if ss%PRINT_N == 0:
        print("Time %.3f"%(t))
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

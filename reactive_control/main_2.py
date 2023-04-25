import numpy as np
from numpy import nan
from numpy.linalg import inv,pinv,norm,matrix_rank as rank
import matplotlib.pyplot as plt
import time
from math import sqrt,sin,pi
import sys
sys.path.append("../")
import utils.plot_utils as plut
from utils.robot_wrapper import RobotWrapper
from utils.robot_simulator import RobotSimulator
import main_2_conf as conf
import pinocchio as pin
from reactive_control.local_planner import local_planner 
from solutions.WBC_HO import task,WBC_HO

print("".center(conf.LINE_WIDTH,'#'))       
print(" Quadrupedal Robot".center(conf.LINE_WIDTH, '#'))
print("".center(conf.LINE_WIDTH,'#'), '\n')

PLOT_EE_POS = 1
PLOT_BODY_POS = 1
PLOT_DOG_JOINT_POS = 1
PLOT_ARM_JOINT_POS = 1 
PLOT_DOG_TORQUES = 1
PLOT_ARM_TORQUES = 1

rmodel, rcollision_model, rvisual_model = pin.buildModelsFromUrdf("../a1_description/urdf/a1.urdf", "../",pin.JointModelFreeFlyer())
robot = RobotWrapper(rmodel, rcollision_model, rvisual_model)
simu = RobotSimulator(conf, robot)
local_plan  = local_planner ()
simu.add_contact_surface("ground",conf.ground_pos,conf.ground_normal,
                         conf.ground_Kp,conf.ground_Kd,conf.ground_mu)
[simu.add_candidate_contact_point(foot) for foot in conf.Foot_frame]
simu.add_candidate_contact_point("trunk")

nx, ndx = 3, 3
nv = robot.nv
na = robot.na
nq = robot.nq
nt = conf.nt# number of direct motors
N = int(conf.T_SIMULATION/conf.dt)      # number of time steps
PRINT_N = int(conf.PRINT_T/conf.dt)

t = 0.0
#
#these will  keep the datum
tau     = np.empty((robot.na, N))*nan    # joint torques
q       = np.empty((robot.nq, N+1))*nan  # joint angles
v       = np.empty((robot.nv, N+1))*nan  # joint velocities
dv      = np.empty((robot.nv, N+1))*nan  # joint accelerations
#
x       = np.empty((nx,  N))*nan        # end-effector position
dx      = np.empty((ndx, N))*nan        # end-effector velocity
ddx     = np.empty((ndx, N))*nan        # end effector acceleration
#
mx   = np.empty((nx,  N))*nan        # end-effector reference position
mx_ref   = np.empty((nx,  N))*nan        # end-effector reference position
dx_ref  = np.empty((ndx, N))*nan        # end-effector reference velocity
ddx_ref = np.empty((ndx, N))*nan        # end-effector reference acceleration
ddx_des = np.empty((ndx, N))*nan        # end-effector desired acceleration


#
foot_mu = conf.ground_mu/sqrt(2)
B_ = np.array([[ 1, 0,-foot_mu],
               [-1, 0,-foot_mu],
               [ 0, 1,-foot_mu],
               [ 0,-1,-foot_mu],
               [ 0, 0,-1],
               [ 0, 0, 1]])
beta_ =np.array([0,0,0,0,0,100])
S = np.zeros((nt,nv))
S[:,6:]=np.eye(nt)
#some variables
J_f = np.zeros((12,nv))#leg
dJdq_f = np.zeros(12)
p_f = np.zeros((4,3))
v_f = np.zeros((4,3))
j_contact = 0

##control body pos and orientation
J_bp = np.zeros((3,nv))
J_bR = np.zeros((3,nv))
dJdq_bp = np.zeros(3)
dJdq_bR = np.zeros(3)

for ss in range(0, N):#ss: simualtion step
    time_start = time.time()
    q[:,ss] = simu.q
    v[:,ss] = simu.v
    dv[:,ss] =simu.dv
    # compute mass matrix M, bias terms h, gravity terms g
    robot.computeAllTerms(q[:,ss], v[:,ss])
    M = robot.mass(q[:,ss], False)
    h = robot.nle(q[:,ss], v[:,ss], False)#include gravity force
    g = robot.gravity(q[:,ss])
    #

    #here is contact
    n_contact = local_plan.contact_num()
    J_st = np.zeros((3*n_contact,nv))
    dJdq_st = np.zeros(3*n_contact)
    J_sw = np.zeros((3*(4-n_contact),nv))
    dJdq_sw = np.zeros(3*(4-n_contact))

    p_sw = np.zeros(3*(4-n_contact))
    dp_sw = np.zeros(3*(4-n_contact))
    p_sw_des = np.zeros(3*(4-n_contact))
    dp_sw_des = np.zeros(3*(4-n_contact))
    ddp_sw_des = np.zeros(3*(4-n_contact))

    D2 = np.zeros((n_contact*B_.shape[0],n_contact*B_.shape[1]))
    f2 = np.zeros(n_contact*beta_.shape[0])

    #state feedback and Jacobian and djacobian@dq
    for j in range(len(conf.Foot_frame)) :
        tasks = []
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

    p_h = np.zeros((4,3))
    v_h = np.zeros((4,3))
    for j in range(len(conf.Hip_frame)) :
        frame_id = robot.model.getFrameId(conf.Hip_frame[j])
        H = robot.framePlacement(q[:,ss], frame_id, False)
        v_frame = robot.frameVelocity(q[:,ss], v[:,ss], frame_id, False)
        p_h[j,:] = H.translation+np.array([0.0,(-1)**j*0.084,0.0])
        v_h[j,:] = v_frame.linear

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
        d_spe = (nt,nv)
        A1 = Q_u.T@np.hstack([-M,S.T])#so important that this can help imporve the efficiency
        b1 = Q_u.T@h
        D1 = np.block([[np.zeros(d_spe), np.eye(nt)],
                       [np.zeros(d_spe),-np.eye(nt)]])
        f1 = np.block([np.ones(nt)*33.5,np.ones(nt)*33.5])

        A2 = np.hstack([J_st,np.zeros((3*n_contact,nt))])
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
        #task1
        Q,R = np.linalg.qr(J_st.T,'complete')
        R = R[:R.shape[1],:]
        Q_u = Q[:,3*n_contact:]# when n_contact == 0, something tricky will happen
        Q_c = Q[:,:3*n_contact]
        d_spe = (nt,nv)
        A1 = Q_u.T@np.hstack([-M,S.T])#so important that this can help imporve the efficiency
        b1 = Q_u.T@h
        D1 = np.block([[np.zeros(d_spe), np.eye(nt)],
                       [np.zeros(d_spe),-np.eye(nt)]])
        f1 = np.block([np.ones(nt)*33.5,np.ones(nt)*33.5])
        #task2
        A2 = np.hstack([J_st,np.zeros((3*n_contact,nt))])
        b2 = -dJdq_st
        D2 = B_st@inv(R)@Q_c.T@np.hstack([M,-S.T])
        f2 = beta_st - B_st@inv(R)@Q_c.T@h

        #task4
        Kp_sw = 10
        Kd_sw = 2*sqrt(Kp_sw)
        A4 = np.hstack([J_sw,np.zeros((3*(4-n_contact),nt))])
        b4 = -dJdq_sw+Kp_sw*(p_sw_des-p_sw)+Kd_sw*(dp_sw_des-dp_sw)+ddp_sw_des
        #add all the task
        tasks.append(task(A1,b1,D1,f1,0))
        tasks.append(task(A2,b2,D2,f2,1))
        tasks.append(task(A4,b4,None,None,4))


    else:
        J_sw = J_f
        dJdq_sw = dJdq_sw
        A4 = np.hstack([J_sw,np.zeros((3*(4-n_contact),nt))])
        b4 = -dJdq_sw+Kp_sw*(p_sw_des-p_sw)+Kd_sw*(dp_sw_des-dp_sw)
        tasks.append(task(A4,b4,None,None,3))


    #the feedback part of the body
    frame_id = robot.model.getFrameId("trunk")
    H = robot.framePlacement(q[:,ss], frame_id, False)
    v_frame = robot.frameVelocity(q[:,ss], v[:,ss], frame_id, False)
    a_frame_no_ddq = robot.frameAcceleration(q[:,ss], v[:,ss], None, frame_id, False)
    #state feedback
    x_bp= H.translation # take the 3d position of the end-effector
    x_bR = H.rotation
    dx_bp = v_frame.linear # take linear part of 6d velocity
    dx_bR = v_frame.angular
    #Jacobian
    J_bp = robot.frameJacobian(q[:,ss], frame_id, False)[:3,:]
    dJdq_bp = a_frame_no_ddq.linear
    J_bR = robot.frameJacobian(q[:,ss], frame_id, False)[3:,:]
    dJdq_bR = a_frame_no_ddq.angular
    #set reference
    Kp_bp = 10
    Kd_bp = 2*sqrt(Kp_bp)
    Kp_bR = 10
    Kd_bR = 2*sqrt(Kp_bR)
    x_bp_des = np.array([0.0,0,0.32])
    x_bR_des = np.eye(3)
    dx_bp_des = np.array([0.0,0,0])
    dx_bR_des = np.array([0.0,0.0,0.0])
    ddx_bp_des = np.array([0.0,0,0])
    #create task
    A3 = np.vstack([np.hstack([J_bp,np.zeros((3,nt))]),
                    np.hstack([J_bR,np.zeros((3,nt))])])
    b3 = np.hstack([-dJdq_bp+Kp_bp*(x_bp_des-x_bp)+Kd_bp*(dx_bp_des-dx_bp),
                    -dJdq_bR+Kp_bR*(pin.log3(x_bR_des.dot(x_bR.T)))+Kd_bR*(dx_bR_des-dx_bR)])
    tasks.append(task(A3,b3,None,None,4))
    #record to print 

    #the feedback of the manipulator
    frame_id = robot.model.getFrameId("j2s6s200_end_effector")
    H = robot.framePlacement(q[:,ss], frame_id, False)
    v_frame = robot.frameVelocity(q[:,ss], v[:,ss], frame_id, False)
    a_frame_no_ddq = robot.frameAcceleration(q[:,ss], v[:,ss], None, frame_id, False)
    #state feedback
    x_mp= H.translation # take the 3d position of the end-effector
    x_mR = H.rotation
    dx_mp = v_frame.linear # take linear part of 6d velocity
    dx_mR = v_frame.angular
    #Jacobian
    J_mp = robot.frameJacobian(q[:,ss], frame_id, False)[:3,:]
    dJdq_mp = a_frame_no_ddq.linear
    J_mR = robot.frameJacobian(q[:,ss], frame_id, False)[3:,:]
    dJdq_mR = a_frame_no_ddq.angular
    #set reference (world frame)
    Kp_mp = 10
    Kd_mp = 2*sqrt(Kp_mp)
    Kp_mR = 10
    Kd_mR = 2*sqrt(Kp_mR)
    f = 1
    omega = 2*pi*f
    amp = np.array([0.0,0.0,0.1])
    x_mp_des = np.array([ 0.703, -0.01 ,  0.661])+amp*np.sin([omega*t,omega*t+2*pi/3,omega*t-2*pi/3])
    x_mR_des = np.eye(3)
    dx_mp_des= np.array([0,0,0])+amp*omega*np.cos([omega*t,omega*t+2*pi/3,omega*t-2*pi/3])
    dx_mR_des = np.array([0,0,0])
    ddx_mp_des= -amp*omega*omega*np.sin([omega*t,omega*t+2*pi/3,omega*t-2*pi/3])
    #create task
    A5 = np.vstack([np.hstack([J_mp,np.zeros((3,nt))]),
                    np.hstack([J_mR,np.zeros((3,nt))])])
    b5 = np.hstack([-dJdq_mp+Kp_mp*(x_mp_des-x_mp)+Kd_mp*(dx_mp_des-dx_mp),
                    -dJdq_mR+Kp_mR*(pin.log3(x_mR_des.dot(x_mR.T)))+Kd_mR*(dx_mR_des-dx_mR)])
    tasks.append(task(A5,b5,None,None,3))
    
    #record to print the data
    mx[:,ss]= x_mp
    mx_ref[:,ss] = x_mp_des
    #calculate the ouput
    out = WBC_HO(tasks).solve()
    F = inv(R)@Q_c.T@(M@out[:nv]+h-S.T@out[nv:])
    
    tau[:,ss] = np.hstack([np.zeros(6),out[nv:]])
    local_plan.update(conf.dt,p_f,p_h,dx_bp)
    # send joint torques to simulator
    simu.simulate(tau[:,ss], conf.dt, conf.ndt)
    if ss%PRINT_N == 0:
        print("Time %.3f"%(t))
    t += conf.dt

LABEL={
    0:'x',1:'y',2:'z'
}
# PLOT STUFF
time = np.arange(0.0, N*conf.dt, conf.dt)
if(PLOT_EE_POS):    
    (f, ax) = plut.create_empty_figure(nx)
    title = "EE_POS"
    f.suptitle(title, fontsize=16)
    f.canvas.manager.set_window_title(title)
    ax = ax.reshape(nx)
    for i in range(nx):
        ax[i].plot(time, mx[i,:], label='EE_pos')
        ax[i].plot(time, mx_ref[i,:], '--', label='ref')
        ax[i].set_xlabel('Time [s]')
        ax[i].set_ylabel(r''+LABEL[i]+' [m]')
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

if(PLOT_ARM_JOINT_POS):    
    (f, ax) = plut.create_empty_figure(6)
    ax = ax.reshape(6)
    title = "ARM_JOINT_POS"
    f.suptitle(title, fontsize=16)
    f.canvas.manager.set_window_title(title)
    for i in range(6):
        ax[i].plot(time, q[i-6,:-1], label='q')
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

if(PLOT_ARM_TORQUES):    
    (f, ax) = plut.create_empty_figure(6)
    ax = ax.reshape(6)
    title = "ARM_TORQUES"
    f.suptitle(title, fontsize=16)
    f.canvas.manager.set_window_title(title)
    for i in range(6):
        ax[i].plot(time, tau[i-6,:], label=r'$\tau$ '+str(i))
        ax[i].set_xlabel('Time [s]')
        ax[i].set_ylabel(r'$\tau_{'+str(i)+'}$ [N/m]')
    leg = ax[0].legend()
    leg.get_frame().set_alpha(0.5)
        
plt.show()

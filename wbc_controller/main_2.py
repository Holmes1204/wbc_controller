import numpy as np
from numpy import nan
from numpy.linalg import inv,pinv,norm,matrix_rank as rank
import matplotlib.pyplot as plt
import time
from math import sqrt,sin,pi
import sys
sys.path.append("/home/holmes/Desktop/graduation/code/graduation_simulation_code")
import utils.plot_utils as plut
from utils.robot_wrapper import RobotWrapper
from utils.robot_simulator import RobotSimulator
import main_2_conf as conf
import pinocchio as pin
from reactive_control.local_planner import local_planner,reduce_convex
from solutions.WBC_HO import task,WBC_HO
f_path = '/home/holmes/Desktop/graduation/hitsz_paper/pictures/'


print("".center(conf.LINE_WIDTH,'#'))       
print(" Quadrupedal Robot".center(conf.LINE_WIDTH, '#'))
print("".center(conf.LINE_WIDTH,'#'), '\n')

PLOT_EE_POS = 0
PLOT_EE_VEL = 0

PLOT_BODY_POS = 0
PLOT_BODY_VEL = 0
PLOT_BODY_ACC = 0

PLOT_DOG_JOINT_POS = 0
PLOT_DOG_JOINT_VEL = 0
PLOT_DOG_JOINT_ACC = 0
PLOT_DOG_JOINT_TOR = 0

PLOT_ARM_JOINT_POS =0
PLOT_ARM_JOINT_VEL =0
PLOT_ARM_JOINT_ACC =0
PLOT_ARM_JOINT_TOR =0

rmodel, rcollision_model, rvisual_model = pin.buildModelsFromUrdf("./a1_description/urdf/a1_kinova.urdf", "./",pin.JointModelFreeFlyer())
robot = RobotWrapper(rmodel, rcollision_model, rvisual_model)
simu = RobotSimulator(conf, robot)
local_plan  = local_planner(conf,1)
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
dmx   = np.empty((nx,  N))*nan        # end-effector reference position
dmx_ref   = np.empty((nx,  N))*nan        # end-effector reference position
ddmx_ref   = np.empty((nx,  N))*nan        # end-effector reference position

traj_bp  = np.empty((ndx, N))*nan        # end-effector reference velocity
traj_dbp = np.empty((ndx, N))*nan        # end-effector reference acceleration
traj_ddbp = np.empty((ndx, N))*nan        # end-effector desired acceleration


#
foot_mu = conf.ground_mu/sqrt(2)
B_ = np.array([[ 1, 0,-foot_mu],
               [-1, 0,-foot_mu],
               [ 0, 1,-foot_mu],
               [ 0,-1,-foot_mu],
               [ 0, 0,-1],
               [ 0, 0, 1]])
beta_ =np.array([0,0,0,0,0,1000])
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
    #foot state feed back
    p_h = np.zeros((4,3))
    v_h = np.zeros((4,3))
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
    for j in range(len(conf.Hip_frame)) :
        frame_id = robot.model.getFrameId(conf.Hip_frame[j])
        H = robot.framePlacement(q[:,ss], frame_id, False)
        v_frame = robot.frameVelocity(q[:,ss], v[:,ss], frame_id, False)
        p_h[j,:] = H.translation+np.array([0.0,(-1)**j*0.084,0.0])
        v_h[j,:] = v_frame.linear
    #the feedback part of the body
    frame_id = robot.model.getFrameId("trunk")
    H = robot.framePlacement(q[:,ss], frame_id, False)
    v_frame = robot.frameVelocity(q[:,ss], v[:,ss], frame_id, False)
    a_frame_no_ddq = robot.frameAcceleration(q[:,ss], v[:,ss], None, frame_id, False)
    x_bp= H.translation # take the 3d position of the end-effector
    dx_bp = v_frame.linear # take linear part of 6d velocity
    x_bR = H.rotation
    dx_bR = v_frame.angular
    J_bp = robot.frameJacobian(q[:,ss], frame_id, False)[:3,:]
    dJdq_bp = a_frame_no_ddq.linear
    J_bR = robot.frameJacobian(q[:,ss], frame_id, False)[3:,:]
    dJdq_bR = a_frame_no_ddq.angular
    #the feedback of the manipulator
    frame_id = robot.model.getFrameId("j2s6s200_end_effector")
    H = robot.framePlacement(q[:,ss], frame_id, False)
    v_frame = robot.frameVelocity(q[:,ss], v[:,ss], frame_id, False)
    a_frame_no_ddq = robot.frameAcceleration(q[:,ss], v[:,ss], None, frame_id, False)
    x_mp= H.translation # take the 3d position of the end-effector
    x_mR = H.rotation
    dx_mp = v_frame.linear # take linear part of 6d velocity
    dx_mR = v_frame.angular
    J_mp = robot.frameJacobian(q[:,ss], frame_id, False)[:3,:]
    dJdq_mp = a_frame_no_ddq.linear

    J_mR = robot.frameJacobian(q[:,ss], frame_id, False)[3:,:]
    dJdq_mR = a_frame_no_ddq.angular
    #----------feed back over---------
    #foot update
    local_plan.update_foot(p_f,p_h,dx_bp,dx_bp)
    if ss == 0:
        # local_plan.body_traj_show()
        stp = x_bp[:2]
        dstp =dx_bp[:2]
        ddstp =np.zeros(2)
        fp=x_bp[:2]+np.array([0.3,0])*1
        support_polygon = local_plan.get_support_polygon(p_f[:,:2])
        shrink_polygon,edge = reduce_convex(support_polygon)
        local_plan.body_traj_plan(stp,dstp,ddstp,fp,edge,support_polygon,shrink_polygon)
        # print_each_support_polygon(support_polygon,shrink_polygon,edge)
        # print_all_support_polygon(support_polygon,shrink_polygon)
        # plt.show()


    
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
    # D2 = np.zeros((n_contact*B_.shape[0],n_contact*B_.shape[1]))
    # f2 = np.zeros(n_contact*beta_.shape[0])
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
                p_sw_des[3*j_sw:3*j_sw+3],dp_sw_des[3*j_sw:3*j_sw+3],ddp_sw_des[3*j_sw:3*j_sw+3] = local_plan.swing_foot_traj(j)
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
        Kp_sw = 1000
        Kd_sw = 2*sqrt(Kp_sw)
        A4 = np.hstack([J_sw,np.zeros((3*(4-n_contact),nt))])
        b4 = -dJdq_sw+Kp_sw*(p_sw_des-p_sw)+Kd_sw*(dp_sw_des-dp_sw)+ddp_sw_des
        #add all the task
        tasks.append(task(A1,b1,D1,f1,0))
        tasks.append(task(A2,b2,D2,f2,1))
        tasks.append(task(A4,b4,None,None,2))
    else:
        J_sw = J_f
        dJdq_sw = dJdq_sw
        A4 = np.hstack([J_sw,np.zeros((3*(4-n_contact),nt))])
        b4 = -dJdq_sw+Kp_sw*(p_sw_des-p_sw)+Kd_sw*(dp_sw_des-dp_sw)
        tasks.append(task(A4,b4,None,None,3))

    #set reference
    Kp_bp = 1000
    Kd_bp = 2*sqrt(Kp_bp)
    Kp_bR = 100
    Kd_bR = 2*sqrt(Kp_bR)
    traj_p,traj_dp,traj_ddp = local_plan.body_traj_update(conf.dt)
    # x_bp_des = np.array([traj_p[0],traj_dp[1],0.32])
    # dx_bp_des = np.array([traj_dp[0],traj_dp[1],0])
    # ddx_bp_des = np.array([traj_ddp[0],traj_ddp[1],0])
    x_bp_des = np.array([0.75*(ss/N)*(ss/N)/2.0,0.0,0.32])
    dx_bp_des = np.array([0.75*ss/N,0.0,0])
    ddx_bp_des = np.array([0.75,0,0])
    # x_bp_des   = np.array([0.0,0.0,0.32])
    # dx_bp_des  = np.array([0.0,0.0,0.0])
    # ddx_bp_des = np.array([0.0,0.0,0.0])
    x_bR_des = np.eye(3)
    dx_bR_des = np.array([0.0,0.0,0.0])
    #
    traj_bp[:,ss]= x_bp_des
    traj_dbp[:,ss]= dx_bp_des
    traj_ddbp[:,ss]= ddx_bp_des
    #create task
    A3 = np.vstack([np.hstack([J_bp,np.zeros((3,nt))]),
                    np.hstack([J_bR,np.zeros((3,nt))])])
    b3 = np.hstack([-dJdq_bp+Kp_bp*(x_bp_des-x_bp)+Kd_bp*(dx_bp_des-dx_bp)+ddx_bp_des,
                    -dJdq_bR+Kp_bR*(pin.log3(x_bR_des.dot(x_bR.T)))+Kd_bR*(dx_bR_des-dx_bR)])
    tasks.append(task(A3,b3,None,None,2))
    #set reference (world frame)
    Kp_mp = 100
    Kd_mp = 2*sqrt(Kp_mp)
    Kp_mR = 100
    Kd_mR = 2*sqrt(Kp_mR)
    f = 1
    omega = 2*pi*f
    amp = np.array([0.05,-0.05,0.05])
    x_mp_des = x_bp_des+np.array([ 0.703, -0.01 ,  0.661-x_bp_des[2]])+amp*np.sin([omega*t,omega*t+2*pi/3,omega*t-2*pi/3])
    dx_mp_des= np.array([0,0,0])+amp*omega*np.cos([omega*t,omega*t+2*pi/3,omega*t-2*pi/3])
    ddx_mp_des= -amp*omega*omega*np.sin([omega*t,omega*t+2*pi/3,omega*t-2*pi/3])
    x_mR_des = np.eye(3)
    dx_mR_des = np.array([0,0,0])
    #create task
    A5 = np.vstack([np.hstack([J_mp,np.zeros((3,nt))]),
                    np.hstack([J_mR,np.zeros((3,nt))])])
    b5 = np.hstack([-dJdq_mp+Kp_mp*(x_mp_des-x_mp)+Kd_mp*(dx_mp_des-dx_mp)+ddx_mp_des,
                    -dJdq_mR+Kp_mR*(pin.log3(x_mR_des.dot(x_mR.T)))+Kd_mR*(dx_mR_des-dx_mR)])
    # q0 =[1.5707,2.618,-1.5707,-1.5707,3.1415, 0.]
    # A5 = np.hstack([np.zeros((6,18)),np.eye(6),np.zeros((6,nt))])
    # b5 = np.hstack(Kp_mp*(q0-q[-6:,ss])-Kd_mp*v[-6:,ss])
    tasks.append(task(A5,b5,None,None,3))
    
    #record to print the data
    mx[:,ss]= x_mp
    mx_ref[:,ss] = x_mp_des
    dmx[:,ss]= dx_mp
    dmx_ref[:,ss] = dx_mp_des
    #calculate the ouput
    out = WBC_HO(tasks).solve()
    F = inv(R)@Q_c.T@(M@out[:nv]+h-S.T@out[nv:])
    
    tau[:,ss] = np.hstack([np.zeros(6),out[nv:]])

    # send joint torques to simulator
    simu.simulate(tau[:,ss], conf.dt, conf.ndt)
    if ss%PRINT_N == 0:
        print("Time %.3f"%(t))
    local_plan.update_phase(conf.dt)
    t += conf.dt


LABEL={
    0:'x',1:'y',2:'z'
}

LEG_LABEL={
    0:'LF',1:'RF',2:'LR',3:'RR'
}

import matplotlib.ticker as ticker
def set_ticks(ax,x,y):
    ax.xaxis.set_major_locator(ticker.MultipleLocator(x))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(y))
    ax.xaxis.set_minor_locator(ticker.MultipleLocator(x/10))
    ax.yaxis.set_minor_locator(ticker.MultipleLocator(y/10))

# PLOT STUFF
time = np.arange(0.0, N*conf.dt, conf.dt)
if(PLOT_EE_POS):    
    for i in range(nx):
        f, ax = plt.subplots()
        title = "EE_POS"+LABEL[i]
        f.canvas.manager.set_window_title(title)
        ax.plot(time, mx[i,:], label=LABEL[i])
        ax.plot(time, mx_ref[i,:], '--', label=LABEL[i]+'期望')
        ax.set_xlabel('时间(s)')
        ax.set_ylabel('位移(m)')
        ax.set_xlim(0,conf.T_SIMULATION)
        set_ticks(ax,conf.T_SIMULATION/5,0.02)
        ax.legend(loc=1)
        f.savefig(f_path+"exp3/"+title+".pdf",pad_inches=0.005,bbox_inches='tight')
    
if(PLOT_EE_VEL):  
    for i in range(nx):  
        f, ax = plt.subplots()
        title = "EE_VEL"+LABEL[i]
        f.canvas.manager.set_window_title(title)
        ax.plot(time, dmx[i,:], label=LABEL[i])
        ax.plot(time, dmx_ref[i,:], '--', label=LABEL[i]+'期望')
        ax.set_xlabel('时间(s)')
        ax.set_ylabel('速度(m/s)')
        ax.set_xlim(0,conf.T_SIMULATION)
        set_ticks(ax,conf.T_SIMULATION/5,0.2)
        ax.legend(loc=1)
        f.savefig(f_path+"exp3/"+title+".pdf",pad_inches=0.005,bbox_inches='tight')



if(PLOT_BODY_POS):   
    for i in range(nx):  
        f, ax = plt.subplots()
        title = "BODY_POS"+LABEL[i]
        f.canvas.manager.set_window_title(title)
        ax.plot(time, q[i,:-1], label=LABEL[i])
        ax.plot(time, traj_bp[i,:], '--', label=LABEL[i]+'期望')
        ax.set_xlabel('时间(s)')
        ax.set_ylabel('位移(m)')
        ax.set_xlim(0,conf.T_SIMULATION)
        set_ticks(ax,conf.T_SIMULATION/5,0.2)
        ax.legend(loc=1)
        f.savefig(f_path+"exp3/"+title+".pdf",pad_inches=0.005,bbox_inches='tight') 


if(PLOT_BODY_VEL):    
        for i in range(nx):  
            f, ax = plt.subplots()
            title = "BODY_VEL"+LABEL[i]
            f.canvas.manager.set_window_title(title)
            ax.plot(time, v[i,:-1], label=LABEL[i])
            ax.plot(time, traj_dbp[i,:], '--', label=LABEL[i]+'期望')
            ax.set_xlabel('时间(s)')
            ax.set_ylabel('速度(m/s)')
            ax.set_xlim(0,conf.T_SIMULATION)
            set_ticks(ax,conf.T_SIMULATION/5,0.2)
            ax.legend(loc=1)
            f.savefig(f_path+"exp3/"+title+".pdf",pad_inches=0.005,bbox_inches='tight') 



if(PLOT_BODY_ACC):    
        for i in range(nx):  
            f, ax = plt.subplots()
            title = "BODY_ACC"+LABEL[i]
            f.canvas.manager.set_window_title(title)
            ax.plot(time, dv[i,:-1], label=LABEL[i])
            ax.plot(time, traj_ddbp[i,:], '--', label=LABEL[i]+'期望')
            ax.set_xlabel('时间')
            ax.set_ylabel('加速度')
            ax.set_xlim(0,conf.T_SIMULATION)
            set_ticks(ax,conf.T_SIMULATION/5,0.2)
            ax.legend(loc=1)
            f.savefig(f_path+"exp3/"+title+".pdf",pad_inches=0.005,bbox_inches='tight') 



if(PLOT_DOG_JOINT_POS):    
    for i in range(4):
        (f, ax) = plt.subplots()
        title = LEG_LABEL[i]+'_POS'
        f.canvas.manager.set_window_title(title)
        for j in range(3):
            ax.plot(time, q[7+i*4+j,:-1], label='关节'+str(i*3+j))
        ax.set_xlabel('时间(s)')
        ax.set_ylabel('关节角度(rad)')
        ax.set_xlim(0,conf.T_SIMULATION)
        set_ticks(ax,conf.T_SIMULATION/5,0.5)
        ax.legend(loc=1)
        f.savefig(f_path+"exp3/"+title+".pdf",pad_inches=0.005,bbox_inches='tight')




if(PLOT_DOG_JOINT_VEL):    
    for i in range(4):
        (f, ax) = plt.subplots()
        title = LEG_LABEL[i]+'_VEL'
        f.canvas.manager.set_window_title(title)
        for j in range(3):
            ax.plot(time, v[6+i*4+j,:-1], label='关节'+str(i*3+j))
        ax.set_xlabel('时间(s)')
        ax.set_ylabel('关节角速度(rad/s)')
        ax.set_xlim(0,conf.T_SIMULATION)    
        ax.set_ylim(-0.4,0.4)
        set_ticks(ax,conf.T_SIMULATION/5,0.2)
        ax.legend(loc=1)
        f.savefig(f_path+"exp3/"+title+".pdf",pad_inches=0.005,bbox_inches='tight')



# if(PLOT_DOG_JOINT_ACC):    
#     for i in range(4):
#         (f, ax) = plut.create_empty_figure(1,1)
#         title = LEG_LABEL[i]+'_ACC'
#         f.canvas.manager.set_window_title(title)
#         for j in range(3):
#             ax.plot(time, dv[6+i*4+j,:-1], label=r'$\ddot{q}_{'+str(i*3+j)+r'}$')
#         ax.set_xlabel('时间(s)')
#         ax.set_ylabel(r'Angle Acceleration [$rad/s^2$]')
#         leg = ax.legend(loc=1)
#         leg.get_frame().set_alpha(0.5)

if(PLOT_DOG_JOINT_TOR):
    for i in range(4):
        (f, ax) = plt.subplots()
        title = LEG_LABEL[i]+'_TORQUE'
        f.canvas.manager.set_window_title(title)
        for j in range(3):
            ax.plot(time, tau[6+i*4+j,:], label='关节'+str(i*3+j))
        ax.set_xlabel('时间(s)')
        ax.set_ylabel('关节力矩(N/m)')
        ax.set_xlim(0,conf.T_SIMULATION)
        set_ticks(ax,conf.T_SIMULATION/5,5)
        ax.legend(loc=1)
        f.savefig(f_path+"exp3/"+title+".pdf",pad_inches=0.005,bbox_inches='tight')




if(PLOT_ARM_JOINT_POS):
    (f, ax) = plt.subplots()
    title = 'Manipulation_POS'
    f.canvas.manager.set_window_title(title)
    for j in range(6):
        ax.plot(time, q[j-6,:-1], label='关节'+str(12+j))
    ax.set_xlabel('时间(s)')
    ax.set_ylabel('关节角度(rad)')
    ax.set_xlim(0,conf.T_SIMULATION)
    set_ticks(ax,conf.T_SIMULATION/5,1)
    ax.legend(loc=1)
    f.savefig(f_path+"exp3/"+title+".pdf",pad_inches=0.005,bbox_inches='tight')

    


if(PLOT_ARM_JOINT_VEL):
    (f, ax) = plt.subplots()
    title = 'Manipulation_VEL'
    f.canvas.manager.set_window_title(title)
    for j in range(6):
        ax.plot(time, v[j-6,:-1], label='关节'+str(12+j))
    ax.set_xlabel('时间(s)')
    ax.set_ylabel('关节角速度(rad/s)')
    ax.set_xlim(0,conf.T_SIMULATION)
    ax.set_ylim(-5,5)
    set_ticks(ax,conf.T_SIMULATION/5,1)
    ax.legend(loc=1)
    f.savefig(f_path+"exp3/"+title+".pdf",pad_inches=0.005,bbox_inches='tight')



# if(PLOT_ARM_JOINT_ACC):
#     (f, ax) = plt.subplots(figsize=(5.9,0.75*5.9))
#     title = 'Manipulation_ACC'
#     f.canvas.manager.set_window_title(title)
#     for j in range(6):
#         ax.plot(time, dv[j-6,:-1], label='关节'+str(12+j))
#     ax.set_xlabel('时间(s)')
#     ax.set_ylabel(r'Angle Acceleration [$rad/s^2$]')
#     ax.legend(loc=1)



if(PLOT_ARM_JOINT_TOR):    
    (f, ax) = plt.subplots()
    title = 'Manipulation_TOR'
    f.canvas.manager.set_window_title(title)
    for j in range(6):
        ax.plot(time, tau[j-6,:], label='关节'+str(12+j))
    ax.set_xlabel('时间(s)')
    ax.set_ylabel('关节力矩(N/m)')
    ax.set_xlim(0,conf.T_SIMULATION)
    set_ticks(ax,conf.T_SIMULATION/5,5)
    ax.legend(loc=1)
    f.savefig(f_path+"exp3/"+title+".pdf",pad_inches=0.005,bbox_inches='tight')
    
# plt.show()

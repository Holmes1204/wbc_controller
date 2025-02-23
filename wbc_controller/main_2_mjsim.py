import numpy as np
from numpy import nan
from numpy.linalg import inv,pinv,norm,matrix_rank as rank
np.set_printoptions(precision=4, linewidth=2000)
import matplotlib.pyplot as plt
import time
from math import sqrt,sin,pi
import sys
sys.path.append("/home/holmes/Data/python/wbc_controller")

from utils.robot_wrapper import RobotWrapper
import main_2_conf as conf
import pinocchio as pin
from local_planner import local_planner,reduce_convex
from solutions.WBC_HO_qp import task,WBC_HO

def to_mj(q):
    """
    Convert from mujoco to pinocchio
    pin = [x,y,z,w]
    mj = [w,x,y,z]
    """
    q_ = q.copy()
    quat = q[3:7]# x,y,z,w
    q_[3:7] = quat[[3,0,1,2]] # w,x,y,z
    q_[7:] = q[7:][[3,4,5,0,1,2,9,10,11,6,7,8,12,13,14,15,16,17]]
    return q_


def to_mj_v(v):
    v_ = v.copy()
    v_[6:] = v[6:][[3,4,5,0,1,2,9,10,11,6,7,8,12,13,14,15,16,17]]
    return v_


def to_mj_tau(tau):
    return tau[[3,4,5,0,1,2,9,10,11,6,7,8,12,13,14,15,16,17]]
    return tau
  
def to_pin(q):
    """
    Convert from mujoco to pinocchio
    pin = [x,y,z,w]
    mj = [w,x,y,z]
    """
    q_ = q.copy()
    quat = q[3:7]#w,xyz
    q_[3:7] = quat[[1,2,3,0]]
    q_[7:] = q[7:][[3,4,5,0,1,2,9,10,11,6,7,8,12,13,14,15,16,17]]
    return q_


def to_pin_v(v):
    v_ = v.copy()
    v_[6:] = v[6:][[3,4,5,0,1,2,9,10,11,6,7,8,12,13,14,15,16,17]]
    return v_
  

import mujoco
import mujoco.viewer
mj_model = mujoco.MjModel.from_xml_path('/home/holmes/Data/python/wbc_controller/mujoco_sim/a1_kinova_description/scene.xml')
mj_data = mujoco.MjData(mj_model)
mj_model.opt.timestep = conf.dt/conf.ndt
mj_data.qpos = to_mj(conf.q0)
pin_init_j_pos = conf.q0[7:]

print("".center(conf.LINE_WIDTH,'#'))       
print(" Quadrupedal Robot".center(conf.LINE_WIDTH, '#'))
print("".center(conf.LINE_WIDTH,'#'), '\n')


rmodel, rcollision_model, rvisual_model = pin.buildModelsFromUrdf("./a1_description/urdf/a1_kinova.urdf", "./",pin.JointModelFreeFlyer())
robot = RobotWrapper(rmodel, rcollision_model, rvisual_model)

local_plan  = local_planner(conf,1)

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
foot_mu = conf.ground_mu/2
B_ = np.array([[ 1, 0,-foot_mu],
               [-1, 0,-foot_mu],
               [ 0, 1,-foot_mu],
               [ 0,-1,-foot_mu],
               [ 0, 0,-1],
               [ 0, 0, 1]])
beta_ =np.array([0,0,0,0,0,200])
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

ss = 0

with mujoco.viewer.launch_passive(mj_model, mj_data) as viewer:
  sim_start = time.time()
  while viewer.is_running():
    step_start = time.time()
    q = to_pin(mj_data.qpos)
    v = to_pin_v(mj_data.qvel)

    # compute mass matrix M, bias terms h, gravity terms g
    robot.computeAllTerms(q, v)
    M = robot.mass(q)
    h = robot.nle(q, v)#include gravity force
    g = robot.gravity(q)
    #foot state feed back
    p_h = np.zeros((4,3))
    v_h = np.zeros((4,3))
    acc = np.zeros(24)
    for j in range(len(conf.Foot_frame)) :
        tasks = []
        frame_id = robot.model.getFrameId(conf.Foot_frame[j])
        J = robot.frameJacobian(q, frame_id)[:3,:]
        dJdq = robot.frameAcceleration(q, v, acc, frame_id).linear
        dJ = robot.frameJacobianTimeVariation(q,v,frame_id)[:3,:]
        H = robot.framePlacement(q, frame_id)
        v_frame = robot.frameVelocity(q, v, frame_id)
        p_f[j,:] = H.translation
        v_f[j,:] = v_frame.linear
        J_f[3*j:3*j+3,:] = J
        dJdq_f[3*j:3*j+3] = dJdq
    for j in range(len(conf.Hip_frame)) :
        frame_id = robot.model.getFrameId(conf.Hip_frame[j])
        H = robot.framePlacement(q, frame_id)
        v_frame = robot.frameVelocity(q, v, frame_id)
        p_h[j,:] = H.translation+np.array([0.0,(-1)**j*0.084,0.0])
        v_h[j,:] = v_frame.linear
    #the feedback part of the body
    frame_id = robot.model.getFrameId("trunk")
    H = robot.framePlacement(q, frame_id)
    v_frame = robot.frameVelocity(q, v, frame_id)
    a_frame_no_ddq = robot.frameAcceleration(q, v, acc, frame_id)
    x_bp= H.translation # take the 3d position of the end-effector
    dx_bp = v_frame.linear # take linear part of 6d velocity
    x_bR = H.rotation
    dx_bR = v_frame.angular
    J_bp = robot.frameJacobian(q, frame_id)[:3,:]
    dJdq_bp = a_frame_no_ddq.linear
    J_bR = robot.frameJacobian(q, frame_id)[3:,:]
    dJdq_bR = a_frame_no_ddq.angular
    #the feedback of the manipulator
    frame_id = robot.model.getFrameId("j2s6s200_end_effector")
    H = robot.framePlacement(q, frame_id)
    v_frame = robot.frameVelocity(q, v, frame_id)
    a_frame_no_ddq = robot.frameAcceleration(q, v, acc, frame_id)
    x_mp= H.translation # take the 3d position of the end-effector
    x_mR = H.rotation
    dx_mp = v_frame.linear # take linear part of 6d velocity
    dx_mR = v_frame.angular
    J_mp = robot.frameJacobian(q, frame_id)[:3,:]
    dJdq_mp = a_frame_no_ddq.linear

    J_mR = robot.frameJacobian(q, frame_id)[3:,:]
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
    n_contact =4
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
        D1_1 = np.block([[np.zeros(d_spe), np.eye(nt)],
                       [np.zeros(d_spe),-np.eye(nt)]])
        f1_1 = np.block([np.ones(nt)*33.5,np.ones(nt)*33.5])

        A2 = np.hstack([J_st,np.zeros((3*n_contact,nt))])
        b2 = -dJdq_st
        D1_2 = B_st@inv(R)@Q_c.T@np.hstack([M,-S.T])
        f1_2 = beta_st - B_st@inv(R)@Q_c.T@h
        D1 = np.vstack([D1_1,D1_2])
        f1 = np.hstack([f1_1,f1_2])
        # tasks.append(task(A1,b1,D1,f1,0))
        # tasks.append(task(A2,b2,D2,f2,1))
        tasks.append(task(0,(A1,b1),None)) # dynamic 
        tasks.append(task(1,None,(D1,f1))) # tau and friction cone
        tasks.append(task(2,(A2,b2),None)) # no contact motion
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
        f1 = np.block([np.ones(nt)*133.5,np.ones(nt)*133.5])
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
    Kp_bp = 50
    Kd_bp = 0.01*sqrt(Kp_bp)
    Kp_bR = 5
    Kd_bR = 1*sqrt(Kp_bR)
    # traj_p,traj_dp,traj_ddp = local_plan.body_traj_update(conf.dt)
    # x_bp_des = np.array([traj_p[0],traj_dp[1],0.32])
    # dx_bp_des = np.array([traj_dp[0],traj_dp[1],0])
    # ddx_bp_des = np.array([traj_ddp[0],traj_ddp[1],0])
    # x_bp_des = np.array([0.75*(ss/N)*(ss/N)/2.0,0.0,0.32])
    # dx_bp_des = np.array([0.75*ss/N,0.0,0])
    # ddx_bp_des = np.array([0.75,0,0])
    x_bp_des   = np.array([0.0,0.0,0.32])
    dx_bp_des  = np.array([0.0,0.0,0.0])
    ddx_bp_des = np.array([0.0,0.0,0.0])
    x_bR_des = np.eye(3)
    dx_bR_des = np.array([0.0,0.0,0.0])
    #
    # traj_bp= x_bp_des
    # traj_dbp= dx_bp_des
    # traj_ddbp= ddx_bp_des
    #create task
    A3 = np.vstack([np.hstack([J_bp,np.zeros((3,nt))]),
                    np.hstack([J_bR,np.zeros((3,nt))])])
    b3 = np.hstack([-dJdq_bp+Kp_bp*(x_bp_des-x_bp)+Kd_bp*(dx_bp_des-dx_bp)+ddx_bp_des,
                    -dJdq_bR+Kp_bR*(pin.log3(x_bR_des.dot(x_bR.T)))+Kd_bR*(dx_bR_des-dx_bR)])
    # tasks.append(task(A3,b3,None,None,2))
    tasks.append(task(3,(A3,b3),None)) # tracking 
    #set reference (world frame)
    Kp_mp = 5
    Kd_mp = 2*sqrt(Kp_mp)
    Kp_mR = 5
    Kd_mR = 2*sqrt(Kp_mR)

    f = 1
    omega = 2*pi*f
    amp = np.array([0.02,-0.05,0.05])
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
    
    
    # A4
    # Kp_mp = 50
    # Kd_mp = 2*sqrt(Kp_mp)
    # q0 =[1.5707,2.618,4.7707,-1.5707,3.1415, 0.]
    # A5 = np.hstack([np.zeros((6,18)),np.eye(6),np.zeros((6,nt))])
    # b5 = np.hstack(Kp_mp*(q0-q[-6:])-Kd_mp*v[-6:])
    # tasks.append(task(A5,b5,None,None,3))
    tasks.append(task(3,(A5,b5),None)) # minimize force
    
    #record to print the data
    mx= x_mp
    mx_ref = x_mp_des
    dmx= dx_mp
    dmx_ref = dx_mp_des
    #calculate the ouput
    out = WBC_HO(tasks).solve()
    F = inv(R)@Q_c.T@(M@out[:nv]+h-S.T@out[nv:])
    tau = out[nv:]

    print("--------check--------↓")
    # err1 = np.linalg.norm(A1@out-b1)
    # err2 = np.linalg.norm(A2@out-b2)
    # err3 = np.linalg.norm(A3@out-b3)
    # err4 = np.linalg.norm(A4@out-b4)
    print("eval eq 1",np.linalg.norm(A1@out-b1))
    print("eval ie 1",max(0,np.max(D1@out-f1)))
    print("eval eq 2",np.linalg.norm(A2@out-b2))
    print("eval eq 3",np.linalg.norm(A3@out-b3))
    # print("eval eq 4",np.linalg.norm(A4@out-b4))
    print('force',F.reshape(-1,3)) 
    print('Jst\n',J_st) 
    print("--------check--------↑")



    # send joint torques to simulator
    kp =50
    kd = 1*sqrt(kp)
    tau_pd = kp*(pin_init_j_pos-q[7:]) - kd*v[6:]
    mj_data.ctrl = to_mj_tau(tau)
    # for i in range(conf.ndt):
    #     mujoco.mj_step(mj_model, mj_data)
    local_plan.update_phase(conf.dt)
    t += conf.dt

    ss += 1
    if ss % 5 ==0:# 
        print("sim ",t)
        viewer.sync()
    if t>60.0:
        break
sim_end = time.time()
print("time cost",sim_end-sim_start)



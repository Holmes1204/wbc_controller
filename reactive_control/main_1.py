import numpy as np
from numpy import nan
from numpy.linalg import inv
from numpy.linalg import pinv
from numpy.linalg import matrix_rank as rank
from numpy.linalg import norm
import matplotlib.pyplot as plt
import time
from math import sqrt
import sys
sys.path.append("/home/holmes/Code/orc")
import utils.plot_utils as plut
from utils.robot_loaders import loadUR
from utils.robot_wrapper import RobotWrapper
from utils.robot_simulator import RobotSimulator
import main_1_conf as conf
import solutions.main_1_solution as solution
from example_robot_data.robots_loader import load
import pinocchio as pin
print("".center(conf.LINE_WIDTH,'#'))
print(" Quadrupedal Robot".center(conf.LINE_WIDTH, '#'))
print("".center(conf.LINE_WIDTH,'#'), '\n')

PLOT_JOINT_POS = 1
PLOT_JOINT_VEL = 0
PLOT_JOINT_ACC = 1
PLOT_TORQUES = 1
PLOT_EE_POS = 0
# PLOT_EE_VEL = 1

#load('ur5')
r = load('a1')
robot = RobotWrapper(r.model, r.collision_model, r.visual_model)
simu = RobotSimulator(conf, robot)
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

two_pi_f             = 2*np.pi*conf.freq   # frequency (time 2 PI)
two_pi_f_amp         = two_pi_f * conf.amp
two_pi_f_squared_amp = two_pi_f * two_pi_f_amp

S = np.zeros((12,18))
S[:,6:]=np.eye(12)

t = 0.0
kp, kd = conf.kp, conf.kd
PRINT_N = int(conf.PRINT_T/conf.dt)

for ss in range(0, N):#ss: simualtion step
    time_start = time.time()
    # set reference trajectory
    # x_ref[:,ss]  = conf.x0 +  conf.amp*np.sin(two_pi_f*t + conf.phi)
    # dx_ref[:,ss]  = two_pi_f_amp * np.cos(two_pi_f*t + conf.phi)
    # ddx_ref[:,ss] = - two_pi_f_squared_amp * np.sin(two_pi_f*t + conf.phi)
    x_ref[:,ss]  =  np.array([0.0,0.10,0.32])  
    dx_ref[:,ss] = np.array([0.0, 0.0, 0.0])  
    ddx_ref[:,ss] = np.array([0.0, 0.0, 0.0])  
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

    J_s = np.zeros((3*3,18))
    dJdq_s = np.zeros(3*3)
    for j in range(len(conf.Foot_frame)-1) :
        frame_id = robot.model.getFrameId(conf.Foot_frame[j])
        J = robot.frameJacobian(q[:,ss], frame_id, False)[:3,:]
        dJdq = robot.frameAcceleration(q[:,ss], v[:,ss], None, frame_id, False).linear
        dJ = robot.frameJacobianTimeVariation(q[:,ss],v[:,ss],frame_id)[:3,:]
        J_s[3*j:3*j+3,:] = J
        dJdq_s[3*j:3*j+3] = dJdq

    Q,R = np.linalg.qr(J_s.T,'complete')
    Q_u = Q[:,9:]#
    ##control body pos and orientation
    J_bp = np.zeros((3,18))
    J_bR = np.zeros((3,18))
    dJdq_bp = np.zeros(3)
    dJdq_bR = np.zeros(3)
    frame_id = robot.model.getFrameId("trunk")
    #feed back
    H = robot.framePlacement(q[:,ss], frame_id, False)
    x_bp= H.translation # take the 3d position of the end-effector
    x_bR = H.rotation
    x_bp_des = np.array([0,0,0.32])
    x_bR_des = np.eye(3)
    v_frame = robot.frameVelocity(q[:,ss], v[:,ss], frame_id, False)
    dx_bp = v_frame.linear # take linear part of 6d velocity
    dx_bR = v_frame.angular
    dx_bp_des = np.array([0,0,0])
    dx_bR_des = np.array([0,0,0])
    #jacobian
    J = robot.frameJacobian(q[:,ss], frame_id, False)[:3,:]
    dJdq = robot.frameAcceleration(q[:,ss], v[:,ss], None, frame_id, False).angular
    dJ = robot.frameJacobianTimeVariation(q[:,ss],v[:,ss],frame_id)[:3,:]
    J_bp = J
    dJdq_bp = dJdq
    #
    J = robot.frameJacobian(q[:,ss], frame_id, False)[3:,:]
    dJdq = robot.frameAcceleration(q[:,ss], v[:,ss], None, frame_id, False).linear
    dJ = robot.frameJacobianTimeVariation(q[:,ss],v[:,ss],frame_id)[3:,:]
    J_bR = J
    dJdq_bR = dJdq


    #hierarchical optimization
    kexi = np.zeros(30)
    A1 = Q_u.T@np.hstack([-M,S.T])
    b1 = Q_u.T@h
    kexi = solution.WBC_HO(A1,b1)

    Z1 = np.eye(30) - pinv(A1)@A1
    A2 = np.hstack([J_s,np.zeros((9,12))])
    b2 = -dJdq_s
    dkexi = solution.WBC_HO(A2@Z1,b2-A2@kexi)
    kexi += dkexi
    #the differnce of the hierarchical settings
#motion tracking of legs

#
    Kp_bp = 10
    Kd_bp = 2*sqrt(Kp_bp)
    Kp_bR = 10
    Kd_bR = 2*sqrt(Kp_bR)
    Z2 = Z1@(np.eye(30)-pinv(A2@Z1)@A2@Z1)
    A3 = np.vstack([np.hstack([J_bp,np.zeros((3,12))]),
                    np.hstack([J_bR,np.zeros((3,12))])])
    b3 = np.hstack([-dJdq_bp+Kp_bp*(x_bp_des-x_bp)+Kd_bp*(dx_bp_des-dx_bp),
                    -dJdq_bR+Kp_bR*(pin.log3(x_bR_des.dot(x_bR.T)))+Kd_bR*(dx_bR_des-dx_bR)])

    dkexi = solution.WBC_HO(A3@Z2,b3-A3@kexi)
    kexi += dkexi
    #test for motion tracking 

    # Z3 = Z2@(np.eye(30)-pinv(A3@Z2)@A3@Z2)

    # Z4 = Z3@(np.eye(30)-pinv(A4@Z3)@A4@Z3)
    # multi task optimizition same priority
    # J_stack,dJdq_stack,ddx_des_stack=solution.multi_task_jacobian_ref(robot,q[:,ss],v[:,ss],conf)
    tau[:,ss] = np.hstack([np.zeros(6),kexi[18:]])
    # tau[:,ss] = solution.operational_motion_control(q[:,ss], v[:,ss], ddx_des[:,ss], h, M, J, dJdq, conf)
    # tau[:,ss] = solution.task_qp(q[:,ss],v[:,ss], ddx_des[:,ss],M,h,J,dJdq,conf)
    # tau[:,ss] = solution.task_qp(q[:,ss],v[:,ss], ddx_des_stack,M,h,J_stack,dJdq_stack,conf)
    # tau[:,ss] = solution.task_cvxopt(q[:,ss],v[:,ss], ddx_des[:,ss],M,h,J,dJdq,conf)
    # tau[:,ss] = solution.task_osqp(q[:,ss],v[:,ss], ddx_des[:,ss],M,h,J,dJdq,conf)

    # send joint torques to simulator
    simu.simulate(tau[:,ss], conf.dt, conf.ndt)
    tau_c[:,ss] = simu.tau_c#friction
    # print(tau[:,ss])
    if ss%PRINT_N == 0:
        print("Time %.3f"%(t))
    t += conf.dt
        
    time_spent = time.time() - time_start
    if(conf.simulate_real_time and time_spent < conf.dt): 
        time.sleep(conf.dt-time_spent)

# PLOT STUFF
time = np.arange(0.0, N*conf.dt, conf.dt)
if(PLOT_EE_POS):    
    (f, ax) = plut.create_empty_figure(nx)
    ax = ax.reshape(nx)
    for ss in range(nx):
        ax[ss].plot(time, x[ss,:], label='x')
        ax[ss].plot(time, x_ref[ss,:], '--', label='x ref')
        ax[ss].set_xlabel('Time [s]')
        ax[ss].set_ylabel(r'x_'+str(ss)+' [m]')
    leg = ax[0].legend()
    leg.get_frame().set_alpha(0.5)
    
if(PLOT_JOINT_POS):    
    (f, ax) = plut.create_empty_figure(int(robot.nv/2),2)
    ax = ax.reshape(robot.nv)
    for ss in range(robot.nv):
        ax[ss].plot(time, q[ss,:-1], label='q')
        ax[ss].set_xlabel('Time [s]')
        ax[ss].set_ylabel(r'$q_'+str(ss)+'$ [rad]')
    leg = ax[0].legend()
    leg.get_frame().set_alpha(0.5)
        
if(PLOT_JOINT_VEL):    
    (f, ax) = plut.create_empty_figure(int(robot.nv/2),2)
    ax = ax.reshape(robot.nv)
    for ss in range(robot.nv):
        ax[ss].plot(time, v[ss,:-1], label='v')
#        ax[ss].plot(time, v_ref[ss,:], '--', label='v ref')
        ax[ss].set_xlabel('Time [s]')
        ax[ss].set_ylabel(r'v_'+str(ss)+' [rad/s]')
    leg = ax[0].legend()
    leg.get_frame().set_alpha(0.5)
        
if(PLOT_JOINT_ACC):    
    (f, ax) = plut.create_empty_figure(int(robot.nv/2),2)
    ax = ax.reshape(robot.nv)
    for ss in range(robot.nv):
        ax[ss].plot(time, dv[ss,:-1], label=r'$\dot{v}$')
#        ax[ss].plot(time, dv_ref[ss,:], '--', label=r'$\dot{v}$ ref')
        ax[ss].set_xlabel('Time [s]')
        ax[ss].set_ylabel(r'$\dot{v}_'+str(ss)+'$ [rad/s^2]')
    leg = ax[0].legend()
    leg.get_frame().set_alpha(0.5)
   
if(PLOT_TORQUES):    
    (f, ax) = plut.create_empty_figure(int(robot.nv/2),2)
    ax = ax.reshape(robot.nv)
    for ss in range(robot.nv):
        ax[ss].plot(time, tau[ss,:], label=r'$\tau$ '+str(ss))
        ax[ss].plot(time, tau_c[ss,:], label=r'$\tau_c$ '+str(ss))
        ax[ss].set_xlabel('Time [s]')
        ax[ss].set_ylabel('Torque [Nm]')
    leg = ax[0].legend()
    leg.get_frame().set_alpha(0.5)
        
plt.show()

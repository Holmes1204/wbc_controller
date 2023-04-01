#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 28 07:34:55 2021

@author: student
"""
import numpy as np
from numpy.linalg import inv
from cvxopt import matrix
from cvxopt import solvers
from quadprog import solve_qp
import osqp
import scipy as sp
import scipy.optimize
import scipy.stats
from scipy import sparse

def operational_motion_control(q, v, ddx_des, h, M, J, dJdq, conf):
    Minv = inv(M)
    J_Minv = J @ Minv
    Lambda = inv(J_Minv @ J.T)
#    if(not np.isfinite(Lambda).all()):
#    print('Eigenvalues J*Minv*J.T', np.linalg.eigvals(J_Minv @ J.T))
    mu = Lambda @ (J_Minv @ h - dJdq)
    f = Lambda @ ddx_des + mu
    tau = J.T @ f
    # secondary task
    # J_T_pinv = Lambda @ J_Minv
    # nv = J.shape[1]
    # NJ = np.eye(nv) - J.T @ J_T_pinv
    # tau_0 = M @ (conf.kp_j * (conf.q0 - q) - conf.kd_j*v) + h
    # tau += NJ @ tau_0
    
#    tau = h + J.T @ Lambda @ ddx_des + NJ @ tau_0
    # print("1",tau.T)
#    print("tau", tau.T)
#    print("JT*f", (J.T @ f).T)
#    print("dJdq", (J.T @ Lambda @ dJdq).T)

    return tau
#only when the state change, the optimiztion problem has to be changed.
#I think the activate set mehtod is really the good one to try
def task_cvxopt(q, v, ddx_des, M,h,J, dJdq, conf):
    Ac = np.hstack((J,np.zeros((3,12))))
    bc = ddx_des - dJdq
    Aeq = np.hstack((M,-np.eye(12)))
    beq = -h
    P_ =matrix(Ac.T@Ac+0.000000001*np.eye(24)+np.diag(np.hstack([np.zeros(12),np.ones(12)])))
    q_ =matrix(-Ac.T@bc)
    A_= matrix(Aeq)
    b_=  matrix(beq)
    G_ = matrix(np.zeros((24,24)))
    h_ = matrix(np.zeros((24,1)))
    #here comes some problem 
    # the solver asumps that the ran(A) equal the the row
    #the rank(P;A;G) equals to the column
    # print(np.linalg.matrix_rank(A))
    # print(np.linalg.matrix_rank(P))
    # print(np.linalg.matrix_rank(G))
    # print(np.linalg.matrix_rank(np.vstack((P,A,G))))

    solvers.options['show_progress'] = False
    sol = solvers.qp(P_,q_,G_,h_,A_,b_)
    # print(sol['primal objective'])
    tau = np.array(sol['x'])
    # print("2",tau[6:,0].T)
    # print(M@tau[0:6]+g+h)
    # print(h[6:])
    return tau[12:,0]


def task_qp(q, v, ddx_des, M,h,J, dJdq, conf):
    Ac = np.hstack((J,np.zeros((12,12))))
    bc = ddx_des - dJdq
    Aeq = np.hstack((M,-np.eye(12)))
    beq = -h
    # beq = beq.reshape(12)
    G_1 = Ac.T@Ac+1e-8*np.eye(24)+np.diag(np.hstack([np.zeros(12),np.ones(12)*1e-6]))
    #this one have to make sure the matrix in positive define
    a_1 = Ac.T@bc
    # a_1 = a_1.reshape(24)
    xf, f, xu, iters, lagr, iact = solve_qp(G_1, a_1,Aeq.T, beq,meq=12)
    tau = xf
    # print("3\n",tau[12:].reshape(4,3))
    return tau[12:]

def task_osqp(q, v, ddx_des, M,h,J, dJdq, conf):
    Ac = np.hstack((J,np.zeros((3,6))))
    bc = ddx_des - dJdq
    Aeq = np.hstack((M,-np.eye(6)))
    beq = -h
    #this one have to make sure the matrix in positive define

    P_ = sparse.csc_matrix(Ac.T@Ac)
    q_ = -Ac.T@bc
    A_ = sparse.csc_matrix(Aeq)
    l_ = beq
    u_ = beq

    solver = osqp.OSQP()    # Create an OSQP object
    solver.setup(P_, q_, A_, l_, u_)    # Setup workspace
    solver.update_settings(verbose=False,polish=True)
    res = solver.solve()    # Solve problem
    # print("3",tau[6:,0].T)
    tau = res.x
    return tau[6:]

# def task_osqp_new(q, v, ddx_des, M,h,J, dJdq, conf):
#     Ac = np.hstack((J,np.zeros((3,6))))
#     bc = ddx_des - dJdq
#     Aeq = np.hstack((M,-np.eye(6)))
#     beq = -h
#     #this one have to make sure the matrix in positive define

#     P_ = sparse.csc_matrix(Ac.T@Ac)
#     q_ = -Ac.T@bc
#     A_ = sparse.csc_matrix(Aeq)
#     l_ = beq
#     u_ = beq
#     global osqp_setup
#     if not osqp_setup:
#         osqp_solver.setup(P_, q_, A_, l_, u_)    # Setup workspace
#         osqp_solver.update_settings(verbose=False)
#         osqp_setup = True
#     else:
#         osqp_solver.update(q_,l_, u_,Px=Ac.T@Ac,Ax=Aeq) 
#     res = osqp_solver.solve()    # Solve problem
#     # print("3",tau[6:,0].T)
#     tau = res.x
#     return tau[6:]

def solve_qp_scipy(G, a, C, b, meq):
    def f(x):
        return 0.5 * np.dot(x, G).dot(x) - np.dot(a, x)

    constraints = []
    if C is not None:
        constraints = [{
            'type': 'eq' if i < meq else 'ineq',
            'fun': lambda x, C=C, b=b, i=i: (np.dot(C.T, x) - b)[i]
        } for i in range(C.shape[1])]

    result = scipy.optimize.minimize(
        f, x0=np.zeros(len(G)), method='SLSQP', constraints=constraints,
        tol=1e-10, options={'maxiter': 2000})
    return result

def test():
    q = np.random.rand(6,1)+0.1
    v = np.random.rand(6,1)+0.1
    ddx_des = np.random.rand(3,1)+0.1
    M = np.eye(6)
    h = np.random.rand(6,1)+0.1
    g = np.random.rand(6,1)+0.1
    J = np.random.rand(3,6)+0.100
    dJdq= np.random.rand(3,1)+0.1
    # task_qp(q,v,ddx_des,M,h,J,dJdq,None)
    # task_cvxopt(q,v,ddx_des,M,h,J,dJdq,None)
    task_osqp(q,v,ddx_des,M,h,J,dJdq,None)


def multi_task_jacobian_ref(robot,q,v,conf):
    J_stack = np.zeros((12,12))
    dJdq_stack = np.zeros(12)
    ddx_des_stack = np.zeros(12)
    x_ref = np.array([[0.205, 0.1308, -0.3],[0.205, -0.1308, -0.3],[-0.205, 0.1308, -0.3],[-0.205, -0.1308, -0.3]])
    for i in range(len(conf.Foot_frame)) :
        frame_id = robot.model.getFrameId(conf.Foot_frame[i])
        J = robot.frameJacobian(q, frame_id, False)[:3,:]        # take first 3 rows of J6
        dJ = robot.frameJacobianTimeVariation(q,v,frame_id)
        dJdq = robot.frameAcceleration(q, v, None, frame_id, False).linear
        #
        x = robot.framePlacement(q, frame_id, False).translation
        dx = robot.frameVelocity(q, v, frame_id, False).linear
        # implement your control law here
        ddx_des = conf.kp*(x_ref[i] - x) + conf.kd*( - dx)
        #
        J_stack[3*i:3*i+3,:]=J
        dJdq_stack[3*i:3*i+3]=dJdq
        ddx_des_stack[3*i:3*i+3]=ddx_des

    return J_stack, dJdq_stack,ddx_des_stack


def WBC_HO(A,b):
    #this one have to make sure the matrix in positive define
    G_1 = A.T@A +1e-9*np.eye(30)
    a_1 = A.T@b
    # a_1 = a_1.reshape(24)
    xf, f, xu, iters, lagr, iact = solve_qp(G_1, a_1)

    # print("3\n",tau[12:].reshape(4,3))
    return xf




if __name__ =="__main__":
    test()




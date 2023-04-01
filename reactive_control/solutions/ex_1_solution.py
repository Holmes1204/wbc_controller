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
    
#    tau[:,i] = h + J.T @ Lambda @ ddx_des[:,i] + NJ @ tau_0
    # print("1",tau.T)
#    print("tau", tau[:,i].T)
#    print("JT*f", (J.T @ f).T)
#    print("dJdq", (J.T @ Lambda @ dJdq).T)

    return tau

def task_cvxopt(q, v, ddx_des, M,h,J, dJdq, conf):
    Ac = np.hstack((J,np.zeros((3,6))))
    bc = ddx_des - dJdq
    Aeq = np.hstack((M,-np.eye(6)))
    beq = -h
    P_ =matrix(Ac.T@Ac+0.000000001*np.eye(12)+np.diag(np.hstack([np.zeros(6),np.ones(6)])))
    q_ =matrix(-Ac.T@bc)
    A_= matrix(Aeq)
    b_=  matrix(beq)
    G_ = matrix(np.zeros((12,12)))
    h_ = matrix(np.zeros((12,1)))
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
    return tau[6:,0]


def task_qp(q, v, ddx_des, M,h,J, dJdq, conf):
    Ac = np.hstack((J,np.zeros((3,6))))
    bc = ddx_des - dJdq
    Aeq = np.hstack((M,-np.eye(6)))
    beq = -h
    beq = beq.reshape(6)
    G_1 = Ac.T@Ac+0.000000001*np.eye(12)+np.diag(np.hstack([np.ones(6)*0.1,np.zeros(6)]))
    #this one have to make sure the matrix in positive define
    a_1 = Ac.T@bc
    a_1 = a_1.reshape(12)
    xf, f, xu, iters, lagr, iact = solve_qp(G_1, a_1,Aeq.T, beq,meq=6)
    tau = xf.reshape(12,1)
    # print("3",tau[6:,0].T)
    return tau[6:,0]

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


if __name__ =="__main__":
    test()




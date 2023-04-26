import numpy as np
import matplotlib.pyplot as plt
from quadprog import solve_qp
from numpy.linalg import matrix_rank as rank,inv
from numpy import hstack,vstack
from qpsolvers import solve_qp as qp_solve

def Acc(T,alpha=1e-8):
    return np.array([[400.0/7.0*pow(T,7),40*pow(T,6),120.0/5.0*pow(T,5),10*pow(T,4),0,0],
                     [40*pow(T,6),144/5*pow(T,5),18*pow(T,4),8*pow(T,3),0,0],
                     [120.0/5.0*pow(T,5),18*pow(T,4),12*pow(T,3),6*pow(T,2),0,0],
                     [10*pow(T,4),8*pow(T,3),6*pow(T,2),4*T,0,0],
                     [0,0,0,0,alpha,0],
                     [0,0,0,0,0,alpha]])


def nt(t):
    return np.array([[pow(t,5),pow(t,4),pow(t,3),pow(t,2),t,1]]) 


def ddnt(t):
    return np.array([[20*pow(t,3),12*pow(t,2),6*t,2,0,0]]) 


def dnt(t):
    return np.array([[5*pow(t,4),4*pow(t,3),3*pow(t,2),2*t,1,0]]) 

def T(t):
    return diagm([nt(t),ddnt(t),dnt(t)])

def diagm(matlist: list,rcol = 0):
    mat  = None
    for it in matlist :
        if mat is None:
            mat = it
        else:
            mat = np.block([[mat,np.zeros((mat.shape[0],it.shape[1]-rcol))],
                            [np.zeros((it.shape[0],mat.shape[1]-rcol)),it]])
    return mat


def diagv(veclist: list,rcol = 0):
    vec  = None
    for it in veclist :
        if vec is None:
            vec = it
        else:
            vec = np.block([vec,np.zeros(rcol),it])
    return vec


def Q_traj(Tm,time:np.array,p:np.array):
    Q_ = np.zeros((6,6))
    a_ = np.zeros(6)
    for i in range(len(time)):
        Q_ += Tm(time[i]).T@Tm(time[i])
        a_ +=Tm(time[i]).reshape(-1)*p[i]
    return Q_,a_
 

def traj_opt(duration,stp,dstp,ddstp,fp,p=None,dp=None,ddp=None):
    """
        p: all the sample points of postion for each segment part,structure like this [[],[]]
        dp: all the sample points of postion for each segment part,structure like this [[],[]]
        ddp: all the sample points of postion for each segment part,structure like this [[],[]]
    """
    Q_all = np.zeros((0,0))
    a_all = np.zeros(0)
    Ceq_all = np.zeros((0,0))
    beq_all = np.zeros(0)
    Ciq_all = np.zeros((0,0))
    biq_all = np.zeros(0)
    delta =0.01#in meter
    n_seg = len(duration)
    dim  = len(stp)#
    for i in range(n_seg):
        #for the sampling of the line, we have some others formulations
        #for the qp slover
        Q_ = None
        a_ = None
        Ceq_ = None
        beq_ = None
        Ceq_sm_=None
        beq_sm_=None
        Ciq_ = None
        biq_ = None
        Q_seg=[]
        a_seg=[]
        Ceq_seg=[]
        beq_seg=[] 
        Ceq_sm=[]
        beq_sm=[] 
        Ciq_seg=[]
        biq_seg=[]

        if i == 0 :#first segment
            for j in range(dim):
                Q_seg.append(Acc(duration[i]))
                a_seg.append(np.zeros(6))
                Ceq_seg.append(vstack([nt(0),dnt(0),ddnt(0)]))
                beq_seg.append(hstack([stp[j],dstp[j],ddstp[j]]))
            Q_=diagm(Q_seg)
            a_=hstack(a_seg)
            Ceq_=diagm(Ceq_seg)
            beq_= hstack(beq_seg)

        if i > 0 and i < n_seg-1:#middle segment
            for j in range(dim):
                Q_seg.append(Acc(duration[i]))
                a_seg.append(np.zeros(6))
                #judge here by SAT about the acc smoothness
                Ceq_sm.append(vstack([hstack([nt(duration[i-1]),np.zeros((1,6*(dim-1))),-nt(0)]),
                                    hstack([dnt(duration[i-1]),np.zeros((1,6*(dim-1))),-dnt(0)]),
                                    hstack([ddnt(duration[i-1]),np.zeros((1,6*(dim-1))),-ddnt(0)])]))
                beq_sm.append(hstack([0,0,0]))
            Q_=diagm(Q_seg)
            a_=hstack(a_seg)
            Ceq_sm_=diagm(Ceq_sm,6*dim)
            beq_sm_= hstack(beq_sm)


        if i == n_seg-1:#final segment 
            for j in range(dim):
                # coeff = 1
                # coeff_1= 0.1
                # traj_Q,traj_a = Q_traj(nt,[0.1],[p1[j]])
                Q_seg.append(Acc(duration[i])+nt(duration[i]).T@nt(duration[i]))
                a_seg.append(np.zeros(6)+(nt(duration[i])*fp[j]).reshape(-1))
                #judge here by SAT about the acc smoothness
                Ceq_sm.append(vstack([hstack([nt(duration[i-1]),np.zeros((1,6*(dim-1))),-nt(0)]),
                                    hstack([dnt(duration[i-1]),np.zeros((1,6*(dim-1))),-dnt(0)]),
                                    hstack([ddnt(duration[i-1]),np.zeros((1,6*(dim-1))),-ddnt(0)])]))
                beq_sm.append(hstack([0,0,0]))
                #
                Ciq_seg.append(vstack([nt(duration[i]),-nt(duration[i])]))
                biq_seg.append(hstack([fp[j]-delta,-(fp[j]+delta)]))
            #
            Q_=diagm(Q_seg)
            a_=hstack(a_seg)
            Ciq_ = diagm(Ciq_seg)
            biq_ = hstack(biq_seg)
            Ceq_sm_=diagm(Ceq_sm,6*dim)
            beq_sm_= hstack(beq_sm)

        #calculate Q,a,c,b, eq constraints
        if not Q_ is None:
            Q_all= diagm([Q_all,Q_],Q_all.shape[1]-i*6*dim)
            a_all=diagv([a_all,a_])
        if not Ceq_sm_ is None:
            Ceq_all= diagm([Ceq_all,Ceq_sm_],Ceq_all.shape[1]-(i-1)*6*dim)
            beq_all=diagv([beq_all,beq_sm_])
        if not Ceq_ is None:
            Ceq_all= diagm([Ceq_all,Ceq_],Ceq_all.shape[1]-i*6*dim)
            beq_all=diagv([beq_all,beq_])
        else:
            Ceq_all= hstack([Ceq_all,np.zeros((Ceq_all.shape[0],(i+1)*6*dim-Ceq_all.shape[1]))])
        if not Ciq_ is None:
            Ciq_all= diagm([Ciq_all,Ciq_],Ciq_all.shape[1]-i*6*dim)
            biq_all=diagv([biq_all,biq_])
        else:
            Ciq_all= hstack([Ciq_all,np.zeros((Ciq_all.shape[0],(i+1)*6*dim-Ciq_all.shape[1]))])

    eqns = Ceq_all.shape[0]
    C = vstack([Ceq_all,Ciq_all])
    b = hstack([beq_all,biq_all])
    # x, f, xu, iters, lagr, iact = solve_qp(Q_all,a_all,C.T,b,eqns)
    x = qp_solve(Q_all, -a_all, -Ciq_all, -biq_all, Ceq_all, beq_all, solver="qpswift")
    # x = qp_solve(Q_all, -a_all, -Ciq_all, -biq_all, Ceq_all, beq_all, solver="quadprog")
    return x


def traj_show(duration,dim,coeff):
        ### need coeff,dim,n_seg
    N =100
    n_seg = len(duration)
    traj_p = np.zeros((dim,n_seg*N))
    traj_dp = np.zeros((dim,n_seg*N))
    traj_ddp = np.zeros((dim,n_seg*N))
    tot_time = np.zeros(n_seg*N)

    for i in range(n_seg):  
        time = np.linspace(0,duration[i],N)
        for j in range(N):
            tot_time[j+i*N] = tot_time[i*N-1]+time[j]
            for k in range(dim):
                traj_p[k,j+i*N] = nt(time[j])@coeff[i*6*dim+k*6:i*6*dim+(k+1)*6]
                traj_dp[k,j+i*N] = dnt(time[j])@coeff[i*6*dim+k*6:i*6*dim+(k+1)*6]
                traj_ddp[k,j+i*N] = ddnt(time[j])@coeff[i*6*dim+k*6:i*6*dim+(k+1)*6]

    LABEL={0:'x',1:'y',2:'z'}
    plt.figure()
    for j in range(dim):
        plt.plot(tot_time,traj_p[j,:],label='$pos_'+LABEL[j]+"$")
    plt.legend()
    plt.grid()

    plt.figure()
    for j in range(dim):
        plt.plot(tot_time,traj_dp[j,:],label='$vel_'+LABEL[j]+"$")
    plt.legend()
    plt.grid()

    plt.figure()
    for j in range(dim):
        plt.plot(tot_time,traj_ddp[j,:],label="$acc_"+LABEL[j]+"$")
    plt.legend()
    plt.grid()


    plt.figure()
    plt.plot(traj_p[0,:],traj_p[1,:])
    plt.grid()
    plt.show()
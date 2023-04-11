import numpy as np
import matplotlib.pyplot as plt
from quadprog import solve_qp
from numpy.linalg import matrix_rank as rank,inv
n = 2 #number of segments
d = 1 #number of coordinates
duration = [[1,1],
            [1,1]]#duration of each spline
# tot_t = sum(duration)#total horizon
dt = 0.01 #in seconds sample time

def diagm(matlist: list,rcol = 0):
    mat  = None
    for it in matlist :
        if mat is None:
            mat = it
        else:
            mat = np.block([[mat,np.zeros((mat.shape[0],it.shape[1]-rcol))],
                            [np.zeros((it.shape[0],mat.shape[1]-rcol)),it]])
    return mat

def Acc(T,alpha=1e-8):
    return np.array([[400.0/7.0*pow(T,7),40*pow(T,6),120.0/5.0*pow(T,5),10*pow(T,4),0,0],
                     [40*pow(T,6),144/5*pow(T,5),18*pow(T,4),8*pow(T,3),0,0],
                     [120.0/5.0*pow(T,5),18*pow(T,4),12*pow(T,3),6*pow(T,2),0,0],
                     [10*pow(T,4),8*pow(T,3),6*pow(T,2),4*T,0,0],
                     [0,0,0,0,alpha,0],
                     [0,0,0,0,0,alpha]])

def nt(t):
    return np.array([[pow(t,5),pow(t,4),pow(t,3),pow(t,2),t,1]]) 

def dnt(t):
    return np.array([[5*pow(t,4),4*pow(t,3),3*pow(t,2),2*t,1,0]]) 

def ddnt(t):
    return np.array([[20*pow(t,3),12*pow(t,2),6*t,2,0,0]]) 

t = 1
Tmat = np.zeros((0,0))
dTmat = np.zeros((0,0))
ddTmat = np.zeros((0,0))
Accmat = np.zeros((0,0))
cnst = np.zeros((0,0))#smooth constraints
dcnst = np.zeros((0,0))#smooth constraints
ddcnst = np.zeros((0,0))#smooth constraints
bcnst = np.zeros(0)
dbcnst = np.zeros(0)
ddbcnst= np.zeros(0)

pr =np.zeros(0)

stp = np.array([0,0,0])
dstp = np.array([0,0,0])
ddstp = np.array([0,0,0])


def path(xy,t):
    if xy == 0 :
        return 0.5*t

#how to count the time
for i in range(n):
    #end time
    if i == 0 :
        for j in range(d):
            cnst = diagm([cnst,nt(0)])
            dcnst = diagm([dcnst,dnt(0)])
            ddcnst = diagm([ddcnst,ddnt(0)])
            bcnst = np.hstack([bcnst,stp[j]])
            dbcnst = np.hstack([dbcnst,dstp[j]])
            ddbcnst = np.hstack([ddbcnst,ddstp[j]])
    for j in range(d):
        Tmat = diagm([Tmat,nt(t)])
        dTmat = diagm([dTmat,dnt(t)])
        ddTmat = diagm([ddTmat,ddnt(t)])
        pr = np.hstack([pr,path(j,t)])
        Accmat = diagm([Accmat,Acc(duration[i][j])])
        if n!=1 and i+1<n:
            cnst = diagm([cnst,np.hstack([nt(duration[i][j]),np.zeros((1,6*(d-1))),-nt(0)])],6*d)
            dcnst = diagm([dcnst,np.hstack([dnt(duration[i][j]),np.zeros((1,6*(d-1))),-dnt(0)])],6*d)
            ddcnst = diagm([ddcnst,np.hstack([ddnt(duration[i][j]),np.zeros((1,6*(d-1))),-ddnt(0)])],6*d)
            bcnst = np.hstack([bcnst,0.])
            dbcnst = np.hstack([dbcnst,0.])
            ddbcnst = np.hstack([ddbcnst,0.])
        

    # if i+1 == n:
    #     for j in range(d):
    #         cnst = diagm([cnst,nt(t)])
    #         dcnst = diagm([dcnst,dnt(t)])
    #         ddcnst = diagm([ddcnst,ddnt(t)])
G = Tmat.T@Tmat+1e-9*np.eye(n*d*6)+Accmat
h = Tmat.T@pr
Aeq = np.vstack([cnst,dcnst,ddcnst])
beq = np.hstack([bcnst,dbcnst,ddbcnst])
xf, f, xu, iters, lagr, iact = solve_qp(G,h,Aeq.T,beq,beq.shape[0])


N =  1000
traj = np.zeros(N*n*d)
vel = np.zeros(N*n*d)
acc = np.zeros(N*n*d)
tot_time = np.linspace(0,sum(duration[:][0]),N*n*d)

for k in range(n):
    time = np.linspace(0,duration[k][0],N)
    for i in range(len(time)):
        traj[i+N*k] = nt(time[i])@xf[k*d*6:(k+1)*d*6]
        vel[i+N*k] = dnt(time[i])@xf[k*d*6:(k+1)*d*6]
        acc[i+N*k] = ddnt(time[i])@xf[k*d*6:(k+1)*d*6]

plt.plot(tot_time,traj,label='traj')
plt.plot(tot_time,vel,label='vel')
plt.plot(tot_time,acc,label='acc')
plt.legend()
plt.show()
import numpy as np
import matplotlib.pyplot as plt
from quadprog import solve_qp
from numpy.linalg import matrix_rank as rank,inv

p_s = np.array([0,0,0])
dp_s = np.array([0,0,0])
ddp_s = np.array([0,0,0])

p_m = np.array([0,0,1])
dp_m = np.array([0,0,0])
ddp_m = np.array([0,0,0])

p_e = np.array([0,0,0])
dp_e = np.array([0,0,0])
ddp_e = np.array([0,0,0])

T = 1

def Acc(T,alpha=1e-8):
    return np.array([[400.0/7.0*pow(T,7),40*pow(T,6),120.0/5.0*pow(T,5),10*pow(T,4),0,0],
                     [40*pow(T,6),144/5*pow(T,5),18*pow(T,4),8*pow(T,3),0,0],
                     [120.0/5.0*pow(T,5),18*pow(T,4),12*pow(T,3),6*pow(T,2),0,0],
                     [10*pow(T,4),8*pow(T,3),6*pow(T,2),4*T,0,0],
                     [0,0,0,0,alpha,0],
                     [0,0,0,0,0,alpha]])


def mnt(t):
    def nt(t):
        return np.array([pow(t,5),pow(t,4),pow(t,3),pow(t,2),t,1]) 
    return np.block([
        [nt(t),np.zeros(6),np.zeros(6)],
        [np.zeros(6),nt(t),np.zeros(6)],
        [np.zeros(6),np.zeros(6),nt(t)]])



def mdnt(t):
    def dnt(t):
        return np.array([5*pow(t,4),4*pow(t,3),3*pow(t,2),2*t,1,0]) 
    return np.block([
        [dnt(t),np.zeros(6),np.zeros(6)],
        [np.zeros(6),dnt(t),np.zeros(6)],
        [np.zeros(6),np.zeros(6),dnt(t)]])


def mddnt(t):
    def ddnt(t):
        return np.array([20*pow(t,3),12*pow(t,2),6*t,2,0,0]) 
    return np.block([
        [ddnt(t),np.zeros(6),np.zeros(6)],
        [np.zeros(6),ddnt(t),np.zeros(6)],
        [np.zeros(6),np.zeros(6),ddnt(t)]])



def pt(p_s,dp_s,ddp_s,p_m,dp_m,ddp_m,p_e,dp_e,ddp_e):
    A = np.vstack([mnt(0),mdnt(0),mddnt(0),mnt(T),mdnt(T),mddnt(T)])
    A_ = np.block([[A,np.zeros(A.shape)],[np.zeros(A.shape),A]])
    b = np.hstack([p_s,dp_s,ddp_s,p_m,dp_m,ddp_m,p_m,dp_m,ddp_m,p_e,dp_e,ddp_e])
    G = A_.T@A_
    h = A_.T@b
    xf, f, xu, iters, lagr, iact = solve_qp(G,h)

    return xf


def pt1(p_s,dp_s,ddp_s,p_m,dp_m,ddp_m,p_e,dp_e,ddp_e):
    def nt(t):
        return np.array([pow(t,5),pow(t,4),pow(t,3),pow(t,2),t,1]) 
    def ddnt(t):
        return np.array([20*pow(t,3),12*pow(t,2),6*t,2,0,0]) 
    def dnt(t):
        return np.array([5*pow(t,4),4*pow(t,3),3*pow(t,2),2*t,1,0]) 
    t = 0.1
    p = np.array([1])
    dp = np.array([1])
    ddp = np.array([1])
    G = Acc(1)+nt(t).T@nt(t)+dnt(t).T@dnt(t)+ddnt(t).T@ddnt(t)
    h = nt(t).T@p+dnt(t).T@dp + ddnt(t).T@ddp
    Aeq = np.vstack([nt(0.),dnt(0.),ddnt(0.)])
    beq = np.array([1,1,0],dtype=np.float64)
    xf, f, xu, iters, lagr, iact = solve_qp(G,h,Aeq.T,beq,3)
    return xf

#because of the chance of requirement we can set the velocities of this
#fifth order hermite spline
def nt(t):
    return np.array([[pow(t,5),pow(t,4),pow(t,3),pow(t,2),t,1]]) 
def ddnt(t):
    return np.array([[20*pow(t,3),12*pow(t,2),6*t,2,0,0]]) 
def dnt(t):
    return np.array([[5*pow(t,4),4*pow(t,3),3*pow(t,2),2*t,1,0]]) 
T = 0.25
t = 0.2
p = np.array([0.25*t*t])
dp = np.array([0.5*t])
ddp = np.array([0.5])
G = nt(t).T@nt(t)+dnt(t).T@dnt(t)+ddnt(t).T@ddnt(t)+1e-9*Acc(T)#+1e-9*np.eye(6)
h = nt(t).T@p+dnt(t).T@dp + ddnt(t).T@ddp
Aeq = np.vstack([nt(0.),dnt(0.),ddnt(0.)])
beq = np.array([0,0,0],dtype=np.float64)
xf, f, xu, iters, lagr, iact = solve_qp(G,h,Aeq.T,beq,3)
# xf, f, xu, iters, lagr, iact = solve_qp(G,h)
print(p,dp,ddp)
print(nt(t)@xf,dnt(t)@xf,ddnt(t)@xf)
print(nt(T)@xf,dnt(T)@xf,ddnt(T)@xf)
N =  100
time = np.linspace(0,T,N)
traj = np.zeros(N)
vel = np.zeros(N)
acc = np.zeros(N)
for i in range(len(time)):
    traj[i] = nt(time[i])@xf
    vel[i] = dnt(time[i])@xf
    acc[i] = ddnt(time[i])@xf
plt.plot(time,traj,label='traj')
plt.plot(time,vel,label='vel')
plt.plot(time,acc,label='acc')
plt.legend()
plt.show()

# ax = plt.figure().add_subplot(projection='3d')

# x = np.zeros(200)
# y = np.zeros(200)
# z = np.zeros(200)
# traj = np.zeros((3,200))
# vel  = np.zeros((3,200))
# # Prepare arrays x, y, z
# t = np.linspace(0,1,100)

# for i in range(200):
#     if(i<100):
#         traj[:,i] = mnt(t[i])@coeff[:18]
#         vel[:,i] = mdnt(t[i])@coeff[:18]
#     else:
#         traj[:,i] = mnt(t[i%100])@coeff[18:]
#         vel[:,i] = mdnt(t[i%100])@coeff[18:]



# x = traj[0,:]
# y = traj[1,:]
# z = traj[2,:]

# u = vel[0,:]
# v = vel[1,:]
# w = vel[2,:]

# ax.plot(x, y, z, label='parametric curve')
# ax.quiver(x[0], y[0], z[0], u[0], v[0], w[0], length=0.2, normalize=True,color='black')
# ax.quiver(x[100], y[100], z[100], u[100], v[100], w[100], length=0.2, normalize=True,color='black')
# ax.quiver(x[-1], y[-1], z[-1], u[-1], v[-1], w[-1], length=0.2, normalize=True,color='black')
# ax.legend()
# plt.show()

n = 2
duration = [1,1]


def diagm(matlist: list):
    mat  = None
    for it in matlist :
        if mat is None:
            mat = it
        else:
            mat = np.block([[mat,np.zeros((mat.shape[0],it.shape[1]))],
                            [np.zeros((it.shape[0],mat.shape[1])),it]])
    return mat
#n
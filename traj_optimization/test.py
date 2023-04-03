import numpy as np
import matplotlib.pyplot as plt
from quadprog import solve_qp
from numpy.linalg import matrix_rank as rank
from numpy.linalg import inv

p_s = np.array([0,0,0])
dp_s = np.array([0,0,0])
ddp_s = np.array([0,0,0])

p_m = np.array([1,1,1])
dp_m = np.array([1,1,0])
ddp_m = np.array([0,0,0])

p_e = np.array([2,2,0])
dp_e = np.array([0,0,0])
ddp_e = np.array([0,0,0])

T = 1


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



def pt():
    A = np.vstack([mnt(0),mdnt(0),mddnt(0),mnt(T),mdnt(T),mddnt(T)])
    A_ = np.block([[A,np.zeros(A.shape)],[np.zeros(A.shape),A]])
    b = np.hstack([p_s,dp_s,ddp_s,p_m,dp_m,ddp_m,p_m,dp_m,ddp_m,p_e,dp_e,ddp_e])
    G = A_.T@A_
    h = A_.T@b
    xf, f, xu, iters, lagr, iact = solve_qp(G,h)
    
    return xf



#because of the chance of requirement we can set the velocities of this
#fifth order hermite spline
coeff = pt()
ax = plt.figure().add_subplot(projection='3d')

x = np.zeros(200)
y = np.zeros(200)
z = np.zeros(200)
traj = np.zeros((3,200))
vel  = np.zeros((3,200))
# Prepare arrays x, y, z
t = np.linspace(0,1,100)

for i in range(200):
    if(i<100):
        traj[:,i] = mnt(t[i])@coeff[:18]
        vel[:,i] = mdnt(t[i])@coeff[:18]
    else:
        traj[:,i] = mnt(t[i%100])@coeff[18:]
        vel[:,i] = mdnt(t[i%100])@coeff[18:]



x = traj[0,:]
y = traj[1,:]
z = traj[2,:]

u = vel[0,:]
v = vel[1,:]
w = vel[2,:]

ax.plot(x, y, z, label='parametric curve')
ax.quiver(x[0], y[0], z[0], u[0], v[0], w[0], length=0.2, normalize=True,color='black')
ax.quiver(x[100], y[100], z[100], u[100], v[100], w[100], length=0.2, normalize=True,color='black')
ax.quiver(x[-1], y[-1], z[-1], u[-1], v[-1], w[-1], length=0.2, normalize=True,color='black')
ax.legend()

plt.show()


#yici
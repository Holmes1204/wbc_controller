import numpy as np
import matplotlib.pyplot as plt
from quadprog import solve_qp
from numpy.linalg import matrix_rank as rank,inv
from numpy import hstack,vstack
from plot_polygon import plot_convex_shape

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
    b = hstack([p_s,dp_s,ddp_s,p_m,dp_m,ddp_m,p_m,dp_m,ddp_m,p_e,dp_e,ddp_e])
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

def diagm(matlist: list,rcol = 0):
    mat  = None
    for it in matlist :
        if mat is None:
            mat = it
        else:
            mat = np.block([[mat,np.zeros((mat.shape[0],it.shape[1]-rcol))],
                            [np.zeros((it.shape[0],mat.shape[1]-rcol)),it]])
    return mat
#n
def diagv(veclist: list,rcol = 0):
    vec  = None
    for it in veclist :
        if vec is None:
            vec = it
        else:
            vec = np.block([vec,np.zeros(rcol),it])
    return vec


#ordered sequnece
def reduce_convex(polygon_set,s=0.1,w=0.05):
    """reduce the shape of the support polygons"""
    reduce_polygon = []
    for (vertex,duration) in polygon_set:
        num_vertices = len(vertex)
        new_vertex = vertex.copy()
        if num_vertices>2:
            print(">2")
            c = np.zeros(num_vertices)
            vec_n = np.zeros((num_vertices,2))
            for i in range(num_vertices):
                direct = vertex[(i + 1) % num_vertices] - vertex[i]
                normal = np.array([direct[1], -direct[0]])#inside polygons direct
                normal /= np.linalg.norm(normal)
                new_vertex[i] +=s*normal
                vec_n[i] = normal
                c[i] = -normal@new_vertex[i]
                # new_vertex[(i + 1) % num_vertices] +=s*normal
            for i in range(num_vertices):
                new_vertex[i] = -np.linalg.inv(vec_n[[i,(i+1)% num_vertices]])@c[[i,(i+1)% num_vertices]]
            reduce_polygon.append((new_vertex,duration))
        else:
            direct = vertex[1] - vertex[0]
            direct /=np.linalg.norm(direct)
            normal = np.array([direct[1], -direct[0]])#inside polygons direct
            normal /= np.linalg.norm(normal)
            v1 = vertex[0]+w*normal+s*direct
            v2 = vertex[1]+w*normal-s*direct
            v3 = vertex[1]-w*normal-s*direct
            v4 = vertex[0]-w*normal+s*direct
            reduce_polygon.append(([v1,v2,v3,v4],duration))
            print(2)
    return reduce_polygon

def regular_path_generation(stp,dstp,ddstp,fp):
    """ for some problem provide the solutions
        min 1/2 x^T G x  -a^Tx
        s.t. C.T x>= b
    """
    n = 2 #number of segments
    d = 1 #number of coordinates
    t = 1
    Q_all = np.zeros((0,0))
    a_all = np.zeros(0)
    Ceq_all = np.zeros((0,0))
    beq_all = np.zeros(0)
    Ciq_all = np.zeros((0,0))
    biq_all = np.zeros(0)
    for i in range(n):
        #for the sampling of the line, we have some others formulations
        Q = None
        a = None
        Ceq = None
        beq = None
        #for the qp slover
        Ciq = None
        biq = None
        if i == 0 :#first segment 
            Accmat = diagm([Acc(duration[i]) for j in range(d)])
            #
            Ceq = diagm([nt(0) for j in range(d)])
            beq = hstack([stp[j] for j in range(d)])
            #
            dCeq = diagm([dnt(0) for j in range(d)])
            dbeq = hstack([dstp[j] for j in range(d)])
            #
            ddCeq = diagm([ddnt(0) for j in range(d)])
            ddbeq = hstack([ddstp[j] for j in range(d)])
            #
            Q= Accmat
            # a=hstack([np.zeros(6*d)])
            Ceq=vstack([Ceq,dCeq,ddCeq])
            beq=hstack([beq,dbeq,ddbeq])
            # Ciq
            # biq


        if n > 1 and i < n-1:#middle segment
            Accmat = diagm([Acc(duration[i]) for j in range(d)])
            #
            Ceq = diagm([hstack([nt(duration[i]),np.zeros((1,6*(d-1))),-nt(0)]) for j in range(d)],6*d)     
            beq = hstack([0. for j in range(d)])
            #
            dCeq = diagm([hstack([dnt(duration[i]),np.zeros((1,6*(d-1))),-dnt(0)]) for j in range(d)],6*d)
            dbeq = hstack([0. for j in range(d)])
            #
            ddCeq = diagm([hstack([ddnt(duration[i]),np.zeros((1,6*(d-1))),-ddnt(0)])for j in range(d)],6*d)
            ddbeq = hstack([0. for j in range(d)])
            #
            Q=Accmat
            # a
            Ceq=vstack([Ceq,dCeq,ddCeq])
            beq=hstack([beq,dbeq,ddbeq])
            # Ciq
            # biq


        if i == n-1:#final segment 
            Accmat = diagm([Acc(duration[i])for j in range(d)])
            Tm = diagm([nt(duration[i]).T@nt(duration[i]) for j in range(d)])
            am = hstack([nt(duration[i])*fp[j] for j in range(d)])
            # #
            # Ceq = diagm([nt(t)for j in range(d)])
            # beq = hstack([0. for j in range(d)])
            # #
            # dCeq = diagm([dnt(t)for j in range(d)])
            # dbeq = hstack([0. for j in range(d)])
            # #
            # ddCeq = diagm([ddnt(t)for j in range(d)])
            # ddbeq = hstack([0. for j in range(d)])
            Q=Accmat+Tm
            a=am
            # Ceq=vstack([Ceq,dCeq,ddCeq])
            # beq=hstack([beq,dbeq,ddbeq])
            # Ciq
            # biq

        #calculate Q,a,c,b, eq constraints
        if not Q is None:
            Q_all= diagm([Q_all,Q],Q_all.shape[1]-i*6*d)
        if not a is None:
            a_all=diagv([a_all,a],i*6*d-a_all.shape[0])
        if not Ceq is None:
            Ceq_all= diagm([Ceq_all,Ceq],Ceq_all.shape[1]-i*6*d)
            beq_all=diagv([beq_all,beq],i*6*d-beq_all.shape[0])
        if not Ciq is None:
            Ciq_all= diagm([Ciq_all,Ciq],Ciq_all.shape[1]-i*6*d)
            biq_all=diagv([biq_all,biq],i*6*d-biq_all.shape[0])
    

    return 

# T = 0.25
# t = 0.2
# p = np.array([0.25*t*t])
# dp = np.array([0.5*t])
# ddp = np.array([0.5])
# G = nt(t).T@nt(t)+dnt(t).T@dnt(t)+ddnt(t).T@ddnt(t)+1e-9*Acc(T)#+1e-9*np.eye(6)
# h = nt(t).T@p+dnt(t).T@dp + ddnt(t).T@ddp
# Aeq = np.vstack([nt(0.),dnt(0.),ddnt(0.)])
# beq = np.array([0,0,0],dtype=np.float64)
# xf, f, xu, iters, lagr, iact = solve_qp(G,h,Aeq.T,beq,3)
# # xf, f, xu, iters, lagr, iact = solve_qp(G,h)
# print(p,dp,ddp)
# print(nt(t)@xf,dnt(t)@xf,ddnt(t)@xf)
# print(nt(T)@xf,dnt(T)@xf,ddnt(T)@xf)
# N =  100
# time = np.linspace(0,T,N)
# traj = np.zeros(N)
# vel = np.zeros(N)
# acc = np.zeros(N)
# for i in range(len(time)):
#     traj[i] = nt(time[i])@xf
#     vel[i] = dnt(time[i])@xf
#     acc[i] = ddnt(time[i])@xf
# plt.plot(time,traj,label='traj')
# plt.plot(time,vel,label='vel')
# plt.plot(time,acc,label='acc')
# plt.legend()
# plt.show()
# plt.figure()

#(vertexs:[[x,y],],duration:t)
#just for planar trajectory
dx = 0.2
dy = 0.2
#leg sequence was changed, and the  3rd column is the RR leg, and the 4th column is the RF leg
polygons = [(np.array([[0.5,   0.25],[0.5   ,-0.25] ,[-0.5   ,-0.25],[-0.5   ,0.25]]),0.05),#0.05-0.0 all down until LF lift
            (np.array([              [0.5   ,-0.25] ,[-0.5   ,-0.25],[-0.5   ,0.25]]),0.15),#0.2-0.05,LF lift until RR lift
            (np.array([              [0.5   ,-0.25]                 ,[-0.5   ,0.25]]),0.10),#0.3-0.2 RR lift until LF touch
            (np.array([[0.5+dx,0.25],[0.5   ,-0.25]                 ,[-0.5   ,0.25]]),0.15),#0.45-0.3LF touch until RR touch
            (np.array([[0.5+dx,0.25],[0.5   ,-0.25] ,[-0.5+dx,-0.25],[-0.5   ,0.25]]),0.10),#0.55-0.45 all touch until LR lift
            (np.array([[0.5+dx,0.25],                [-0.5+dx,-0.25],[-0.5   ,0.25]]),0.15),#0.70-0.55 FR lift until RL lift
            (np.array([[0.5+dx,0.25],                [-0.5+dx,-0.25]               ]),0.10),#0.80-0.70 RL lift until FR touch
            (np.array([[0.5+dx,0.25],[0.5+dx,-0.25] ,[-0.5+dx,-0.25]               ]),0.15),#0.95-0.8FR touch until RL touch
            (np.array([[0.5+dx,0.25],[0.5+dx,-0.25] ,[-0.5+dx,-0.25],[-0.5+dx,0.25]]),0.05)]#1.0 -0.95 RL touch until next sequnece

# polygons = [([[0.5,   0.25],[0.5   ,-0.25] ,[-0.5   ,0.25],[-0.5   ,-0.25]],0.05),#0.05-0.0 all down until LF lift
#             ([              [0.5   ,-0.25] ,[-0.5   ,0.25],[-0.5   ,-0.25]],0.15),#0.2-0.05,LF lift until RR lift
#             ([              [0.5   ,-0.25] ,[-0.5   ,0.25]                ],0.10),#0.3-0.2 RR lift until LF touch
#             ([[0.5+dx,0.25],[0.5   ,-0.25] ,[-0.5   ,0.25]                ],0.15),#0.45-0.3LF touch until RR touch
#             ([[0.5+dx,0.25],[0.5   ,-0.25] ,[-0.5   ,0.25],[-0.5+dx,-0.25]],0.10),#0.55-0.45 all touch until LR lift
#             ([[0.5+dx,0.25],                [-0.5   ,0.25],[-0.5+dx,-0.25]],0.15),#0.70-0.55 FR lift until RL lift
#             ([[0.5+dx,0.25],                               [-0.5+dx,-0.25]],0.10),#0.80-0.70 RL lift until FR touch
#             ([[0.5+dx,0.25],[0.5+dx,-0.25]                ,[-0.5+dx,-0.25]],0.15),#0.95-0.8FR touch until RL touch
#             ([[0.5+dx,0.25],[0.5+dx,-0.25] ,[-0.5+dx,0.25],[-0.5+dx,-0.25]],0.05)]#1.0 -0.95 RL touch until next sequnece
 

    
# plot_convex_shape(polygons[0][0],'b')

a = reduce_convex(polygons)
# plt.xlim([-0.8,0.8])
# plt.ylim([-0.8,0.8])
# plt.grid()
# # plot_convex_shape(a[0][0],'r')
# for (j,i) in polygons:
#     plot_convex_shape(j,'k')
plt.figure()
plt.xlim([-0.6,0.6])
plt.ylim([-0.6,0.6])
plt.grid()
for (j,i) in a:
    plot_convex_shape(j,'r')

#calcualate the regularize  path




# plt.xlim([-0.8,0.8])
# plt.ylim([-0.8,0.8])
# plt.grid()
# plt.show()

# for l in range(len(polygons)):
#     plt.figure()
#     plot_convex_shape(polygons[l][0],'b')
#     plot_convex_shape(a[l][0],'r')
#     plt.xlim([-0.8,0.8])
#     plt.ylim([-0.8,0.8])
#     plt.grid()
#     plt.show()
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

duration = [1,1,1,1,1]#duration of each spline,suppose that each segment has the same duration in each coordinate
# tot_t = sum(duration)#total horizon
dt = 0.01 #in seconds sample time


p = np.array([0,0,0])
dp = np.array([0,0,0])
ddp = np.array([0,0,0])
f_p = np.array([0,0,0])
regular_path_generation(p,dp,ddp,f_p)
#matrix structure

import numpy as np
import matplotlib.pyplot as plt
from quadprog import solve_qp
from numpy.linalg import matrix_rank as rank,inv
from numpy import hstack,vstack
from plot_polygon import plot_convex_shape
from traj_opt import traj_opt

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
 


""" 
for some problem provide the solutions
min 1/2 x^T G x  -a^Tx
s.t. C.T x>= b
"""
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




"""
    changed a lot, will be improved 
"""
duration=[polygons[j][1] for j in range(len(polygons))]
c_point = [sum(polygons[j][0])/4 for j in range(len(polygons))]
stp=[0,2]
dstp=[0.,0.]
ddstp=[0.,0.]
fp=[1,0.5]
p1=[0.5,1.25]
#
n_seg = 2 #number of segments
dim = 2 #number of coordinates
delta = 0.01
xf =traj_opt(n_seg,dim,duration,stp,dstp,ddstp,fp)


### need xf,dim,n_seg
N =100
traj_p = np.zeros((dim,n_seg*N))
traj_dp = np.zeros((dim,n_seg*N))
traj_ddp = np.zeros((dim,n_seg*N))
tot_time = np.zeros(n_seg*N)

for i in range(n_seg):  
    time = np.linspace(0,duration[i],N)
    for j in range(N):
        tot_time[j+i*N] = tot_time[i*N-1]+time[j]
        for k in range(dim):
            traj_p[k,j+i*N] = nt(time[j])@xf[i*6*dim+k*6:i*6*dim+(k+1)*6]
            traj_dp[k,j+i*N] = dnt(time[j])@xf[i*6*dim+k*6:i*6*dim+(k+1)*6]
            traj_ddp[k,j+i*N] = ddnt(time[j])@xf[i*6*dim+k*6:i*6*dim+(k+1)*6]

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
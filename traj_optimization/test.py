import numpy as np
import matplotlib.pyplot as plt
from quadprog import solve_qp
from numpy.linalg import matrix_rank as rank,inv
from numpy import hstack,vstack
from plot_polygon import plot_convex_shape


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
#n
def diagv(veclist: list,rcol = 0):
    vec  = None
    for it in veclist :
        if vec is None:
            vec = it
        else:
            vec = np.block([vec,np.zeros(rcol),it])
    return vec

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
delta = 0.01
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



n = 2 #number of segments
d = 2 #number of coordinates

duration=[polygons[j][1] for j in range(len(polygons))]
c_point = [sum(polygons[j][0])/4 for j in range(len(polygons))]
stp=c_point[0]
dstp=[0.,0.,0.]
ddstp=[0.,0.,0.]
fp=c_point[1]
#
Q_all = np.zeros((0,0))
a_all = np.zeros(0)
Ceq_all = np.zeros((0,0))
beq_all = np.zeros(0)
Ciq_all = np.zeros((0,0))
biq_all = np.zeros(0)

for i in range(n):
    #for the sampling of the line, we have some others formulations
    Q_ = None
    a_ = None
    Ceq_ = None
    beq_ = None
    #for the qp slover
    Ciq_ = None
    biq_ = None
    if i == 0 :#first segment 
        Accmat = diagm([Acc(duration[i]) for j in range(d)])
        #
        pCeq = diagm([nt(0) for j in range(d)])
        pbeq = hstack([stp[j] for j in range(d)])
        #
        pdCeq = diagm([dnt(0) for j in range(d)])
        pdbeq = hstack([dstp[j] for j in range(d)])
        #
        pddCeq = diagm([ddnt(0) for j in range(d)])
        pddbeq = hstack([ddstp[j] for j in range(d)])
        #
        if n>1:
            Ceq = diagm([hstack([nt(duration[i]),np.zeros((1,6*(d-1))),-nt(0)]) for j in range(d)],6*d)     
            beq = hstack([0. for j in range(d)])
            #
            dCeq = diagm([hstack([dnt(duration[i]),np.zeros((1,6*(d-1))),-dnt(0)]) for j in range(d)],6*d)
            dbeq = hstack([0. for j in range(d)])
            #
            ddCeq = diagm([hstack([ddnt(duration[i]),np.zeros((1,6*(d-1))),-ddnt(0)])for j in range(d)],6*d)
            ddbeq = hstack([0. for j in range(d)])
        Q_= Accmat
        # a=hstack([np.zeros(6*d)])
        Ceq_=diagm([vstack([pCeq,pdCeq,pddCeq]),vstack([Ceq,dCeq,ddCeq])],6*d)
        beq_=hstack([pbeq,pdbeq,pddbeq,beq,dbeq,ddbeq])
        # Ciq
        # biq

    if i > 1 and i < n-1:#middle segment
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
        Q_=Accmat
        # a
        Ceq_=vstack([Ceq,dCeq,ddCeq])
        beq_=hstack([beq,dbeq,ddbeq])
        # Ciq
        # biq

    if i == n-1:#final segment 
        Accmat = diagm([Acc(duration[i])for j in range(d)])
        Tm = diagm([nt(duration[i]).T@nt(duration[i]) for j in range(d)])
        am = hstack([nt(duration[i])*fp[j] for j in range(d)])
        am = am.reshape(-1)
        #
        Ciq = diagm([vstack([nt(duration[i]),-nt(duration[i])]) for j in range(d)])
        biq = hstack([hstack([fp[j]-delta,-(fp[j]+delta)]) for j in range(d)])
        #
        # dCeq = diagm([dnt(t)for j in range(d)])
        # dbeq = hstack([0. for j in range(d)])
        # #
        # ddCeq = diagm([ddnt(t)for j in range(d)])
        # ddbeq = hstack([0. for j in range(d)])
        Q_=Accmat+Tm
        a_=am
        # Ceq=vstack([Ceq,dCeq,ddCeq])
        # beq=hstack([beq,dbeq,ddbeq])
        Ciq_ = Ciq
        biq_ = biq
    #calculate Q,a,c,b, eq constraints
    if not Q_ is None:
        Q_all= diagm([Q_all,Q_],Q_all.shape[1]-i*6*d)
    if not a_ is None:
        a_all=diagv([a_all,a_],i*6*d-a_all.shape[0])
    if not Ceq_ is None:
        Ceq_all= diagm([Ceq_all,Ceq_],Ceq_all.shape[1]-i*6*d)
        beq_all=diagv([beq_all,beq_])
    else:
        Ceq_all= hstack([Ceq_all,np.zeros((Ceq_all.shape[0],(i+1)*6*d-Ceq_all.shape[1]))])
    if not Ciq_ is None:
        Ciq_all= diagm([Ciq_all,Ciq_],Ciq_all.shape[1]-i*6*d)
        biq_all=diagv([biq_all,biq_])
    else:
        Ciq_all= hstack([Ciq_all,np.zeros((Ciq_all.shape[0],(i+1)*6*d-Ciq_all.shape[1]))])


eqns = Ceq_all.shape[0]
C = vstack([Ceq_all,Ciq_all])
b = hstack([beq_all,biq_all])
xf, f, xu, iters, lagr, iact = solve_qp(Q_all,a_all,C.T,b,eqns)

N =10
traj_p = np.zeros((d,n*N))
traj_dp = np.zeros((d,n*N))
traj_ddp = np.zeros((d,n*N))
tot_time = np.zeros(n*N)

for i in range(n):  
    time = np.linspace(0,duration[i],N)
    for j in range(N):
        tot_time[j+i*N] = tot_time[i*N-1]+time[j]
        for k in range(d):
            traj_p[k,j+i*N] = nt(time[j])@xf[i*6+k*6:i*6+(k+1)*6]
            traj_dp[k,j+i*N] = dnt(time[j])@xf[i*6+k*6:i*6+(k+1)*6]
            traj_ddp[k,j+i*N] = ddnt(time[j])@xf[i*6+k*6:i*6+(k+1)*6]

plt.figure()
plt.plot(tot_time,traj_p[1,:],label='pos')
plt.legend()
plt.grid()

plt.figure()
plt.plot(tot_time,traj_dp[1,:],label='vel')
plt.legend()
plt.grid()

plt.figure()
plt.plot(tot_time,traj_ddp[1,:],label="acc")
plt.legend()
plt.grid()
# plt.figure()
# plt.show(tra)
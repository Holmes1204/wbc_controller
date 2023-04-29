import numpy as np
import matplotlib.pyplot as plt
from quadprog import solve_qp
from numpy.linalg import matrix_rank as rank,inv
from numpy import hstack,vstack,array
from copy import deepcopy
from traj import traj_opt,traj_show

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

def zmp1(t):
    z =  0.3
    g =  9.8
    ddz=  0.0
    matrix = np.array([1,-z/(g+ddz)])@np.vstack([nt(t),ddnt(t)])
    matrix.resize(1,6)
    return matrix

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
s.t. 
    C.T x>= b
"""
def plot_convex_shape(vertices_, color='k'):
    """Plot a convex shape given its vertices using matplotlib."""
    vertices = deepcopy(vertices_)
    for i in range(len(vertices)-1,-1,-1):
        if vertices[i] is None:
            vertices.pop(i)
    num_vertices = len(vertices)
    x = [vertices[i][0] for i in range(num_vertices)]
    y = [vertices[i][1] for i in range(num_vertices)]
    # plt.fill(x, y, color=color)
    x.append(vertices[0][0])  # Add the first vertex to close the shape
    y.append(vertices[0][1])  # Add the first vertex to close the shape
    plt.plot(x, y,color+"-.")


def plot_convex_quiver(vertices,edge=None, color='k'):
    """Plot a convex shape and its normal vector"""
    num_vertices = len(vertices)
    x = [vertices[i][0] for i in range(num_vertices)]
    y = [vertices[i][1] for i in range(num_vertices)]
    x.append(vertices[0][0])  # Add the first vertex to close the shape
    y.append(vertices[0][1])  # Add the first vertex to close the shape
    midx = [(x[i]+x[i+1])/2.0 for i in range(len(x)-1)]
    midy = [(y[i]+y[i+1])/2.0 for i in range(len(y)-1)]
    plt.plot(x, y, color+"-.")
    if edge is not None:
        plt.quiver(midx,midy,edge[:,0],edge[:,1],color='k')


def reduce_convex(polygon_set,s=0.025,w=0.025):
    """reduce the shape of the support polygons"""
    def calcualte_p(num,modified_vertex,origin_vertex):
        c = np.zeros(num)
        vec_n = np.zeros((num,2))
        for i in range(num):
            direct = origin_vertex[(i + 1) % num] - origin_vertex[i]
            normal = np.array([direct[1], -direct[0]])#inside polygons direct
            normal /= np.linalg.norm(normal)
            modified_vertex[i] +=s*normal
            vec_n[i] = normal
            c[i] = -normal@modified_vertex[i]
            # modified_vertex[(i + 1) % num] +=s*normal
        for i in range(num):
            modified_vertex[(i+1)%num] = -np.linalg.inv(vec_n[[i,(i+1)% num]])@c[[i,(i+1)% num]]
        reduce_polygon.append((modified_vertex,duration))
        edge.append(np.hstack([vec_n,c.reshape(1,-1).T]))

    #
    polygons_ = deepcopy(polygon_set)
    reduce_polygon = []
    edge = []
    #get the normal vector and the bias of each edge
    for [vertex,duration] in polygons_:
        vertex_ = []
        new_vertex = []
        for k in vertex:
            if k is not None:
                new_vertex.append(k.copy())
                vertex_.append(k.copy())
        num_vertices = len(new_vertex)
        if num_vertices>2:
            calcualte_p(num_vertices,new_vertex,vertex_)
        else:
            direct = new_vertex[1] - new_vertex[0]
            direct /=np.linalg.norm(direct)
            normal = np.array([direct[1], -direct[0]])#inside polygons direct
            normal /= np.linalg.norm(normal)
            v1 = new_vertex[0]+(s+w)*normal
            v2 = new_vertex[0]-(s+w)*normal
            v3 = new_vertex[1]-(s+w)*normal
            v4 = new_vertex[1]+(s+w)*normal
            ver_ = [v1,v2,v3,v4]
            calcualte_p(4,deepcopy(ver_),ver_)
    return reduce_polygon,edge




#leg sequence is changed, and the  3rd column is the RR leg, and the 4th column is the RF leg
# dx = 0.2
# dy = 0.2
# polygons = [(np.array([[0.5,   0.25],[0.5   ,-0.25] ,[-0.5   ,-0.25],[-0.5   ,0.25]]),0.05),#0.05-0.0 all down until LF lift
#             (np.array([              [0.5   ,-0.25] ,[-0.5   ,-0.25],[-0.5   ,0.25]]),0.15),#0.2-0.05,LF lift until RR lift
#             (np.array([              [0.5   ,-0.25]                 ,[-0.5   ,0.25]]),0.10),#0.3-0.2 RR lift until LF touch
#             (np.array([[0.5+dx,0.25],[0.5   ,-0.25]                 ,[-0.5   ,0.25]]),0.15),#0.45-0.3LF touch until RR touch
#             (np.array([[0.5+dx,0.25],[0.5   ,-0.25] ,[-0.5+dx,-0.25],[-0.5   ,0.25]]),0.10),#0.55-0.45 all touch until LR lift
#             (np.array([[0.5+dx,0.25],                [-0.5+dx,-0.25],[-0.5   ,0.25]]),0.15),#0.70-0.55 FR lift until RL lift
#             (np.array([[0.5+dx,0.25],                [-0.5+dx,-0.25]               ]),0.10),#0.80-0.70 RL lift until FR touch
#             (np.array([[0.5+dx,0.25],[0.5+dx,-0.25] ,[-0.5+dx,-0.25]               ]),0.15),#0.95-0.8FR touch until RL touch
#             (np.array([[0.5+dx,0.25],[0.5+dx,-0.25] ,[-0.5+dx,-0.25],[-0.5+dx,0.25]]),0.05)]#1.0 -0.95 RL touch until next sequnece

# polygons = [[[[0.5,   0.25],[0.5   ,-0.25] ,[-0.5   ,-0.25],[-0.5   ,0.25]],0.05],#0.05-0.0 all down until LF lift
#             [[None         ,[0.5   ,-0.25] ,[-0.5   ,-0.25],[-0.5   ,0.25]],0.15],#0.2-0.05,LF lift until RR lift
#             [[None         ,[0.5   ,-0.25] ,None           ,[-0.5   ,0.25]],0.10],#0.3-0.2 RR lift until LF touch
#             [[[0.5+dx,0.25],[0.5   ,-0.25] ,None           ,[-0.5   ,0.25]],0.15],#0.45-0.3LF touch until RR touch
#             [[[0.5+dx,0.25],[0.5   ,-0.25] ,[-0.5+dx,-0.25],[-0.5   ,0.25]],0.10],#0.55-0.45 all touch until LR lift
#             [[[0.5+dx,0.25],None           ,[-0.5+dx,-0.25],[-0.5   ,0.25]],0.15],#0.70-0.55 FR lift until RL lift
#             [[[0.5+dx,0.25],None           ,[-0.5+dx,-0.25],None          ],0.10],#0.80-0.70 RL lift until FR touch
#             [[[0.5+dx,0.25],[0.5+dx,-0.25] ,[-0.5+dx,-0.25],None          ],0.15],#0.95-0.8FR touch until RL touch
#             [[[0.5+dx,0.25],[0.5+dx,-0.25] ,[-0.5+dx,-0.25],[-0.5+dx,0.25]],0.05]]#1.0 -0.95 RL touch until next sequnece
# for i in range(len(polygons)):
#     for j in range(len(polygons[i][0])):
#         if polygons[i][0][j] is not None:
#             polygons[i][0][j] = np.array(polygons[i][0][j])
#         else:
#             polygons[i][0][j] = None

polygons = [[[array([0.2, 0.1]),
   array([ 0.2, -0.1]),
   array([-0.2, -0.1]),
   array([-0.2,  0.1])],0.05],
 [[None, array([ 0.2, -0.1]), array([-0.2, -0.1]), array([-0.2,  0.1])],
  0.15000000000000002],
 [[None, array([ 0.2, -0.1]), None, array([-0.2,  0.1])], 0.09999999999999998],
 [[array([0.422, 0.1  ]), array([ 0.2, -0.1]), None, array([-0.2,  0.1])],
  0.15000000000000002],
 [[array([0.422, 0.1  ]),
   array([ 0.2, -0.1]),
   array([ 0.022, -0.1  ]),
   array([-0.2,  0.1])],
  0.10000000000000003],
 [[array([0.422, 0.1  ]), None, array([ 0.022, -0.1  ]), array([-0.2,  0.1])],
  0.1499999999999999],
 [[array([0.422, 0.1  ]), None, array([ 0.022, -0.1  ]), None],
  0.10000000000000009],
 [[array([0.422, 0.1  ]),
   array([ 0.422, -0.1  ]),
   array([ 0.022, -0.1  ]),
   None],
  0.1499999999999999],
 [[array([0.422, 0.1  ]),
   array([ 0.422, -0.1  ]),
   array([ 0.022, -0.1  ]),
   array([0.022, 0.1  ])],
  0.050000000000000044]]


#modified data structure

#

# print(polygons[2])
# for i in range(len(polygons)):
#     plt.figure()
#     plot_convex_shape(polygons[i][0])
#     plot_convex_quiver(a[i][0],edge[i],'r')
#     plt.grid()
#     # plt.xlim([-1,1])
#     # plt.ylim([-1,1])
# # for i in range(len(a)):
# plt.show()

"""
    changed a lot, will be improved 
"""
duration=[polygons[j][1] for j in range(len(polygons))]
# duration = duration[:2]
shrink_support,edge = reduce_convex(polygons)
# c_point = [sum(polygons[j][0])/4 for j in range(len(polygons))]
# duration_=duration
stp=[0,0]
dstp=[0.,0.]
ddstp=[0.,0.]
fp=[0.36,0.0]
p1=[0.5,1.25]
#
n_seg = 2 #number of segments
dim = 2 #number of coordinates
delta = 0.01


coeff =traj_opt(duration,stp,dstp,ddstp,fp,edge)

N =1000
n_seg = len(duration)
traj_p = np.zeros((dim,n_seg*N))
traj_zmp = np.zeros((dim,n_seg*N))
traj_dp = np.zeros((dim,n_seg*N))
traj_ddp = np.zeros((dim,n_seg*N))
tot_time = np.zeros(n_seg*N)





for i in range(n_seg):  
    time = np.linspace(0,duration[i],N)
    for j in range(N):
        tot_time[j+i*N] = tot_time[i*N-1]+time[j]
        for k in range(dim):
            traj_zmp[k,j+i*N] = zmp1(time[j])@coeff[i*6*dim+k*6:i*6*dim+(k+1)*6]
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
plt.plot(traj_p[0,:],traj_p[1,:],"k")
plt.plot(traj_zmp[0,:],traj_zmp[1,:],"--g")
for i in range(len(polygons)):
    plot_convex_shape(polygons[i][0],'b')
    plot_convex_shape(shrink_support[i][0],'r')
    # plot_convex_quiver(shrink_support[i][0],edge[i],'r')
plt.xlim(-0.4,0.4)
plt.ylim(-0.15,0.15)
plt.grid()
plt.show()
import numpy as np
import matplotlib.pyplot as plt
from quadprog import solve_qp
from numpy.linalg import matrix_rank as rank,inv
from numpy import hstack,vstack,array
from copy import deepcopy
from traj import traj_opt,traj_opt_regular,body_traj_show
import sys
sys.path.append("/home/holmes/Desktop/graduation/code/graduation_simulation_code")
import utils.plot_utils as plut
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
def plot_convex_shape(ax,vertices_, color='k'):
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
    ax.plot(x, y,color+"-.")
    ax.grid()



def plot_convex_quiver(ax,vertices,edge=None, color='k'):
    """Plot a convex shape and its normal vector"""
    num_vertices = len(vertices)
    x = [vertices[i][0] for i in range(num_vertices)]
    y = [vertices[i][1] for i in range(num_vertices)]
    x.append(vertices[0][0])  # Add the first vertex to close the shape
    y.append(vertices[0][1])  # Add the first vertex to close the shape
    midx = [(x[i]+x[i+1])/2.0 for i in range(len(x)-1)]
    midy = [(y[i]+y[i+1])/2.0 for i in range(len(y)-1)]
    ax.plot(x, y, color+"-.")
    ax.grid()
    if edge is not None:
        ax.quiver(midx,midy,edge[:,0],edge[:,1],color='k')


def reduce_convex(polygon_set,s=0.05,w=0.025):
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





#motion 2
# polygons=[[[None, array([ 0.301, -0.149]), None, array([-0.066,  0.124])], 0.04999999999999949],
# [[None, array([ 0.301, -0.149]), array([-0.033, -0.146]), array([-0.066,  0.124])], 0.1499999999999999],
# [[array([0.354, 0.121]), array([ 0.301, -0.149]), array([-0.033, -0.146]), array([-0.066,  0.124])], 0.10000000000000003],
# [[array([0.354, 0.121]), array([ 0.301, -0.149]), array([-0.033, -0.146]), None], 0.15000000000000002],
# [[array([0.354, 0.121]), None, array([-0.033, -0.146]), None], 0.09999999999999992],
# [[array([0.354, 0.121]), None, array([-0.033, -0.146]), array([-0.001,  0.107])], 0.15000000000000013],
# [[array([0.354, 0.121]), array([ 0.36 , -0.155]), array([-0.033, -0.146]), array([-0.001,  0.107])], 0.09999999999999998],
# [[array([0.354, 0.121]), array([ 0.36 , -0.155]), None, array([-0.001,  0.107])], 0.1499999999999999],
# [[None, array([ 0.36 , -0.155]), None, array([-0.001,  0.107])], 0.0500000000000006]]
# stp=array([ 0.067, -0.024])
# dstp=array([ 0.188, -0.345])
# ddstp=array([0., 0.])
# fp=array([ 0.177, -0.024])
#modified data structure
#motion 1

# from matplotlib.font_manager import fontManager as fm
# import matplotlib.pyplot as plt
# import matplotlib.ticker as ticker
# fm.addfont('/home/holmes/.local/share/fonts/Nsimsun.ttf')
# fm.addfont('/usr/share/fonts/truetype/msttcorefonts/Times_New_Roman.ttf')
# textwidth = 5.9#inch
# w = 0.45*textwidth
# font_size = 10.5
# # 创建一个图表
# plt.rcParams['figure.figsize']=(w,w)
# plt.rcParams['figure.dpi']=300
# plt.rcParams['savefig.dpi']=300
# plt.rcParams['xtick.labelsize']=font_size
# plt.rcParams['ytick.labelsize']=font_size
# plt.rcParams['font.family'] = ['sans-serif','NSimSun']
# plt.rcParams['font.sans-serif'] = ['Times New Roman']
# plt.rcParams['font.size']=font_size

polygons=[[[array([0.174, 0.131]), array([ 0.174, -0.131]), array([-0.187, -0.131]), array([-0.187,  0.131])], 0.05],
[[array([0.174, 0.131]), array([ 0.174, -0.131]), array([-0.187, -0.131]), None], 0.15000000000000002],
[[array([0.174, 0.131]), None, array([-0.187, -0.131]), None], 0.09999999999999998],
[[array([0.174, 0.131]), None, array([-0.187, -0.131]), array([-0.068,  0.131])], 0.15000000000000002],
[[array([0.174, 0.131]), array([ 0.293, -0.131]), array([-0.187, -0.131]), array([-0.068,  0.131])], 0.10000000000000003],
[[array([0.174, 0.131]), array([ 0.293, -0.131]), None, array([-0.068,  0.131])], 0.1499999999999999],
[[None, array([ 0.293, -0.131]), None, array([-0.068,  0.131])], 0.10000000000000009],
[[None, array([ 0.293, -0.131]), array([-0.068, -0.131]), array([-0.068,  0.131])], 0.1499999999999999],
[[array([0.293, 0.131]), array([ 0.293, -0.131]), array([-0.068, -0.131]), array([-0.068,  0.131])], 0.050000000000000044]]
stp=array([0., 0.])
dstp=array([0., 0.])
ddstp=array([0., 0.])
fp=array([0.1, 0. ])
#

# print(polygons[2])
# for i in range(len(polygons)):
#     fig,ax = plt.subplots()
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
duration_=duration
#
dim = 2 #number of coordinates
# for i in range(len(polygons)):
#     fig,ax = plt.subplots()
#     plot_convex_shape(ax,polygons[i][0],'b')
#     # plot_convex_shape(ax,shrink_support[i][0],'y')
#     plot_convex_quiver(ax,shrink_support[i][0],edge[i],'r')
# plt.show()
coeff_regular =traj_opt_regular(duration,stp,dstp,ddstp,fp)
coeff =traj_opt(duration,stp,dstp,ddstp,fp,edge,coeff_regular)
body_traj_show(duration,polygons,shrink_support,dim,coeff)
# N =100S
# n_seg = len(duration)
# traj_p = np.zeros((dim,n_seg*N))
# traj_zmp = np.zeros((dim,n_seg*N))
# traj_dp = np.zeros((dim,n_seg*N))
# traj_ddp = np.zeros((dim,n_seg*N))
# tot_time = np.zeros(n_seg*N)


# for i in range(n_seg):  
#     time = np.linspace(0,duration[i],N)
#     for j in range(N):
#         tot_time[j+i*N] = tot_time[i*N-1]+time[j]
#         for k in range(dim):
#             traj_zmp[k,j+i*N] = zmp1(time[j])@coeff[i*6*dim+k*6:i*6*dim+(k+1)*6]
#             traj_p[k,j+i*N] = nt(time[j])@coeff[i*6*dim+k*6:i*6*dim+(k+1)*6]
#             traj_dp[k,j+i*N] = dnt(time[j])@coeff[i*6*dim+k*6:i*6*dim+(k+1)*6]
#             traj_ddp[k,j+i*N] = ddnt(time[j])@coeff[i*6*dim+k*6:i*6*dim+(k+1)*6]
        

# LABEL={0:'x',1:'y',2:'z'}
# fig,ax = plt.subplots()
# for j in range(dim):
#     ax.plot(tot_time,traj_p[j,:],label='$pos_'+LABEL[j]+"$")
# ax.legend()
# ax.grid()

# fig,ax = plt.subplots()
# for j in range(dim):
#     ax.plot(tot_time,traj_dp[j,:],label='$vel_'+LABEL[j]+"$")
# ax.legend()
# ax.grid()

# fig,ax = plt.subplots()
# for j in range(dim):
#     ax.plot(tot_time,traj_ddp[j,:],label="$acc_"+LABEL[j]+"$")
# ax.legend()
# ax.grid()

# fig,ax = plt.subplots()
# ax.plot(traj_p[0,:],traj_p[1,:],"k")
# ax.plot(traj_p[0,0:-1:N],traj_p[1,0:-1:N],"ko")
# ax.plot(traj_zmp[0,:],traj_zmp[1,:],"--g")
# ax.plot(traj_zmp[0,0:-1:N],traj_zmp[1,0:-1:N],"r*")
# ax.grid()

# for i in range(len(polygons)):
#     fig,ax = plt.subplots()
#     ax.plot(traj_p[0,:],traj_p[1,:],"k")
#     ax.plot(traj_p[0,[i*N,(i+1)*N-1]],traj_p[1,[i*N,(i+1)*N-1]],"ko")
#     ax.plot(traj_zmp[0,:],traj_zmp[1,:],"--g")
#     ax.plot(traj_zmp[0,[i*N,(i+1)*N-1]],traj_zmp[1,[i*N,(i+1)*N-1]],"r*")
#     plot_convex_shape(ax,polygons[i][0],'b')
#     plot_convex_shape(ax,shrink_support[i][0],'y')
#     ax.grid()
#     # plot_convex_quiver(shrink_support[i][0],edge[i],'r')

plt.show()
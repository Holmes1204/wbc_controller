import numpy as np
from numpy import nan
from numpy.linalg import inv,pinv,norm,matrix_rank as rank
import matplotlib.pyplot as plt
import time
from math import sqrt

from scipy.spatial.transform import Rotation as R
from kinematics import A1_IK

init_pos= np.array([0.0,0.0,0.32,1.0,0.0,0.0,0.0,0.0,0.67, -1.3, -0.0, 0.67, -1.3, 0.0, 0.67, -1.3, -0.0, 0.67, -1.3])
# mujoco
import mujoco
import mujoco.viewer
mj_model = mujoco.MjModel.from_xml_path('/home/holmes/Data/python/wbc_controller/unitree_a1/scene.xml')
mj_data = mujoco.MjData(mj_model)
mj_data.qpos = init_pos
# pinocchio
import pinocchio as pin
import sys
sys.path.append("/home/holmes/Data/python/wbc_controller")
from utils.robot_wrapper import RobotWrapper
rmodel, rcollision_model, rvisual_model = pin.buildModelsFromUrdf("./a1_description/urdf/a1.urdf", ".",pin.JointModelFreeFlyer())
robot = RobotWrapper(rmodel, rcollision_model, rvisual_model) 
from local_planner_mj import local_planner,reduce_convex,print_each_support_polygon,print_all_support_polygon

def close_set(vertices):
    def calculate_angle(vertex, centroid):
      angle = np.arctan2(vertex[1] - centroid[1], vertex[0] - centroid[0])
      return angle
    centroid = np.mean(vertices, axis=0)
    angles = [calculate_angle(vertex, centroid) for vertex in vertices]
    sorted_indices = np.argsort(angles)
    sorted_vertices = vertices[sorted_indices]
    return sorted_vertices

def draw_polygon(scene,num_geom,vertices_,rbga_=[0.0, 0.0, 1.0, 0.8],size_=0.005):
    vertices = close_set(vertices_)
    for i in range(len(vertices)):
      mujoco.mjv_initGeom(
            scene.geoms[num_geom],
            type=mujoco.mjtGeom.mjGEOM_SPHERE,
            size=[size_, 0.0, 0.0],
            pos=vertices[i],
            mat=R.from_euler("zyx", np.array([0, 0, 0])).as_matrix().flatten(),
            rgba=rbga_
        )
    
        # Connect the current vertex to the next vertex
      if len(vertices) > 1:
        mujoco.mjv_connector(
              scene.geoms[num_geom],
              mujoco.mjtGeom.mjGEOM_CAPSULE,
              size_,
              vertices[i],
              vertices[(i + 1) % len(vertices)]
          )
      num_geom += 1
    return num_geom


with mujoco.viewer.launch_passive(mj_model, mj_data,show_left_ui=True,show_right_ui=False) as viewer:
  viewer.cam.azimuth= 135.0
  viewer.cam.distance= 2.0
  viewer.cam.elevation= -40.0
  viewer.cam.lookat= np.array([0., 0., 0.])
  # ------------ plan the body trajectory ----------------
  v_ref = np.array([0.0,0.3,0.])
  p_f = np.array([[ 0.174,  0.131,  0.002],
                  [ 0.174, -0.131,  0.002],
                  [-0.187,  0.131,  0.002],
                  [-0.187, -0.131,  0.002]])  
  foot_pos = np.array([0.174, 0.131, -0.3,0.174, -0.131, -0.3,-0.174, 0.131, -0.3,-0.174, -0.131, -0.3])
  base_pos = init_pos[:3]
  v_f = np.zeros((4,3))


  # local_plan.body_traj_show()
  stp = np.zeros(2)
  dstp =np.zeros(2)
  ddstp =np.zeros(2)
  final_pos=np.zeros(2)+v_ref[:2]*0.5
  # local planner
  local_plan  = local_planner(None,1)
  # local_plan.update(0,p_f,p_f,v_ref,v_ref)
  
  # support_polygon = local_plan.get_support_polygon(p_f[:,:2])
  # shrink_polygon,edge = reduce_convex(support_polygon)
  # local_plan.body_traj_plan(stp,dstp,ddstp,final_pos,edge,support_polygon,shrink_polygon)
  
  #存在bug
  # print_each_support_polygon(support_polygon,shrink_polygon,edge)
  # print_all_support_polygon(support_polygon,shrink_polygon)
  # plt.show()

  t = 0
  counter = 0
  dt = 0.001
  num_geom = 0
  traj = np.load("./traj.npz")['traj']
  traj = np.hstack((traj,0.32*np.ones((traj.shape[0],1))))

  for i_dt in range(traj.shape[0]-1):
    mujoco.mjv_initGeom(
          viewer.user_scn.geoms[num_geom],
          type=mujoco.mjtGeom.mjGEOM_SPHERE,
          size=[1, 0, 0],
          pos=np.array([0, 0, 0]),
          mat=R.from_euler("zyx",np.array([1,0,0])).as_matrix().flatten(),
          rgba=np.array([ 0.7, 0.7,0.7, 1.0]))
    mujoco.mjv_connector(viewer.user_scn.geoms[num_geom],
                    mujoco.mjtGeom.mjGEOM_CAPSULE, 0.0025,
                    traj[i_dt,:3],
                    traj[i_dt+1,:3])
    num_geom += 1
  obstacle_list = [(0.4, 0.4, 0.2),
                 (0.3,0.8,0.2),
                 (0.8,0.3,0.2)]
  for (x,y,l) in obstacle_list:
    h = 0.4
    mujoco.mjv_initGeom(
    viewer.user_scn.geoms[num_geom],
    type=mujoco.mjtGeom.mjGEOM_CYLINDER,
    size=[l/2, h, 0],
    pos=np.array([x, y, h]),
    mat=R.from_euler("zyx",np.array([1,0,0])).as_matrix().flatten(),
    rgba=np.array([ 0,0,0, 0.5]))
    num_geom += 1
  # for j in range(2):
  #   h = 0.5
  #   w = 0.1
  #   d = 0.1
  #   mujoco.mjv_initGeom(
  #   viewer.user_scn.geoms[num_geom],
  #   type=mujoco.mjtGeom.mjGEOM_CYLINDER,
  #   size=[w/2, d/2, h/2],
  #   pos=np.array([-1*j, 1*j, h/2]),
  #   mat=R.from_euler("zyx",np.array([1,0,0])).as_matrix().flatten(),
  #   rgba=0.5*np.array([ 0.1*j, 0.2*j,0.3*j, 0.7]))
  #   num_geom += 1
  # Create the polygon by connecting the vertices
  # num_geom = 0
  num_geom = draw_polygon(viewer.user_scn, num_geom, p_f)
  p_f  = p_f + np.array([0.2,0.0,0.0])
  num_geom = draw_polygon(viewer.user_scn, num_geom, p_f,rbga_=[0.0, 1.0, 0.0, 0.8])
  p_f  = 0.7*np.array([[ 0.174,  0.131,  0.002],
                  [ 0.174, -0.131,  0.002],
                  [-0.187, -0.131,  0.002]])
  num_geom = draw_polygon(viewer.user_scn, num_geom, p_f,rbga_=[1.0, 0.0, 0.0, 0.8])
  viewer.user_scn.ngeom = num_geom
  while viewer.is_running():
    # local_plan.update(t,p_f0,p_f,v_ref,v_ref)
    # traj_p,traj_dp,traj_ddp = local_plan.body_traj_update(t)
    # base_pos[:2] = traj_p

    # for j in range(4) :
    #     if not local_plan.in_contact(j):
    #         p_f[j],v,a= local_plan.swing_foot_traj(j,t)
    #     foot_pos[3*j:3*(j+1)] = p_f[j] - base_pos
    jp = A1_IK.computeIK(foot_pos)
    # mujoco display
    mj_data.qpos[:3] = base_pos
    mj_data.qpos[3:7] = np.array([1,0,0,0]) 
    mj_data.qpos[7:] = jp
    mujoco.mj_forward(mj_model, mj_data)
    viewer.sync()
    time.sleep(dt)
    counter += 1
    t += dt





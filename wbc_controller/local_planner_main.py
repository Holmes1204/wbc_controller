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
from traj_optimization.traj import body_traj_show

def close_set(vertices):
    def calculate_angle(vertex, centroid):
      angle = np.arctan2(vertex[1] - centroid[1], vertex[0] - centroid[0])
      return angle
    centroid = np.mean(vertices, axis=0)
    angles = [calculate_angle(vertex, centroid) for vertex in vertices]
    sorted_indices = np.argsort(angles)
    sorted_vertices = vertices[sorted_indices]
    return sorted_vertices

def draw_polygon(scene,num_geom,vertices,rbga_=[0.0, 0.0, 1.0, 0.8],size_=0.005):
    if len(vertices) > 1:
      vertices = close_set(vertices)
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

def draw_trajectory(scene,num_geom,traj,rbga_=[1.0, 1.0, 1.0, 0.8],size_=0.0025):
  for i_dt in range(traj.shape[0]-1):
    mujoco.mjv_initGeom(
          scene.geoms[num_geom],
          type=mujoco.mjtGeom.mjGEOM_SPHERE,
          size=[size_, 0, 0],
          pos=np.array([0, 0, 0]),
          mat=R.from_euler("zyx",np.array([1,0,0])).as_matrix().flatten(),
          rgba=rbga_)
    mujoco.mjv_connector(scene.geoms[num_geom],
                    mujoco.mjtGeom.mjGEOM_CAPSULE, size_,
                    traj[i_dt,:3],
                    traj[i_dt+1,:3])
    num_geom += 1
  return num_geom


def draw_obstacle(scene,num_geom,obstacle_list,rgba_=[0.0, 0.0, 0.0, 0.5],height=0.4):
  for (x,y,l) in obstacle_list:
    h = height
    mujoco.mjv_initGeom(
    viewer.user_scn.geoms[num_geom],
    type=mujoco.mjtGeom.mjGEOM_CYLINDER,
    size=[l/2, h, 0],
    pos=np.array([x, y, h]),
    mat=R.from_euler("zyx",np.array([1,0,0])).as_matrix().flatten(),
    rgba=rgba_)
    num_geom += 1
  return num_geom
  
with mujoco.viewer.launch_passive(mj_model, mj_data,show_left_ui=True,show_right_ui=False) as viewer:
  viewer.cam.azimuth= 135.0
  viewer.cam.distance= 2.0
  viewer.cam.elevation= -40.0
  viewer.cam.lookat= np.array([0., 0., 0.])
  viewer.user_scn.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = 0
  viewer.user_scn.flags[mujoco.mjtRndFlag.mjRND_SKYBOX] = 0
  viewer.user_scn.flags[mujoco.mjtRndFlag.mjRND_REFLECTION] = 0
  viewer.sync()
  # ------------ plan the body trajectory ----------------
  num_geom = 0
  # traj = np.load("./traj.npz")['traj']
  # traj = np.hstack((traj,0.32*np.ones((traj.shape[0],1))))

  # obstacle_list = [(0.4, 0.4, 0.2),
  #                (0.3,0.8,0.2),
  #                (0.8,0.3,0.2)]
  
  # num_geom = draw_trajectory(viewer.user_scn,num_geom,traj)
  # num_geom = draw_obstacle(viewer.user_scn,num_geom,obstacle_list)
  # ------------- -----------------------------
  p_f = np.array([[ 0.174,  0.131,  0.002],
                  [ 0.174, -0.131,  0.002],
                  [-0.187,  0.131,  0.002],
                  [-0.187, -0.131,  0.002]])
  foot_pos = np.array([0.174, 0.131, -0.3,0.174, -0.131, -0.3,-0.174, 0.131, -0.3,-0.174, -0.131, -0.3])
  base_pos = init_pos[:3]
  t = 0
  counter = 0
  dt = 0.001

  phi = 0.0
  v_ref = np.array([0.5,0.0,0.])
  v = np.array([0.2,0.0,0.0])
  w_cmd = 0.0

  # local_plan.body_traj_show()
  stp = np.zeros(2)
  dstp =np.zeros(2)
  ddstp =np.zeros(2)
  final_pos=stp+v_ref[:2]*0.25
  # local planner
  local_plan  = local_planner(None,0.25)
  local_plan.predict_future_foothold(base_pos,phi,v,v,w_cmd)
  local_plan.update(0.0,p_f,base_pos,phi,v,v,w_cmd)
  support_polygon = local_plan.get_support_polygon(p_f[:,:2])
  duration=[support_polygon[j][1] for j in range(len(support_polygon))]
  shrink_polygon,edge = reduce_convex(support_polygon)
  local_plan.body_traj_plan(0,duration,stp,dstp,ddstp,final_pos,edge)
  #存在bug
  # print_each_support_polygon(support_polygon,shrink_polygon,edge)
  # print_all_support_polygon(support_polygon,shrink_polygon)
  # body_traj_show(duration,support_polygon,shrink_polygon,2,local_plan.coeff)
  # plt.show()


  viewer.user_scn.ngeom = num_geom
  while viewer.is_running():
    if counter%250==0:
      num_geom = draw_polygon(viewer.user_scn, num_geom, [[base_pos[0],base_pos[1],0.0]],rbga_=[1.0, 0.0, 0.0, 0.8])
      stp = base_pos[:2]
      final_pos=stp+v_ref[:2]*0.25
      num_geom = draw_polygon(viewer.user_scn, num_geom, [[final_pos[0],final_pos[1],0.0]],rbga_=[0.0, 1.0, 0.0, 0.8])
      support_polygon = local_plan.get_support_polygon(p_f[:,:2])
      shrink_polygon,edge = reduce_convex(support_polygon)
      t0 = t
      duration=[support_polygon[j][1] for j in range(len(support_polygon))]
      local_plan.body_traj_plan(t,duration,stp,dstp,ddstp,final_pos,edge)
    local_plan.predict_future_foothold(base_pos,phi,v,v,w_cmd)
    local_plan.update(t,p_f,base_pos,phi,v,v,w_cmd)
    # after replan the time should be reset
    traj_p,traj_dp,traj_ddp = local_plan.body_traj_update(t)
    base_pos[:2] = traj_p

    for j in range(4) :
        if not local_plan.in_contact(j):
            p_f[j],v_f,a_f= local_plan.swing_foot_traj(j,t)
        foot_pos[3*j:3*(j+1)] = p_f[j] - base_pos
    foot_num_geom = num_geom
    foot_num_geom = draw_polygon(viewer.user_scn, foot_num_geom, local_plan.cur_foot,rbga_=[0.0, 0.0, 1.0, 0.8])
    foot_num_geom = draw_polygon(viewer.user_scn, foot_num_geom, local_plan.next_foot,rbga_=[0.0, 1.0, 0.0, 0.8])
    polygon_time = 0
    for i,j in shrink_polygon:
       if t-t0>polygon_time and t-t0<polygon_time+j:
          foot_pos_world = np.hstack((i,np.zeros((i.shape[0],1))))
          foot_num_geom = draw_polygon(viewer.user_scn,foot_num_geom,foot_pos_world,rbga_=[1.0, 0.0, 0.0, 0.8])
          break
       polygon_time += j
    viewer.user_scn.ngeom = foot_num_geom
    jp = A1_IK.computeIK(foot_pos)
    # mujoco display
    mj_data.qpos[:3] = base_pos
    mj_data.qpos[3:7] = np.array([1,0,0,0]) 
    mj_data.qpos[7:] = jp[[3,4,5,0,1,2,9,10,11,6,7,8]]
    mujoco.mj_forward(mj_model, mj_data)
    viewer.sync()
    time.sleep(dt)
    counter += 1
    t += dt
    time.sleep(0.01)





```[python]
for name in conf.contact_frames:
    simu.add_candidate_contact_point(name)
simu.add_contact_surface(conf.contact_surface_name, conf.contact_surface_pos, 
                         conf.contact_normal, conf.K, conf.B, conf.mu)
this is contact is really important for quadruped simulation, which have real contacts with world
```
/home/holmes/miniconda3/envs/dynamic_grasp/lib/python3.8/site-packages/cmeel.prefix/lib/python3.8/site-packages/example_robot_data/robots_loader.py
line180     free_flyer = False#True


for frame_idx in range(robot.model.nframes):
    frame_name = robot.model.frames[frame_idx].name
    print("Frame %d: %s" % (frame_idx, frame_name))
Frame 0: universe
Frame 1: root_joint
Frame 2: base
Frame 3: floating_base
Frame 4: trunk
Frame 5: FL_hip_joint
Frame 6: FL_hip
Frame 7: FL_hip_fixed
Frame 8: FL_thigh_shoulder
Frame 9: FL_thigh_joint
Frame 10: FL_thigh
Frame 11: FL_calf_joint
Frame 12: FL_calf
Frame 13: FL_foot_fixed
Frame 14: FL_foot
Frame 15: FR_hip_joint
Frame 16: FR_hip
Frame 17: FR_hip_fixed
Frame 18: FR_thigh_shoulder
Frame 19: FR_thigh_joint
Frame 20: FR_thigh
Frame 21: FR_calf_joint
Frame 22: FR_calf
Frame 23: FR_foot_fixed
Frame 24: FR_foot
Frame 25: RL_hip_joint
Frame 26: RL_hip
Frame 27: RL_hip_fixed
Frame 28: RL_thigh_shoulder
Frame 29: RL_thigh_joint
Frame 30: RL_thigh
Frame 31: RL_calf_joint
Frame 32: RL_calf
Frame 33: RL_foot_fixed
Frame 34: RL_foot
Frame 35: RR_hip_joint
Frame 36: RR_hip
Frame 37: RR_hip_fixed
Frame 38: RR_thigh_shoulder
Frame 39: RR_thigh_joint
Frame 40: RR_thigh
Frame 41: RR_calf_joint
Frame 42: RR_calf
Frame 43: RR_foot_fixed
Frame 44: RR_foot
Frame 45: imu_joint
Frame 46: imu_link

for robot 2 
Frame 0: universe
Frame 1: root_joint
Frame 2: base
Frame 3: floating_base
Frame 4: trunk
Frame 5: FL_hip_joint
Frame 6: FL_hip
Frame 7: FL_hip_fixed
Frame 8: FL_thigh_shoulder
Frame 9: FL_thigh_joint
Frame 10: FL_thigh
Frame 11: FL_calf_joint
Frame 12: FL_calf
Frame 13: FL_foot_fixed
Frame 14: FL_foot
Frame 15: FR_hip_joint
Frame 16: FR_hip
Frame 17: FR_hip_fixed
Frame 18: FR_thigh_shoulder
Frame 19: FR_thigh_joint
Frame 20: FR_thigh
Frame 21: FR_calf_joint
Frame 22: FR_calf
Frame 23: FR_foot_fixed
Frame 24: FR_foot
Frame 25: RL_hip_joint
Frame 26: RL_hip
Frame 27: RL_hip_fixed
Frame 28: RL_thigh_shoulder
Frame 29: RL_thigh_joint
Frame 30: RL_thigh
Frame 31: RL_calf_joint
Frame 32: RL_calf
Frame 33: RL_foot_fixed
Frame 34: RL_foot
Frame 35: RR_hip_joint
Frame 36: RR_hip
Frame 37: RR_hip_fixed
Frame 38: RR_thigh_shoulder
Frame 39: RR_thigh_joint
Frame 40: RR_thigh
Frame 41: RR_calf_joint
Frame 42: RR_calf
Frame 43: RR_foot_fixed
Frame 44: RR_foot
Frame 45: imu_joint
Frame 46: imu_link
Frame 47: jaco_mounting_block_to_j2s6s200_link_base
Frame 48: j2s6s200_link_base
Frame 49: j2s6s200_joint_1
Frame 50: j2s6s200_link_1
Frame 51: j2s6s200_joint_2
Frame 52: j2s6s200_link_2
Frame 53: j2s6s200_joint_3
Frame 54: j2s6s200_link_3
Frame 55: j2s6s200_joint_4
Frame 56: j2s6s200_link_4
Frame 57: j2s6s200_joint_5
Frame 58: j2s6s200_link_5
Frame 59: j2s6s200_joint_6
Frame 60: j2s6s200_link_6
Frame 61: j2s6s200_joint_end_effector
Frame 62: j2s6s200_end_effector
Frame 63: j2s6s200_joint_finger_1
Frame 64: j2s6s200_link_finger_1
Frame 65: j2s6s200_joint_finger_tip_1
Frame 66: j2s6s200_link_finger_tip_1
Frame 67: j2s6s200_joint_finger_2
Frame 68: j2s6s200_link_finger_2
Frame 69: j2s6s200_joint_finger_tip_2
Frame 70: j2s6s200_link_finger_tip_2


robot.framePlacement(conf.q0, frame_id, False).translation
look for position of the frame
robot.frameJacobian(q[:,ss], 4, False,pin.ReferenceFrame.LOCAL)
array([[1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
       [0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
       [0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
       [0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
       [0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
       [0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]])
       
robot.frameJacobian(q[:,ss], 4, False,pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)
array([[ 0.001, -1.   ,  0.   ,  0.   ,  0.   ,  0.   ,  0.   ,  0.   ,  0.   ,  0.   ,  0.   ,  0.   ,  0.   ,  0.   ,  0.   ,  0.   ,  0.   ,  0.   ],
       [ 1.   ,  0.001,  0.   ,  0.   ,  0.   ,  0.   ,  0.   ,  0.   ,  0.   ,  0.   ,  0.   ,  0.   ,  0.   ,  0.   ,  0.   ,  0.   ,  0.   ,  0.   ],
       [ 0.   ,  0.   ,  1.   ,  0.   ,  0.   ,  0.   ,  0.   ,  0.   ,  0.   ,  0.   ,  0.   ,  0.   ,  0.   ,  0.   ,  0.   ,  0.   ,  0.   ,  0.   ],
       [ 0.   ,  0.   ,  0.   ,  0.001, -1.   ,  0.   ,  0.   ,  0.   ,  0.   ,  0.   ,  0.   ,  0.   ,  0.   ,  0.   ,  0.   ,  0.   ,  0.   ,  0.   ],
       [ 0.   ,  0.   ,  0.   ,  1.   ,  0.001,  0.   ,  0.   ,  0.   ,  0.   ,  0.   ,  0.   ,  0.   ,  0.   ,  0.   ,  0.   ,  0.   ,  0.   ,  0.   ],
       [ 0.   ,  0.   ,  0.   ,  0.   ,  0.   ,  1.   ,  0.   ,  0.   ,  0.   ,  0.   ,  0.   ,  0.   ,  0.   ,  0.   ,  0.   ,  0.   ,  0.   ,  0.   ]])

here exists a rotation relationship, so the fist 6 rows representing the velocity and angular velocity expressed in the body frame.
M
array([[13.741,  0.   ,  0.   , -0.   , -0.317, -0.025,  0.   , -0.079, -0.023,  0.   , -0.079, -0.023,  0.   , -0.079, -0.023,  0.   , -0.079, -0.023],
       [ 0.   , 13.741,  0.   ,  0.317, -0.   , -0.118,  0.079,  0.   ,  0.   ,  0.079,  0.   ,  0.   ,  0.079,  0.   ,  0.   ,  0.079,  0.   ,  0.   ],
       [ 0.   ,  0.   , 13.741,  0.025,  0.118, -0.   ,  0.082,  0.029, -0.018, -0.082,  0.029, -0.018,  0.082,  0.029, -0.018, -0.082,  0.029, -0.018],
       [-0.   ,  0.317,  0.025,  0.177, -0.   , -0.013,  0.032,  0.003, -0.002,  0.032, -0.003,  0.002,  0.032,  0.003, -0.002,  0.032, -0.003,  0.002],
       [-0.317, -0.   ,  0.118, -0.   ,  0.39 , -0.   , -0.013,  0.02 ,  0.012,  0.013,  0.02 ,  0.012,  0.017,  0.03 ,  0.005, -0.017,  0.03 ,  0.005],
       [-0.025, -0.118, -0.   , -0.013, -0.   ,  0.384,  0.011,  0.01 ,  0.003,  0.011, -0.01 , -0.003, -0.017,  0.01 ,  0.003, -0.017, -0.01 , -0.003],
       [ 0.   ,  0.079,  0.082,  0.032, -0.013,  0.011,  0.028,  0.002, -0.002,  0.   ,  0.   ,  0.   ,  0.   ,  0.   ,  0.   ,  0.   ,  0.   ,  0.   ],
       [-0.079,  0.   ,  0.029,  0.003,  0.02 ,  0.01 ,  0.002,  0.025,  0.009,  0.   ,  0.   ,  0.   ,  0.   ,  0.   ,  0.   ,  0.   ,  0.   ,  0.   ],
       [-0.023,  0.   , -0.018, -0.002,  0.012,  0.003, -0.002,  0.009,  0.007,  0.   ,  0.   ,  0.   ,  0.   ,  0.   ,  0.   ,  0.   ,  0.   ,  0.   ],
       [ 0.   ,  0.079, -0.082,  0.032,  0.013,  0.011,  0.   ,  0.   ,  0.   ,  0.028, -0.002,  0.002,  0.   ,  0.   ,  0.   ,  0.   ,  0.   ,  0.   ],
       [-0.079,  0.   ,  0.029, -0.003,  0.02 , -0.01 ,  0.   ,  0.   ,  0.   , -0.002,  0.025,  0.009,  0.   ,  0.   ,  0.   ,  0.   ,  0.   ,  0.   ],
       [-0.023,  0.   , -0.018,  0.002,  0.012, -0.003,  0.   ,  0.   ,  0.   ,  0.002,  0.009,  0.007,  0.   ,  0.   ,  0.   ,  0.   ,  0.   ,  0.   ],
       [ 0.   ,  0.079,  0.082,  0.032,  0.017, -0.017,  0.   ,  0.   ,  0.   ,  0.   ,  0.   ,  0.   ,  0.028,  0.002, -0.002,  0.   ,  0.   ,  0.   ],
       [-0.079,  0.   ,  0.029,  0.003,  0.03 ,  0.01 ,  0.   ,  0.   ,  0.   ,  0.   ,  0.   ,  0.   ,  0.002,  0.025,  0.009,  0.   ,  0.   ,  0.   ],
       [-0.023,  0.   , -0.018, -0.002,  0.005,  0.003,  0.   ,  0.   ,  0.   ,  0.   ,  0.   ,  0.   , -0.002,  0.009,  0.007,  0.   ,  0.   ,  0.   ],
       [ 0.   ,  0.079, -0.082,  0.032, -0.017, -0.017,  0.   ,  0.   ,  0.   ,  0.   ,  0.   ,  0.   ,  0.   ,  0.   ,  0.   ,  0.028, -0.002,  0.002],
       [-0.079,  0.   ,  0.029, -0.003,  0.03 , -0.01 ,  0.   ,  0.   ,  0.   ,  0.   ,  0.   ,  0.   ,  0.   ,  0.   ,  0.   , -0.002,  0.025,  0.009],
       [-0.023,  0.   , -0.018,  0.002,  0.005, -0.003,  0.   ,  0.   ,  0.   ,  0.   ,  0.   ,  0.   ,  0.   ,  0.   ,  0.   ,  0.002,  0.009,  0.007]])
h
array([  0.   ,   0.   , 134.799,   0.241,   1.153,   0.   ,   0.801,   0.288,  -0.181,  -0.801,   0.288,  -0.181,   0.801,   0.288,  -0.181,  -0.801,   0.288,  -0.181])


array([[ 1.   ,  0.   ,  0.   ,  0.   ,  0.343,  0.01 ,  0.   ,  0.   ,  0.   ,  0.   ,  0.   ,  0.   ,  0.   ,  0.   ,  0.   ,  0.   ,  0.   ,  0.   , -0.01 , -0.068, -0.287, -0.   ,  0.   ,  0.   ],
       [ 0.   ,  1.   ,  0.   , -0.343,  0.   ,  0.703,  0.   ,  0.   ,  0.   ,  0.   ,  0.   ,  0.   ,  0.   ,  0.   ,  0.   ,  0.   ,  0.   ,  0.   , -0.703, -0.   , -0.   , -0.   , -0.264,  0.   ],
       [ 0.   ,  0.   ,  1.   , -0.01 , -0.703,  0.   ,  0.   ,  0.   ,  0.   ,  0.   ,  0.   ,  0.   ,  0.   ,  0.   ,  0.   ,  0.   ,  0.   ,  0.   ,  0.   ,  0.703, -0.498, -0.   ,  0.   ,  0.   ]])

only use the Jddq+dJdq = 0 as constraints will lead to bad performance when the leg slipped
how to simulate the contact
the detailed of the motion result
and simulation can be understood by the idea of smapling
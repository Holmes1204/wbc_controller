#!/usr/bin/env python

import rospy
import tf
from geometry_msgs.msg import TransformStamped
import pinocchio as pin
import sys
sys.path.append("/home/holmes/code/graduation_simulation_code")
import utils.plot_utils as plut
from utils.robot_wrapper import RobotWrapper
from utils.robot_simulator import RobotSimulator
import test_conf as conf
# Initialize the ROS node

rmodel, rcollision_model, rvisual_model = pin.buildModelsFromUrdf("../a1_description/urdf/a1_test.urdf", "/home/holmes/code/graduation_simulation_code",pin.JointModelFreeFlyer())
robot = RobotWrapper(rmodel, rcollision_model, rvisual_model)
simu = RobotSimulator(conf, robot)

rospy.init_node('transform_publisher')
# Create a tf.TransformBroadcaster object
tf_broadcaster = tf.TransformBroadcaster()
robot.computeAllTerms(simu.q,simu.v)
# Publish the transformation at a fixed rate
rate = rospy.Rate(10)  # 10 Hz
while not rospy.is_shutdown():
# Update the timestamp of the transformation
    time = rospy.Time.now()
    for i in range(len(robot.model.frames)):
        frame_name = robot.model.frames[i].name
        frame_id = robot.model.getFrameId(frame_name)
        # H = robot.framePlacement(simu.q,frame_id)
        H = robot.data.oMf[frame_id]
        p = H.translation
        quat = pin.Quaternion(H.rotation)
        transform = TransformStamped()
        transform.header.stamp = time
        transform.header.frame_id = 'map'  # parent frame
        transform.child_frame_id = frame_name  # child frame
        transform.transform.translation.x = p[0]
        transform.transform.translation.y = p[1]
        transform.transform.translation.z = p[2]
        # transform.transform.translation.x = 1.0  # translation along x-axis
        transform.transform.rotation.w = quat.w  # rotation as quaternion (identity)
        transform.transform.rotation.x = quat.x
        transform.transform.rotation.y = quat.y
        transform.transform.rotation.z = quat.z
        # print(p)
        # print(quat)
        # print(transform)
        # Publish the transformation
        tf_broadcaster.sendTransformMessage(transform)
    # Sleep to achieve the desired publishing rate
    rate.sleep()
print("stop!")
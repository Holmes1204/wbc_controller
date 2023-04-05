import numpy as np
import pinocchio as pin
from pinocchio.robot_wrapper import RobotWrapper
from utils.robot_wrapper import RobotWrapper
from utils.robot_simulator import RobotSimulator
import gepetto.corbaserver
import time
import os
import subprocess
model, collision_model, visual_model = pin.buildModelsFromUrdf("a1_description/urdf/a1.urdf", "/home/holmes/Documents/code/python/orc",pin.JointModelFreeFlyer())
# model, collision_model, visual_model = pin.buildModelsFromUrdf("kinova_description/robots/kinova.urdf", "/home/holmes/Documents/code/python/orc")
robot = RobotWrapper(model, collision_model, visual_model)
try:
    prompt = subprocess.getstatusoutput("ps aux |grep 'gepetto-gui'|grep -v 'grep'|wc -l")
    if int(prompt[1]) == 0:
        os.system('gepetto-gui &')
    time.sleep(1)
except:
    pass
gepetto.corbaserver.Client()

q0 = pin.neutral(model)
q0[-6:] =np.array([1.5707,2.618,-1.5707,-1.5707,3.1415, 0.])
robot.initViewer(loadModel=False)
robot.loadViewerModel()
robot.displayCollisions(False)
robot.displayVisuals(True)
robot.display(q0)
print(robot.model.nq)
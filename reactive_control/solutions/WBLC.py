#compute the necessary output recursively
#data structure 
#we note that q is the full configuration of the robot,so it concludes floating base and actuated joints
class Task:
    def __init__(self) -> None:
        self.J = 0
        self.dJdq =0 
        self.cmd =0
        self.fdb =0
        self.dcmd = 0
        self.dfdb = 0
        self.ddcmd =0

def Null_space(J):
    return J

def pinv(J):
    return J

def KinWBC(task_set,q0):
    delta = 0
    N = 1
    for task in task_set:
        J_pre = task.J*N
        delta += pinv(J_pre)*(task.cmd-task.fdb-task.J*delta)
        dq +=pinv(J_pre)*(task.dcmd-task.J*dq) 
        ddq+=pinv(J_pre)*(task.ddcmd-task.J*ddq-task.dJdq)
        N = N*Null_space(task.J*N)
    q = q0+delta
    return q,dq,ddq

def DynWBC():
    pass
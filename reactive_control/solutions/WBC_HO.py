import numpy as np
from quadprog import solve_qp
from numpy.linalg import pinv


class task:
    def __init__(self,A,b,D,f,priority) -> None:
        self.A = A #shape (m,n)
        self.b = b #shape (m,)
        if D is None:
            self.D = np.zeros((1,A.shape[1]))
            self.f = np.zeros(1)
        else:
            self.D = D #shape (k,n)
            self.f = f #shape (k,)
        self.priority = priority#may not sequential, but has have priority
        return
    


#this algorithm sequentail hierachical quadratic optimizaition
class WBC_HO:
    #list and dict is static variable
    alpha = 1e-9
    count =0
    def __init__(self,task_set):
        # self.n = task_set[0].A.shape[1]# many exist bugs
        self.prioritized_task = {}
        self.priority = []
        for i in task_set:
            if i.priority not in self.prioritized_task.keys():
                self.prioritized_task[i.priority] = []
            self.prioritized_task[i.priority].append(i)
        self.priority = sorted(list(self.prioritized_task.keys()))# 0 -> the highest priority
    
    def solve(self):    
        #suppose each level has at least one task
        Z_ = None
        x = None
        v = None
        Ds_= None
        fs_= None
        va_= None
        Al_= None#last time
        for i in self.priority:
            A_ = None
            b_ = None
            D_ = None
            f_ = None
            for j in self.prioritized_task[i]:
                if A_ is None:
                    A_ = j.A
                    b_ = j.b
                    D_ = j.D
                    f_ = j.f
                else:
                    A_ = np.vstack([A_,j.A])
                    b_ = np.hstack([b_,j.b])
                    D_ = np.vstack([D_,j.D])
                    f_ = np.hstack([f_,j.f])
            #here some thing need to change
            n = A_.shape[1]#the shape of x
            m = D_.shape[0]# the shape of v
            if x is None:
                x = np.zeros(n)
            if Z_ is None:
                Z_ = np.eye(n)
            else:
                Z_ = Z_@(np.eye(n) -pinv(Al_@Z_)@Al_@Z_)
            Al_ = A_

            H = np.block([[(A_@Z_).T@(A_@Z_)+self.alpha*np.eye(n),np.zeros((n,m))],
                          [np.zeros((m,n))    ,np.eye(m)      ]])
            c = np.hstack([(A_@Z_).T@(b_ -A_@x),np.zeros(m)])
            if Ds_ is None:
                Dhat = np.block([[D_@Z_,-np.eye(m)],
                            [np.zeros((m,n)),np.eye(m)]])
                fhat = np.block([f_ - D_@x,np.zeros(m)])
                Ds_ = D_
                fs_ = f_
            else:
                Dhat = np.block([[D_@Z_          ,-np.eye(m)],
                                [Ds_@Z_         ,np.zeros((Ds_.shape[0],m))],
                                [np.zeros((m,n)),np.eye(m)]])
                fhat = np.block([f_ - D_@x,fs_ - Ds_@x + va_,np.zeros(m)])
                Ds_ = np.vstack([Ds_,D_])
                fs_ = np.hstack([fs_,f_])

            xf, func, xu, iters, lagr, iact = solve_qp(H, c,-Dhat.T,-fhat)
            x = x + xf[:n]
            v = xf[n:]
            if va_ is None:
                va_ = v
            else:
                va_ = np.hstack([va_,v])
        # WBC_HO.count +=1
        # print(self.count,"solved!\n")
        return x

if __name__ == '__main__':

    pass
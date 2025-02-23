import numpy as np
from qpsolvers import solve_qp
from numpy.linalg import pinv


class task:
    def __init__(self,priority:int,eq:tuple,ie:tuple) -> None:
        self.priority = priority#may not sequential, but has have priority
        self.eq = eq
        self.ie = ie
    



class WBC_HO:
    alpha = 1e-9
    beta = 1e-6
    count = 0

    def __init__(self, task_set):
        self.prioritized_tasks = {}
        self.ordered_tasks = {}

        for it_ in sorted(task_set, key=lambda x: x.priority):
            if it_.priority not in self.prioritized_tasks:
                self.prioritized_tasks[it_.priority] = [it_]
            else:
                self.prioritized_tasks[it_.priority].append(it_)

        for key, value in self.prioritized_tasks.items():
            eq_A = []
            eq_b = []
            ie_D = []
            ie_f = []
            eq = None
            ie = None
            for i in value:
                if i.eq is not None:
                    eq_A.append(i.eq[0])
                    eq_b.append(i.eq[1])
                if i.ie is not None:
                    ie_D.append(i.ie[0])
                    ie_f.append(i.ie[1])

            if len(eq_A) > 0:
                eq = (np.vstack(eq_A), np.hstack(eq_b))
            if len(ie_D) > 0:
                ie = (np.vstack(ie_D), np.hstack(ie_f))

            self.n_p = eq[0].shape[1] if eq is not None else ie[0].shape[1]
            self.ordered_tasks[key] = task(key, eq, ie)
            
            


    def solve(self):    
        def NullProj(A):
            return (np.eye(A.shape[1])-pinv(A)@A)
        Al_ = None
        Ds_ = None
        fs_ = None
        va_ = None
        n = self.n_p
        for i in self.ordered_tasks.keys():
            # previous 
            if self.count == 0:
                Z_ = np.eye(self.n_p)
                sol = np.zeros(self.n_p)
            else:
                # if last time Al_ is None, then no need to update
                if Al_ is not None:
                    Z_ = Z_@NullProj(Al_@Z_)

            # current problem
            (A_,b_) = self.ordered_tasks[i].eq if self.ordered_tasks[i].eq is not None else (None,None)
            (D_,f_) = self.ordered_tasks[i].ie if self.ordered_tasks[i].ie is not None else (None,None)
            Al_ = A_
            if A_ is None and D_ is not None:
                A_ = np.zeros((self.n_p,self.n_p))
                b_ = np.zeros(self.n_p)


            # construct the problem by A, D
            if D_ is None:
                H = Z_.T@(A_.T@A_)@Z_+self.alpha*np.eye(n)
                c = Z_.T@A_.T@(A_@sol-b_)
                if Ds_ is None:
                    Dhat = None
                    fhat = None
                else:
                    Dhat = Ds_@Z_        
                    fhat = fs_ - Ds_@sol + va_
                    # print(f'in iter {self.count}  run this')

            else:
                m = D_.shape[0]# the size of constraints
                H = np.block([[Z_.T@(A_.T@A_)@Z_+self.alpha*np.eye(n),np.zeros((n,m))],
                            [np.zeros((m,n))    ,np.eye(m)      ]])
                c = np.hstack([Z_.T@A_.T@(A_@sol-b_),np.zeros(m)])

                if Ds_ is None:
                    Dhat = np.block([[D_@Z_,-np.eye(m)],
                                    [np.zeros((m,n)),-np.eye(m)]])
                    fhat = np.block([f_ - D_@sol,np.zeros(m)])
                else:
                    Dhat = np.block([[D_@Z_          ,-np.eye(m)],
                                    [Ds_@Z_         ,np.zeros((Ds_.shape[0],m))],
                                    [np.zeros((m,n)),-np.eye(m)]])
                    fhat = np.block([f_ - D_@sol,fs_ - Ds_@sol + va_,np.zeros(m)])
                

            # solve the problem
            # if Ds_ is not None:
            #     print(f'in iter {self.count} ie Ds_  fr',max(np.max(Ds_@sol-fs_),0))
            xf = solve_qp(H, c,Dhat,fhat,None,None,solver='quadprog')
            if xf is None:
                raise ValueError('QP solver failed')
            # if Dhat is not None:
            #     print(f'in iter {self.count} ie Dhat  ',max(np.max(Dhat@xf-fhat),0))
            z = xf[:n]
            sol = sol + Z_@z
            # post problem
            # print(f'in iter {self.count} eq ',np.linalg.norm(A_@sol-b_))
            # print(f'in iter {self.count} check eq0',np.linalg.norm(self.ordered_tasks[0].eq[0]@sol-self.ordered_tasks[0].eq[1]))
            # print(f'in iter {self.count} check eq2',np.linalg.norm(self.ordered_tasks[2].eq[0]@sol-self.ordered_tasks[2].eq[1]))
            # print(f'in iter {self.count} check eq3',np.linalg.norm(self.ordered_tasks[3].eq[0]@sol-self.ordered_tasks[3].eq[1]))
            # print(f'in iter {self.count} check eq4',np.linalg.norm(self.ordered_tasks[4].eq[0]@sol-self.ordered_tasks[4].eq[1]))
            
            # if D_ is not None:
            #     print(f'in iter {self.count} ie D_  ',max(np.max(D_@sol-f_),0))
            # if Ds_ is not None:
            #     print(f'in iter {self.count} ie Ds_ ',max(np.max(Ds_@sol-fs_),0))
            self.count +=1
            # print('-----------------------------------')
            if D_ is not None:
                if Ds_ is None:
                    Ds_ = D_
                    fs_ = f_
                    va_ = xf[n:]
                else:
                    Ds_ = np.vstack([Ds_,D_])
                    fs_ = np.hstack([fs_,f_])
                    va_ = np.hstack([va_,xf[n:]])

        return sol

if __name__ == '__main__':

    pass
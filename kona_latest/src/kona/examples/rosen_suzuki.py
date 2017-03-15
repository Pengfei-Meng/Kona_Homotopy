from kona.user import UserSolver, BaseVector
import numpy as np

class RosenSuzuki(UserSolver):
    """
    %  Rosen-Suzuki Problem
    %  min  x1^2 + x2^2 + 2*x3^2 + x4^2        - 5*x1 -5*x2 -21*x3 + 7*x4
    %  s.t. 8  - x1^2 -   x2^2 - x3^2 -   x4^2 -   x1 + x2 - x3 + x4 >= 0 
    %       10 - x1^2 - 2*x2^2 - x3^2 - 2*x4^2 +   x1           + x4 >= 0          
    %       5- 2*x1^2 -   x2^2 - x3^2          - 2*x1 + x2      + x4 >= 0            
    %  Initial Point x = [1,1,1,1];   
    %  Solution at   x = [0,1,2,-1]; 
    %                f = -44   
    %  Common wrong solution x = [2.5000, 2.5000, 5.2500, -3.5000]
    %                        f = -79.8750
    """

    def __init__(self, init_x=[1.0, 1.0, 1.0, 1.0]):
        super(RosenSuzuki, self).__init__(
            num_design=4, num_eq=0, num_ineq=3)

        self.init_x = init_x

        self.W_obj = 2*np.eye(4)
        self.W_obj[2,2] = 2

        self.W_con1 = np.diag(np.array([-2,-2,-2,-2]))
        self.W_con2 = np.diag(np.array([-2,-4,-2,-4]))
        self.W_con2 = np.diag(np.array([-4,-2,-2, 0]))

    def enforce_bounds(self, at_design):
        pass

    def eval_obj(self, at_design, at_state):

        x1 = at_design[0]
        x2 = at_design[1]
        x3 = at_design[2]
        x4 = at_design[3]

        square_term = pow(x1,2) + pow(x2,2) + 2*pow(x3,2) + pow(x4,2)
        linear_term = -5*x1 - 5*x2 - 21*x3 + 7*x4
                    
        return square_term + linear_term

    def eval_residual(self, at_design, at_state, store_here):
        store_here.data[:] = 0.

    def eval_ineq_cnstr(self, at_design, at_state):
        x1 = at_design[0]
        x2 = at_design[1]
        x3 = at_design[2]
        x4 = at_design[3]

        c1 = 8 - pow(x1,2) - pow(x2,2) - pow(x3,2) - pow(x4,2) - \
            x1 + x2 - x3 + x4
        c2 = 10 - pow(x1,2) - 2*pow(x2,2) - pow(x3,2) - 2*pow(x4,2) + x1 + x4
        c3 = 5 - 2*pow(x1,2) - pow(x2,2) - pow(x3,2) -2*x1 + x2 + x4

        return np.array([c1,c2,c3])


    def multiply_dCINdX(self, at_design, at_state, in_vec):        
        x1 = at_design[0]
        x2 = at_design[1]
        x3 = at_design[2]
        x4 = at_design[3]

        dCdX = np.array([ [-2*x1-1, -2*x2+1, -2*x3-1, -2*x4+1], 
                          [-2*x1+1, -4*x2, -2*x3, -4*x4+1],
                          [-4*x1-2, -2*x2+1, -2*x3, 1]])

        return dCdX.dot(in_vec)

    def multiply_dCINdX_T(self, at_design, at_state, in_vec):

        x1 = at_design[0]
        x2 = at_design[1]
        x3 = at_design[2]
        x4 = at_design[3]

        dCdX = np.array([ [-2*x1-1, -2*x2+1, -2*x3-1, -2*x4+1], 
                          [-2*x1+1, -4*x2, -2*x3, -4*x4+1],
                          [-4*x1-2, -2*x2+1, -2*x3, 1]])
        dCdX_T = dCdX.transpose()
        return dCdX_T.dot(in_vec)

    def eval_dFdX(self, at_design, at_state):

        x1 = at_design[0]
        x2 = at_design[1]
        x3 = at_design[2]
        x4 = at_design[3]
        
        return np.array([2*x1-5, 2*x2-5, 4*x3-21, 2*x4+7])


    def init_design(self):
        return self.init_x

    def init_slack(self):
        at_design = self.init_design()
        at_state = []
        at_slack = self.eval_ineq_cnstr(at_design, at_state)

        return (np.ones(self.num_ineq), 0)

    def current_solution(self, num_iter, curr_design, curr_state, curr_adj,
                         curr_eq, curr_ineq, curr_slack):

        self.curr_design = curr_design
        self.num_iter = num_iter

        dual = curr_ineq
        slack = curr_slack

        W = self.W_obj + dual[0]*self.W_con1 + dual[1]*self.W_con2 + dual[2]*self.W_con2  

        # print 'Current_dual: ', dual
        # print 'Assembled W: ', W
        return None

import numpy as np
from kona.user import UserSolver
import pdb
"""
    minimize    x1 + x2
    subject to  1 - x1^2 - x2^2   >=  0 
"""

class STA(UserSolver):

    def __init__(self, init_x = [1., 1.]):
        super(STA, self).__init__(
            num_design = 2,
            num_state = 0,
            num_eq = 0,
            num_ineq = 1)
        self.init_x = np.array(init_x)

    def eval_obj(self, at_design, at_state):
        x1 = at_design[0]
        x2 = at_design[1]
        return x1 + x2

    def eval_dFdX(self, at_design, at_state):
        x1 = at_design[0]
        x2 = at_design[1]

        der = np.zeros_like(at_design)
        der[0] = 1.0
        der[1] = 1.0
        return der

    def eval_ineq_cnstr(self, at_design, at_state):
        x1 = at_design[0]
        x2 = at_design[1]
    
        con_ineq = -x1**2 - x2**2 + 1.0 
        return con_ineq

    def multiply_dCINdX(self, at_design, at_state, in_vec):
        x1 = at_design[0]
        x2 = at_design[1]

        in1 = in_vec[0]
        in2 = in_vec[1]

        out_vec = -2*x1*in1 - 2*x2*in2  
 
        return out_vec

    def enforce_bounds(self, design_vec):
        pass

    def multiply_dCINdX_T(self, at_design, at_state, in_vec):
        x1 = at_design[0]
        x2 = at_design[1]

        out_vec = np.zeros(2)        
        out_vec[0] = -2*x1*in_vec  
        out_vec[1] = -2*x2*in_vec
        return out_vec

    def init_design(self):
        return self.init_x

    def init_slack(self):

        at_slack = 0.1*np.ones(self.num_ineq)
        # at_slack = self.eval_ineq_cnstr(self.init_x, [])
        return (at_slack, 0)

    def current_solution(self, num_iter, curr_design, curr_state, curr_adj,
                     curr_eq, curr_ineq, curr_slack):

        self.curr_design = curr_design
        self.curr_dual = curr_ineq
        self.curr_slack = curr_slack

        self.num_iter = num_iter

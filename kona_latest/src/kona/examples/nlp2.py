import numpy as np
from kona.user import UserSolver
import pdb


class NLP2(UserSolver):
    """
    f = exp(x(1))*(4*x(1)^2+2*x(2)^2+4*x(1)*x(2)+2*x(2)+1);
    s.t.   h = x(1)^2 + x(2) - 1 = 0
           g = x(1)*x(2) + 10 >= 0
    """

    def __init__(self, init_x = [-1.,1.]):

        super(NLP2, self).__init__(
            num_design=2, num_state=0, num_eq=1, num_ineq=1)
        self.init_x = init_x

    def eval_obj(self, at_design, at_state):
        x1 = at_design[0]
        x2 = at_design[1]
        result = np.exp(x1)*(4*x1**2+2*x2**2+4*x1*x2+2*x2+1);
        return result

    def eval_residual(self, at_design, at_state, store_here):
        store_here.data[:] = 0.

    def eval_dFdX(self, at_design, at_state):
        x1 = at_design[0]
        x2 = at_design[1]
        df1 = np.exp(x1)*(4*x1**2 + 2*x2**2 + 4*x1*x2 + 2*x2 + 1) + \
          np.exp(x1)*(8*x1 + 4*x2); 
        df2 = np.exp(x1)*(4*x2 + 4*x1 +2)
        return np.array([df1,df2])

    def eval_eq_cnstr(self, at_design, at_state):
        x1 = at_design[0]
        x2 = at_design[1]
        h = x1**2 + x2 - 1;
        return h

    def eval_ineq_cnstr(self, at_design, at_state):
        x1 = at_design[0]
        x2 = at_design[1]
        g = x1*x2 + 10;
        return g
        
    def multiply_dCEQdX(self, at_design, at_state, in_vec):
        x1 = at_design[0]
        x2 = at_design[1]
        dh = np.array([2*x1, 1])
        # pdb.set_trace()
        return dh.dot(in_vec)

    def multiply_dCEQdX_T(self, at_design, at_state, in_vec):
        x1 = at_design[0]
        x2 = at_design[1]
        dh = np.array([2*x1, 1])
        dhT = dh.transpose()
        # pdb.set_trace()
        return dhT.dot(in_vec)

    def multiply_dCINdX(self, at_design, at_state, in_vec):
        x1 = at_design[0]
        x2 = at_design[1]
        dg = np.array([-x2,-x1]);
        return dg.dot(in_vec)

    def multiply_dCINdX_T(self, at_design, at_state, in_vec):
        x1 = at_design[0]
        x2 = at_design[1]
        dg = np.array([-x2,-x1]);
        dgT = dg.transpose()
        return dgT.dot(in_vec)


    def init_design(self):
        return self.init_x

    def current_solution(self, num_iter, curr_design, curr_state, curr_adj,
                         curr_eq, curr_ineq, curr_slack):
        pass
import numpy as np
from kona.user import UserSolver
import pdb

class Constructed_SVDA(UserSolver):
    """
    f = 1/2 x^T Q x + g^T x 
    s.t.  g = Ax - b >= 0
    """

    def __init__(self, numdesign, numineq, init_x):

        super(Constructed_SVDA, self).__init__(
            num_design=numdesign, num_state=0, num_eq=0, num_ineq=numineq)
        self.init_x = init_x

        # numdesign <= numineq, A = SVD;    if numdesign > numineq, A_T = SVD; 
        assert numdesign <= numineq, \
            " numdesign <= numineq for the toy problem!"

        #--------- constructing Q and A ------------  
        A_sigma = np.diag(1./np.array(range(1, numdesign+1))**2 )  
        
        np.random.seed(0)  
        A_U = np.random.randint(10, size=(numineq, numdesign))   # (numineq, numdesign)
        A_V = np.random.randint(10, size=(numdesign, numdesign))    # (numdesign, numdesign)

        Q_U, r_U = np.linalg.qr(A_U)
        Q_V, r_V = np.linalg.qr(A_V)

        self.A = Q_U.dot(A_sigma).dot(Q_V)
        self.Q = np.diag(1./np.array(range(1, numdesign+1)))
        self.g = np.random.rand(numdesign)       # np.ones(numdesign)   
        self.b = np.random.rand(numineq)         # np.ones(numdesign)       

    def eval_obj(self, at_design, at_state):

        quadra = 0.5 * at_design.dot(self.Q.dot(at_design))
        result = quadra + self.g.dot(at_design)

        return result


    def eval_dFdX(self, at_design, at_state):

        result = self.Q.dot(at_design) + self.g
        return result

    def eval_ineq_cnstr(self, at_design, at_state):

        return self.A.dot(at_design) - self.b
        
    def multiply_dCINdX(self, at_design, at_state, in_vec):

        return self.A.dot(in_vec)

    def multiply_dCINdX_T(self, at_design, at_state, in_vec):

        return self.A.transpose().dot(in_vec)

    def init_design(self):
        return self.init_x

    def init_slack(self):
        return (np.ones(self.num_ineq), 0)

    def enforce_bounds(self, design_vec):
        pass

    def current_solution(self, num_iter, curr_design, curr_state, curr_adj,
                         curr_eq, curr_ineq, curr_slack):
        pass
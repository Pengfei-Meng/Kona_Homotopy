import numpy as np
from kona.user import UserSolver
import pdb
from scipy import optimize
import timeit

class NONCONVEX(UserSolver):
    """
    f = x^T D x
    s.t.  lb <= x <= ub
    """

    def __init__(self, numdesign, init_x, lb, ub, outdir):

        super(NONCONVEX, self).__init__(
            num_design=numdesign, num_state=0, num_eq=0, num_ineq=2*numdesign)

        assert len(init_x) == numdesign, \
            " initial x size is wrong"

        self.init_x = init_x

        #--------- constructing Q and A ------------  
        # np.random.seed(1) 
        self.D = np.sign(np.random.random(numdesign) - 0.5 )
        # self.D = np.array([-2])

        self.lb = lb
        self.ub = ub

        self.outdir = outdir
        
        # internal optimization bookkeeping
        self.iterations = 0
        self.duration = 0.
        self.totalTime = 0.
        self.startTime = timeit.default_timer()
        file = open(self.outdir+'/kona_timings.dat', 'w')
        file.write('# Constructed_SVDA iteration timing history\n')
        titles = '# {0:s}    {1:s}    {2:s}    {3:s}   \n'.format(
            'Iter', 'Time (s)', 'Total Time (s)', 'Objective' )
        file.write(titles)
        file.close()



    def eval_obj(self, at_design, at_state):

        quadra = 0.5*np.dot(at_design,  self.D*at_design)   
        # print 'quadra', quadra
        return quadra


    def eval_dFdX(self, at_design, at_state):

        result = self.D*at_design  
        return result

    def eval_ineq_cnstr(self, at_design, at_state):

        return np.concatenate([at_design - self.lb,  self.ub - at_design])
        
    def multiply_dCINdX(self, at_design, at_state, in_vec):

        return np.concatenate([in_vec, -in_vec])

    def multiply_dCINdX_T(self, at_design, at_state, in_vec):

        return in_vec[: self.num_design] - in_vec[self.num_design : ]

    def init_design(self):
        return self.init_x

    def init_slack(self):
        # at_slack = self.eval_ineq_cnstr(self.init_x, [])
        at_slack = 5*np.ones(self.num_ineq)
        return (at_slack, 0)

    def enforce_bounds(self, design_vec):
        design_vec[design_vec < self.lb] = self.lb
        design_vec[design_vec > self.ub] = self.ub

    def current_solution(self, num_iter, curr_design, curr_state, curr_adj,
                         curr_eq, curr_ineq, curr_slack):
        self.curr_design = curr_design
        self.curr_state = curr_state
        self.curr_dual = curr_ineq
        self.curr_slack = curr_slack


        # write timing to file
        # time the iteration
        self.endTime = timeit.default_timer()    
        self.duration = self.endTime - self.startTime
        self.totalTime += self.duration
        self.startTime = self.endTime

        objVal = self.eval_obj(curr_design, curr_state)

        timing = '  {0:3d}        {1:4.2f}        {2:4.2f}        {3:4.6g}    \n'.format(
            num_iter, self.duration, self.totalTime, objVal )
        file = open(self.outdir+'/kona_timings.dat', 'a')
        file.write(timing)
        file.close()


        # print 'Current X: ', self.curr_design


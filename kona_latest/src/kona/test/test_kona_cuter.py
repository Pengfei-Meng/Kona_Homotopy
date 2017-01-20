import numpy as np
import unittest

from kona import Optimizer 
from kona.algorithms import PredictorCorrectorCnstrINEQ, Verifier
from kona.examples import KONA_CUTER

"""
Problem Solution
BT1  -1.0
BT2  0.032568200
BT3  4.09301056 
BT4  3.28903771 
BT6  0.277044924
CAMEL6  -1.031628
"""

class InequalityTestCase(unittest.TestCase):


    def test_with_kona_cuter(self):
        prob_name = 'BT6'
        true_obj = 0.277044924     

        outdir = './temp/'
        solver = KONA_CUTER(prob_name)

        # Optimizer
        optns = {
            'info_file' : outdir + prob_name + '_kona_info.dat',
            'hist_file' : outdir + prob_name + '_kona_hist.dat',

            'max_iter' : 50,
            'opt_tol' : 1e-8,

            'homotopy' : {
                'lambda' : 0.0,
                'inner_tol' : 1e-2,
                'inner_maxiter' : 50,
                'nominal_dist' : 1.0,
                'nominal_angle' : 7.0*np.pi/180.,
            },

            'rsnk' : {
                'precond'       : None,
                # rsnk algorithm settings
                'dynamic_tol'   : False,
                'nu'            : 0.95,
                # reduced KKT matrix settings
                'product_fac'   : 0.001,
                'lambda'        : 0.0,
                'scale'         : 1.0,
                'grad_scale'    : 1.0,
                'feas_scale'    : 1.0,
                # krylov solver settings
                'krylov_file'   : 'kona_krylov.dat',
                'subspace_size' : 10,
                'check_res'     : True,
                'rel_tol'       : 1e-5,
            },
        }

        algorithm = PredictorCorrectorCnstrINEQ
        optimizer = Optimizer(solver, algorithm, optns)
        optimizer.solve()
        
        solution = solver.eval_obj(solver.curr_design, solver.curr_state)
        diff = abs(solution - true_obj)
        print diff
        self.assertTrue(diff < 1e-6)


if __name__ == "__main__":
    unittest.main()
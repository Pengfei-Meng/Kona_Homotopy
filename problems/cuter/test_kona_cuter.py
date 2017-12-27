import numpy as np
import unittest

from kona import Optimizer 
from kona.algorithms import PredictorCorrectorCnstrCond, Verifier
from kona_cuter import KONA_CUTER

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
            'max_iter' : 100,
            'opt_tol' : 1e-7,
            'feas_tol' : 1e-7, 
            'info_file' : outdir + prob_name + '_kona_info.dat',
            'hist_file' : outdir + prob_name + '_kona_hist.dat',

            'homotopy' : {
                'inner_tol' : 0.1,
                'inner_maxiter' : 2,
                'init_step' : 0.05,
                'nominal_dist' : 1.0,
                'nominal_angle' : 7.0*np.pi/180.,
                'max_factor' : 30.0,                  
                'min_factor' : 0.001,                   
                'dmu_max' : -0.0005,       
                'dmu_min' : -0.9,                     
            }, 

            'svd' : {
                'lanczos_size'    : 2, 
                'bfgs_max_stored' : 10, 
                'beta'         : 1.0, 
            }, 

            'rsnk' : {
                'precond'       : None,
                # krylov solver settings
                'krylov_file'   : outdir + prob_name + 'kona_krylov.dat',
                'subspace_size' : 20,
                'check_res'     : True,
                'rel_tol'       : 1e-2,
            },
        }

        algorithm = PredictorCorrectorCnstrCond
        optimizer = Optimizer(solver, algorithm, optns)
        optimizer.solve()
        
        solution = solver.eval_obj(solver.curr_design, solver.curr_state)
        diff = abs(solution - true_obj)
        print diff
        self.assertTrue(diff < 1e-6)


if __name__ == "__main__":
    unittest.main()
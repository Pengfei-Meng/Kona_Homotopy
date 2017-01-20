import numpy as np
import unittest

from kona import Optimizer
from kona.algorithms import PredictorCorrectorCnstrINEQ, Verifier
from kona.examples import RosenSuzuki, NLP2

class InequalityTestCase(unittest.TestCase):

    def test_with_rosen_suzuki(self):
        init_x = [1., 1., 1., 1.]

        outdir = './temp/'

        solver = RosenSuzuki(init_x=init_x)

        optns = {
            'max_iter' : 50,
            'opt_tol' : 1.e-5,
            'feas_tol' : 1.e-5,        
            'info_file' : outdir+'/kona_info.dat',
            'hist_file' : outdir+'/kona_hist.dat',

            'homotopy' : {
                'init_homotopy_parameter' : 1.0, 
                'inner_tol' : 0.01,
                'inner_maxiter' : 20,
                'nominal_dist' : 1.0,          
                'nominal_angle' : 5.0*np.pi/180., 
                'max_factor' : 5.0,                  
                'min_factor' : 0.5,               
                'dmu_max' : -0.001,       
                'dmu_min' : -0.9,        
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
                # FLECS solver settings
                'krylov_file'   : outdir+'/kona_krylov.dat',
                'subspace_size' : 20,                                     
                'check_res'     : True,
                'rel_tol'       : 1e-5,        
            },

        }

        algorithm = PredictorCorrectorCnstrINEQ
        optimizer = Optimizer(solver, algorithm, optns)
        optimizer.solve()
        expected = [0,1.,2.,-1.]
        diff = abs(solver.curr_design - expected)
        self.assertTrue(max(diff) < 1e-3)


if __name__ == "__main__":
    unittest.main()

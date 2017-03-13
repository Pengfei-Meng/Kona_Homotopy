import numpy as np
import unittest

import kona
from kona import Optimizer 
from kona.algorithms import PredictorCorrectorCnstrCond, Verifier
from kona.examples import Constructed_SVDA
import timeit

class InequalityTestCase(unittest.TestCase):

    def test_homotopy_svdA(self):

        prefix = './temp/'
        solver = Constructed_SVDA(500, 500, np.ones(500))

        # Optimizer
        optns = {
            'max_iter' : 100,
            'opt_tol' : 1e-6,
            'feas_tol' : 1e-6,        
            'info_file' : prefix+'/kona_info.dat',
            'hist_file' : prefix+'/kona_hist.dat',

            'homotopy' : {
                'init_homotopy_parameter' : 1.0, 
                'inner_tol' : 0.1,
                'inner_maxiter' : 5,
                'init_step' : 0.05,        
                'nominal_dist' : 1.0,            
                'nominal_angle' : 8.0*np.pi/180., 
                'max_factor' : 30.0,                  
                'min_factor' : 0.5,                   
                'dmu_max' : -0.0005,       
                'dmu_min' : -0.9,        
            },

            'rsnk' : {
                'precond'       : 'svd_pc',   # 'approx_adjoint',      # None,  
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
                'krylov_file'   : prefix+'/kona_krylov.dat',
                'subspace_size' : 20,                                    
                'check_res'     : False,
                'rel_tol'       : 1e-4,        
            },

            'verify' : {
                'primal_vec'     : True,
                'state_vec'      : False,
                'dual_vec_eq'    : False,
                'dual_vec_in'    : True,
                'gradients'      : True,
                'pde_jac'        : False,
                'cnstr_jac_eq'   : False,
                'cnstr_jac_in'   : True,
                'red_grad'       : True,
                'lin_solve'      : True,
                'out_file'       : prefix+'/kona_verify.dat',
            },
        }

        start = timeit.timeit()
        # algorithm = kona.algorithms.Verifier
        algorithm = kona.algorithms.PredictorCorrectorCnstrCond
        optimizer = kona.Optimizer(solver, algorithm, optns)
        optimizer.solve()

        solution = solver.eval_obj(solver.curr_design, solver.curr_state)

        end = timeit.timeit()
        print 'Homotopy completed in time: %f   obj: %f'%((end - start), solution)


        start = timeit.timeit()
        true_obj, true_x = solver.scipy_solution()
        end = timeit.timeit()
        print 'Scipy completed in time: %f   obj:%f'%((end - start), true_obj)

        # import pdb; pdb.set_trace()
        diff = max((solver.curr_design - true_x)/np.linalg.norm(true_x))
        # diff = np.linalg.norm(solver.curr_design.base.data - true_x, np.inf)

        print diff


        # self.assertTrue(diff < 1e-6)


if __name__ == "__main__":
    unittest.main()
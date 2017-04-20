import numpy as np
import unittest
import pdb
from kona import Optimizer
from kona.algorithms import PredictorCorrectorCnstrINEQ, Verifier
from kona.examples import STA

class InequalityTestCase(unittest.TestCase):

    def test_with_rosen_suzuki(self):
        init_x = [0.7, 0.8]

        outdir = './temp/'

        solver = STA(init_x=init_x)

        optns = {
            'max_iter' : 50,
            'opt_tol' : 1.e-6,
            'feas_tol' : 1.e-6,        
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
                'mu_correction' : 1.0,        
            },

            'rsnk' : {
                'precond'       : None,   #'approx_adjoint',  #'svd_pc',  #None, 
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
                'out_file'       : outdir+'/kona_verify.dat',
            },

        }

        algorithm = PredictorCorrectorCnstrINEQ
        # algorithm = Verifier
        optimizer = Optimizer(solver, algorithm, optns)
        optimizer.solve()
        expected = [-np.sqrt(2)/2, -np.sqrt(2)/2]
          

        diff = abs(solver.curr_design - expected)
        self.assertTrue(max(diff) < 1e-5)

        self.kona_obj = solver.eval_obj(solver.curr_design, [])
        self.kona_x = solver.curr_design

        print 'kona_obj ,', self.kona_obj 
        print 'kona_x  ', self.kona_x

if __name__ == "__main__":
    unittest.main()

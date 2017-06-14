import numpy as np
import unittest
import pprint
import os

import kona
from kona import Optimizer 
from kona.algorithms import PredictorCorrectorCnstrCond, Verifier
from kona.examples import Constructed_SVDA
from kona.examples import NONCONVEX
import time
import pdb
from pyoptsparse import Optimization, OPT
from kona.linalg.matrices.hessian import LimitedMemoryBFGS


class InequalityTestCase(unittest.TestCase):

    def setUp(self):

        self.outdir = './output5/nonconvex'
        if not os.path.isdir(self.outdir):
            os.mkdir(self.outdir)

        num_design = 100
        lb = -1
        ub = 1

        np.random.seed(1) 

        init_x = lb + (ub - lb) * np.random.random(num_design)  
        print 'init_x', init_x

        self.solver = NONCONVEX(num_design, init_x, lb, ub, self.outdir)


    def test_kona_optimize(self):

        # Optimizer
        optns = {
            'max_iter' : 300,
            'opt_tol' : 1e-7,
            'feas_tol' : 1e-7,        
            'info_file' : self.outdir+'/kona_info.dat',
            'hist_file' : self.outdir+'/kona_hist.dat',

            'quasi_newton' : {
                'type' : LimitedMemoryBFGS
            },

            'homotopy' : {
                'init_homotopy_parameter' : 1.0, 
                'inner_tol' : 0.1,
                'inner_maxiter' : 2,
                'init_step' : 3.0,                 
                'nominal_dist' : 10.0,               
                'nominal_angle' : 10.0*np.pi/180.,   
                'max_factor' : 20.0,                  
                'min_factor' : 0.001,                   
                'dmu_max' : -0.0005,       
                'dmu_min' : -0.9,      
                'mu_correction' : 1.0,  
                'use_frac_to_bound' : True,
                'mu_pc_on' : 1.0,      
            }, 

            'svd' : {
                'lanczos_size'    : 5, 
                'bfgs_max_stored' : 10, 
            }, 

            'rsnk' : {
                'precond'       : None, #'svd_pc',                  
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
                'krylov_file'   : self.outdir+'/kona_krylov.dat',
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
                'out_file'       : self.outdir+'/kona_verify.dat',
            },
        }

        # pprint.pprint(optns['homotopy'])
        with open(self.outdir+'/kona_optns.txt', 'w') as file:
            pprint.pprint(optns, file)
        # algorithm = kona.algorithms.Verifier
        algorithm = kona.algorithms.PredictorCorrectorCnstrCond
        optimizer = kona.Optimizer(self.solver, algorithm, optns)
        optimizer.solve()

        self.kona_obj = self.solver.eval_obj(self.solver.curr_design, self.solver.curr_state)
        self.kona_x = self.solver.curr_design
        print 'postive dual:   ', self.solver.curr_dual[self.solver.curr_dual > 1e-5]
        print 'negative slack: ', self.solver.curr_slack[self.solver.curr_slack < -1e-5]

        x_true = np.zeros(self.solver.num_design)
        x_true[self.solver.D < 0] = 1.0 

        x_kona = abs(self.kona_x)

        # self.solver.D
        diff = abs(np.rint(sum(x_kona - x_true)))
        print 'self.solver.D: ', self.solver.D
        print 'kona_solution: ', x_kona
        print 'x_true: ', x_true
        print 'number of wrong solutions: ',  diff

if __name__ == "__main__":
    unittest.main()
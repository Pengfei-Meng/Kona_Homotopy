import numpy as np
import unittest
import pprint
import os

import kona
from kona import Optimizer 
from kona.algorithms import PredictorCorrectorCnstrCond, Verifier
from nonconvex import NONCONVEX
import time
import pdb
from pyoptsparse import Optimization, OPT
from kona.linalg.matrices.hessian import LimitedMemoryBFGS


""" 1000 randomly generated initial points for 
    1000 randomly generated nonconvex quadratic problems
"""

# np.random.seed(1) 


outdir = './random'
if not os.path.isdir(outdir):
    os.mkdir(outdir)

lb = -2
ub = 2

num_design = 100
num_case = 10

# Optimizer
optns = {
    'max_iter' : 300,
    'opt_tol' : 1e-7,
    'feas_tol' : 1e-7,        
    'info_file' : outdir+'/kona_info.dat',
    'hist_file' : outdir+'/kona_hist.dat',

    'homotopy' : {
        'init_homotopy_parameter' : 1.0, 
        'inner_tol' : 0.1,                         # Hessian : num_design 
        'inner_maxiter' : 2,                       # -1.0 : 5     -1.0 : 100
        'init_step' : 0.05,                       # 0.5         0.05
        'nominal_dist' : 10,                     # 20           40
        'nominal_angle' : 10.0*np.pi/180.,          # 50           50
        'max_factor' : 50.0,                  
        'min_factor' : 0.001,                   
        'dmu_max' : -0.0005,        # -0.0005
        'dmu_min' : -0.9,      
        'mu_correction' : 1.0,  
        'use_frac_to_bound' : False,
        'mu_pc_on' : 1.0,      
    }, 

    'svd' : {
        'lanczos_size'    : 5, 
        'bfgs_max_stored' : 10, 
        'beta'         : 1.0, 
        'cmin'         : -1e-3,   # negative value, cut-off ineffective; 
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
        'krylov_file'   : outdir+'/kona_krylov.dat',
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
        'out_file'       : outdir+'/kona_verify.dat',
    },
}

with open(outdir+'/kona_optns.txt', 'w') as file:
    pprint.pprint(optns, file)


init_norms = np.zeros(num_case)
wrong_sols = np.zeros(num_case)

for i in xrange(num_case):

    init_x = lb + (ub - lb) * np.random.random(num_design)  
    init_norms[i] = np.linalg.norm(init_x)

    solver = NONCONVEX(num_design, init_x, -1, 1, outdir)

    algorithm = kona.algorithms.PredictorCorrectorCnstrCond
    optimizer = kona.Optimizer(solver, algorithm, optns)
    optimizer.solve()

    kona_obj = solver.eval_obj(solver.curr_design, solver.curr_state)
    kona_x = solver.curr_design

    x_true = np.zeros(num_design)
    x_true[solver.D < 0] = 1.0 

    x_kona = np.rint(abs(kona_x))

    diff = sum(abs(x_kona - x_true))
    wrong_sols[i] = diff

    # ------------- output ------------ #
    sep_str = '----------- Case %d ------------'%i
    D_str = 'solver.D: ' + str(solver.D)
    init_xs = 'init x: ' + str(init_x)
    # kona_sol = 'kona_solution: ' + str(kona_x)
    x_kona = 'kona_solution: ' + str(x_kona)
    true_sol = 'true solution: ' + str(x_true)
    err_diff = 'number of wrong solutions: ' + str(diff)

    with open(outdir+'/kona_optns.txt', 'a') as file:
        pprint.pprint(sep_str, file)
        pprint.pprint(D_str, file)
        pprint.pprint(init_xs, file)
        pprint.pprint(x_kona, file)
        pprint.pprint(true_sol, file)
        pprint.pprint(err_diff, file)


# ------------------ Make Plots ------------------

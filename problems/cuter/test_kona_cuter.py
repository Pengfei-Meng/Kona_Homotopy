import numpy as np
import unittest
import argparse, os, pdb
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

OK0  : not OK, caused by other issue but not Kona
OK1  : solved by Kona, but accuracy not checked,
       as the correct solution for this CUTEr problem is not listed 
       http://www.cuter.rl.ac.uk/Problems/mastsif.shtml
OK2  : solved by Kona, the objective function agrees with 
       the solution objective on the problem page

                                          nominal_dist
ACOPR14   Equ and Ineq            OK1
AUG2D     Equ and Ineq            OK1
AVGASA            Ineq            OK1
BDRY2                             OK0

BT1       Equ and Ineq            OK2       1 
BT2       Equ and Ineq            OK2       10
BT3       Equ and Ineq            OK2       10
"""

parser = argparse.ArgumentParser()
parser.add_argument("--name", help='Cuter problem name', type=str, default='BT1')
parser.add_argument("--v", help='Number of variables', type=int, default=0)
parser.add_argument("--output", help='Ouput Directory', type=str, default='./temp')
args = parser.parse_args()


prob_name = args.name
V = args.v
outdir = args.output + '/' + args.name
 
if not os.path.isdir(outdir):
    os.makedirs(outdir)

# true_obj = 0.277044924     

solver = KONA_CUTER(prob_name, V)

# Optimizer
optns = {
    'max_iter' : 100,
    'opt_tol' : 1e-7,
    'feas_tol' : 1e-7, 
    'info_file' : outdir + '/kona_info.dat',
    'hist_file' : outdir + '/kona_hist.dat',

    'homotopy' : {
        'inner_tol' : 0.1,
        'inner_maxiter' : 2,
        'init_step' : 0.2,
        'nominal_dist' : 10.0,
        'nominal_angle' : 5.0*np.pi/180.,
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
        'krylov_file'   : outdir + '/kona_krylov.dat',
        'subspace_size' : 20,
        'check_res'     : True,
        'rel_tol'       : 1e-2,
    },
}

algorithm = PredictorCorrectorCnstrCond
optimizer = Optimizer(solver, algorithm, optns)
optimizer.solve()

solution = solver.eval_obj(solver.curr_design, solver.curr_state)
# diff = abs(solution - true_obj)
print 'solution : ', solution
# self.assertTrue(diff < 1e-6)

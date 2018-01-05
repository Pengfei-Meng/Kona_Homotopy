import numpy as np
import unittest, timeit
import pprint
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
ACOPR14   Equ and Ineq            OK1                      38 0 28 144
AUG2D     Equ and Ineq            OK1                     220 0 100 200
AVGASA            Ineq            OK1
BDRY2                             OK0

BT1       Equ and Ineq            OK2       1 
BT2       Equ and Ineq            OK2       5              3 0 1 2
BT3       Equ and Ineq            OK2       10             5 0 3 6
"""

parser = argparse.ArgumentParser()
parser.add_argument("--name", help='Cuter problem name', type=str, default='BT1')
parser.add_argument("--v", help='Number of variables', type=int, default=0)
parser.add_argument("--output", help='Ouput Directory', type=str, default='./temp')
parser.add_argument("--precond", help='Preconditioner', type=str, default='Eye')
args = parser.parse_args()

prob_name = args.name
pc_name = args.precond    
V = args.v

solver = KONA_CUTER(prob_name, V)


if pc_name == 'Eye': 
    outdir = args.output + '/' + args.name 
    pc = None 
else: 
    outdir = args.output + '/' + args.name  + '_PC'

    if solver.num_eq == 0 and solver.num_ineq == 0:
        print 'Unconstrained Case, Not Considered In the Algorithm, Try another problem..'
        exit()

    elif solver.num_ineq == 0:
        print 'num_ineq = 0, equality only case, not considered yet'
        exit() 

    elif solver.num_eq == 0:
        print 'num_eq = 0, Inequality only case'
        pc = 'svd_pc_cmu'      

    else: 
        print 'Contains both equality and inequality constraints '
        pc = 'svd_pc5'

if not os.path.isdir(outdir):
    os.makedirs(outdir)

# true_obj = 0.277044924     

# Optimizer
optns = {
    'max_iter' : 300,
    'opt_tol' : 1e-7,
    'feas_tol' : 1e-7, 
    'info_file' : outdir + '/kona_info.dat',
    'hist_file' : outdir + '/kona_hist.dat',

    'homotopy' : {
        'inner_tol' : 0.1,
        'inner_maxiter' : 2,
        'init_step' : 0.05,
        'nominal_dist' : 10.0,
        'nominal_angle' : 10.0*np.pi/180.,
        'max_factor' : 30.0,                  
        'min_factor' : 0.001,                   
        'dmu_max' : -0.0005,       
        'dmu_min' : -0.9,                     
    }, 

    'svd' : {
        'lanczos_size'    : 2, 
        'bfgs_max_stored' : 10, 
        'beta'         : 1.0, 
        'mu_min'       : 1e-2,
    }, 

    'rsnk' : {
        'precond'       : pc,     
        # krylov solver settings
        'krylov_file'   : outdir + '/kona_krylov.dat',
        'subspace_size' : 20,
        'check_res'     : True,
        'rel_tol'       : 1e-2,
    },
}

startTime = timeit.default_timer()
        
algorithm = PredictorCorrectorCnstrCond
optimizer = Optimizer(solver, algorithm, optns)
optimizer.solve()

# ---------- Book-keeping Options and Results -----------
duration = timeit.default_timer() - startTime
solution = solver.eval_obj(solver.curr_design, solver.curr_state)


f_optns = outdir + '/kona_optns.dat'
print 'solution : ', solution

kona_obj = 'Kona objective value at its solution, ' + str(solution)
kona_time = 'Kona runtime, ' + str(duration)

with open(f_optns, 'a') as file:
    pprint.pprint(optns, file)
    pprint.pprint(kona_obj, file)
    pprint.pprint(kona_time, file)

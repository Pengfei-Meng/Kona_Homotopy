import numpy as np
import unittest, timeit
import pprint
import argparse, os, pdb
from kona import Optimizer 
from kona.algorithms import PredictorCorrectorCnstrCond, Verifier
from pyoptsparse import Optimization, OPT
from kona_cuter import KONA_CUTER
import cutermgr

name_list = ['AUG2D',    'AUG2DQP', 'AUG2DC',   'AUG3D',  'AUG3DC',   'AVGASB', 'ALLINQP', 'AVGASA',          # 8
    'BLOCKQP2',  'BLOCKQP3', 'BLOCKQP4', 'BLOCKQP5', 'BDRY2', 'BIGGSC4',
    'CVXQP1', 'CVXQP2', 'NCVXQP9', 'NCVXQP8', 'NCVXQP1', 'NCVXQP7', 'NCVXQP5', 'NCVXQP6', 
    'DEGENQPC', 'DEGTRIDL', 'DTOC3', 'DEGENQP', 'FERRISDC', 
    'GOULDQP1', 'GENHS28', 'GMNCASE4', 'GMNCASE1',
    'HS268','HS76I','HS51','HS53','HS52','HS44','HS76','HS35I','HS21','HS35','HS118','HS35MOD','HATFLDH','HS44NEW', 
    'LOTSCHD', 'MOSARQP2', 'MOSARQP1', 'NASH',
    'PORTSQP', 'POWELL20', 'QPBAND',
    'RDW2D52F','RDW2D52B', 'RDW2D52U', 'RDW2D51F',  'RDW2D51U', 
    'STNQP1', 'STNQP2', 'STCQP1',  'STCQP2', 'SOSQP1', 'SOSQP2', 'S268',
    'TWOD',  'TAME', 'YAO', 'ZECEVIC2',
]


def kona_optimize(args, pc_name, prob_name):
    global f_optns, solver

    # k = args.k
    V1 = args.V1
    V2 = args.V2
    V3 = args.V3
    # pc_name = args.precond 
    
    solver = KONA_CUTER(prob_name, V1, V2, V3)

    print 'num_design, num_state, num_eq, num_ineq', \
        solver.num_design, solver.num_state, solver.num_eq, solver.num_ineq

    if any(x>1000 for x in [solver.num_design, solver.num_eq, solver.num_ineq]):
        print 'Size Too Large, Lenovo Laptop cannot handle it! Exiting...'
        exit()

    outdir = args.output + '/' + prob_name 

    if pc_name == 'Eye': 
        
        pc = None 
        f_info = outdir+'/kona_eye_info.dat'
        f_hist = outdir+'/kona_eye_hist.dat'
        f_krylov = outdir+'/kona_eye_krylov.dat'
        

    else: 
        f_info = outdir+'/kona_pc_info.dat'
        f_hist = outdir+'/kona_pc_hist.dat'
        f_krylov = outdir+'/kona_pc_krylov.dat'

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

    optns = {
        'max_iter' : 300,
        'opt_tol' : 1e-6,
        'feas_tol' : 1e-6, 
        'info_file' : f_info,
        'hist_file' : f_hist,

        'homotopy' : {
            'inner_tol' : 0.1,
            'inner_maxiter' : 2,
            'init_step' : args.iniST,
            'nominal_dist' : args.nomDist,
            'nominal_angle' : args.nomAngle*np.pi/180.,
            'max_factor' : 30.0,                  
            'min_factor' : 0.001,                   
            'dmu_max' : -0.0005,              
            'dmu_min' : -0.9,                     
        }, 

        'svd' : {
            'lanczos_size'    : 30,  # max(int(solver.num_design*0.2), solver.num_design-1), 
            'bfgs_max_stored' : 10, 
            'beta'         : 1.0, 
            'mu_min'       : 1e-4,
        }, 

        'rsnk' : {
            'precond'       : pc,     
            # krylov solver settings
            'krylov_file'   : f_krylov,
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

    print 'Kona Objective : ', solution
    print 'Time Elapse: ', duration

    cuter_nm = ' n, m : ' + str(solver.num_design) + ', ' + str(solver.info['m'])
    cuter_dimension = 'num_design, num_state, num_eq, num_ineq : ' + \
            str(solver.num_design) + '  '  + \
            str(solver.num_state) + '  '  + \
            str(solver.num_eq)  + '  '  + \
            str(solver.num_ineq) \

    kona_obj = 'Kona final objective value with PC ' + pc_name + ' : ' + str(solution)
    kona_time = 'Kona runtime, ' + str(duration)

    with open(f_optns, 'a') as file:
        if pc_name == 'Eye':
            pprint.pprint(optns, file)
            pprint.pprint(cuter_dimension, file)
            pprint.pprint(cuter_nm, file)

        pprint.pprint('===========================', file)
        pprint.pprint(kona_obj, file)
        pprint.pprint(kona_time, file)
        pprint.pprint('===========================', file)

def objfunc(xdict):
    global solver

    x = xdict['xvars']
    funcs = {}
    funcs['obj'] = solver.prob.obj(x)
    funcs['con'] = solver.prob.cons(x)
    fail = False

    return funcs, fail

def sens(xdict, funcs):
    global solver

    x = xdict['xvars']
    f, dfdx = solver.prob.obj(x, True)
    c, dcdx = solver.prob.cons(x, True)

    funcsSens = {}
    funcsSens['obj'] = {'xvars': dfdx }
    funcsSens['con'] = {'xvars': dcdx }
    fail = False

    return funcsSens, fail


def snopt_optimize(args, optName, optOptions): 
    global solver, f_optns

    prob_name = solver.prob_name    # name_list[args.k] 
    outdir = args.output + '/' + prob_name 

    info = solver.prob.getinfo()

    num_design = info['n']
    num_con = info['m']
    init_x = info['x']

    bl = info['bl']
    bu = info['bu']
    cl = info['cl']
    cu = info['cu']

    # ------------------------------------------------
    startTime = timeit.default_timer()

    # Optimization Object
    optProb = Optimization('snopt CUTEr', objfunc)

    optProb.addVarGroup('xvars', num_design, value=init_x)
    optProb.addConGroup('con', num_con, lower = cl, upper = cu)
    optProb.addObj('obj')

    opt = OPT(optName, options=optOptions)

    sol = opt(optProb, sens=sens)  

    sn_time = timeit.default_timer() - startTime

    # Check Solution
    pyopt_obj = sol.objectives['obj'].value
    pyopt_x = np.array(map(lambda x:  sol.variables['xvars'][x].value, xrange(num_design)))
    
    print 'SNOPT Objective : ', pyopt_obj
    print 'SNOPT Time : ' , sn_time

    snopt_obj = 'SNOPT objective value at its solution, ' + str(pyopt_obj)
    snopt_time = 'SNOPT runtime : ' + str(sn_time)

    with open(f_optns, 'a') as file:
        pprint.pprint('========== SNOPT ===========', file)
        pprint.pprint(snopt_obj, file)
        pprint.pprint(snopt_time, file)
        

if __name__ == '__main__':

    global f_optns

    parser = argparse.ArgumentParser()
    parser.add_argument("--iniST", help='init step', type=float, default=0.05)
    parser.add_argument("--nomDist", help='nominal dist', type=float, default=1.0)
    parser.add_argument("--nomAngle", help='nominal angle', type=float, default=5.0)
    parser.add_argument("--V1", help='1st Parameter', type=int, default=0)
    parser.add_argument("--V2", help='2nd Parameter', type=int, default=0)
    parser.add_argument("--V3", help='3rd Parameter', type=float, default=0)
    parser.add_argument("--output", help='Ouput Directory', type=str, default='./KonaSnopt')
    args = parser.parse_args()

    out_put = args.output

    for k in range(2): 
        prob_name = name_list[k]       

        # -------------- Writing to f_optns --------------
        outdir = out_put + '/' + prob_name 

        if not os.path.isdir(outdir):
            os.makedirs(outdir)

        f_optns = outdir+'/output.dat'

        cutermgr.updateClassifications() 
        feature = cutermgr.problemProperties(prob_name)

        with open(f_optns, 'w') as file:
            pprint.pprint(prob_name, file)  
            pprint.pprint(feature, file)        

        kona_optimize(args, 'Eye', prob_name)
        kona_optimize(args, 'pc', prob_name)   
        # sequence cannot be changed!! 


        optOptions = {'Print file': out_put + '/' + prob_name  + '/SNOPT_print.out',
                      'Summary file': out_put + '/' + prob_name  + '/SNOPT_summary.out',
                      'Problem Type':'Minimize',
                      }

        snopt_optimize(args, 'snopt', optOptions)
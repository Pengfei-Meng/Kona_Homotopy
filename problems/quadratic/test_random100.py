import numpy as np
import unittest
import pprint
import os, pickle
import matplotlib.pyplot as plt
import kona
from kona import Optimizer 
from kona.algorithms import PredictorCorrectorCnstrCond, Verifier
from construct_svdA import Constructed_SVDA
import time, timeit
import pdb
from pyoptsparse import Optimization, OPT
from kona.linalg.matrices.hessian import LimitedMemoryBFGS
import argparse

""" 100 randomly generated initial points for 
    100 randomly generated quadratic problems
    Remember to comment out all the np.random.seed(0) in Constructed_SVDA problem
    For each num_design = 100, 200, 300, 400, 500
 1) run the following command, --num_design 100 - 500    
    python test_random100.py --output './random_100' --task 'opt' --num_case 100 --num_design 100
 2) python test_random100.py --output './random_100' --task 'post' --num_case 100

"""


parser = argparse.ArgumentParser()
parser.add_argument("--output", help='Output directory', type=str, default='./random')
parser.add_argument("--task", help='what to do', choices=['opt','post'], default='post')
parser.add_argument("--num_design", type=int, default=100)
parser.add_argument("--num_case", type=int, default=10)
parser.add_argument("--color", type=int, default=1)
args = parser.parse_args()

num_design = args.num_design
num_ineq = num_design
num_case = args.num_case
pic_color = args.color

outdir = args.output    
if not os.path.isdir(outdir):
    os.mkdir(outdir)

print args.task


if args.task == 'opt': 

    optOptions = {'Print file': outdir + '/SNOPT_print.out',
                  'Summary file': outdir + '/SNOPT_summary.out',
                  'Problem Type':'Minimize',
                  }
    storeHistory=False

    if num_design==100:
        init_s = 40
    if num_design==200:
        init_s = 80
    if num_design==300:
        init_s = 120  
    if num_design==400:
        init_s = 160  
    if num_design==500:
        init_s = 200  

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
            'init_step' : init_s,                       # 0.5         0.05
            'nominal_dist' : 10,                     # 20           40
            'nominal_angle' : 20.0*np.pi/180.,          # 50           50
            'max_factor' : 50.0,                  
            'min_factor' : 0.001,                   
            'dmu_max' : -0.0005,        # -0.0005
            'dmu_min' : -0.9,      
            'mu_correction' : 1.0,  
            'use_frac_to_bound' : False,
            'mu_pc_on' : 1.0,      
        }, 

        'svd' : {
            'lanczos_size'    : 2, 
            'bfgs_max_stored' : 10, 
            'beta'         : 1.0, 
        }, 

        'rsnk' : {
            'precond'       : 'svd_pc_cmu',                  
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
            'rel_tol'       : 1e-2,        
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

    kona_times = np.zeros(num_case)
    snopt_times = np.zeros(num_case)

    for i in xrange(num_case):
        print 'Starting case %d'%i

        init_x = np.random.random(num_design)  
        solver = Constructed_SVDA(num_design, num_ineq, init_x, outdir, 'kona_timings.dat')

        start_kona_i = timeit.default_timer()
        # --------- Kona Optimize ----------
        algorithm = kona.algorithms.PredictorCorrectorCnstrCond
        optimizer = kona.Optimizer(solver, algorithm, optns)
        optimizer.solve()

        kona_obj = solver.eval_obj(solver.curr_design, solver.curr_state)
        kona_x = solver.curr_design
        #---------------------------------
        end_kona_i = timeit.default_timer()

        kona_times[i] = end_kona_i - start_kona_i


        start_snopt_i = timeit.default_timer()
        # --------- Entering SNOPT Optimization ---------

        def objfunc(xdict):

            x = xdict['xvars']
            funcs = {}
            funcs['obj'] = 0.5 * np.dot(x.T, np.dot(solver.Q, x)) + np.dot(solver.g, x) 

            conval = np.dot(solver.A,x) - solver.b
            funcs['con'] = conval
            fail = False

            return funcs, fail

        def sens(xdict, funcs):
            x = xdict['xvars']
            funcsSens = {}
            funcsSens['obj'] = {'xvars': np.dot(x.T, solver.Q) + solver.g }
            funcsSens['con'] = {'xvars': solver.A }
            fail = False

            return funcsSens, fail


        # --------- SNOPT Optimize ---------
        # Optimization Object
        optProb = Optimization('quadra', objfunc)

        # Design Variables
        optProb.addVarGroup('xvars', num_design, value=init_x)

        # Constraints
        lower = np.zeros(num_ineq)
        upper = [None]*num_ineq
        optProb.addConGroup('con', num_ineq, lower = lower, upper = upper)

        # Objective
        optProb.addObj('obj')

        # Optimizer
        opt = OPT('snopt', options=optOptions)

        # Solution
        if storeHistory:
            histFileName = '%s.hst' % (optName.lower())
        else:
            histFileName = None

        sol = opt(optProb, sens=sens, storeHistory=histFileName)  
        
        # Check Solution
        pyopt_obj = sol.objectives['obj'].value
        pyopt_x = np.array(map(lambda x:  sol.variables['xvars'][x].value, xrange(num_design)))

        # ----------------------------------
        end_snopt_i = timeit.default_timer()
        snopt_times[i] = end_snopt_i - start_snopt_i

        # --------- Comparing difference Kona, SNOPT -----------
        diff = max( abs( (kona_x - pyopt_x)/np.linalg.norm(pyopt_x) ) )

        # ------------- output ------------ #
        sep_str = '----------- Case %d ------------'%i
        err_diff = 'Kona and SNOPT solution X maximum relative difference, ' + str(diff)
        kona_obj = 'Kona obj, ' + str(kona_obj)
        pyopt_obj = 'SNOPT obj, ' + str(pyopt_obj)

        with open(outdir+'/kona_optns.txt', 'a') as file:
            pprint.pprint(err_diff, file)
            pprint.pprint(kona_obj, file)
            pprint.pprint(pyopt_obj, file) 


    #---- write init_norms, wrong_sols ---- # 
    file_ =  outdir +'/'+ str(num_design) + '_time'     
    A_file = open(file_, 'w')
    pickle.dump([kona_times, snopt_times], A_file)
    A_file.close()


if args.task=='post':

    num_designs = np.array([100,200,300,400,500])
    num_cases = len(num_designs) 

    kona_mean_all = np.zeros(num_cases)
    kona_std_all = np.zeros(num_cases)

    snopt_mean_all = np.zeros(num_cases)
    snopt_std_all = np.zeros(num_cases)

    for k in range(num_cases):
        num_design = num_designs[k]

        file_ =  outdir +'/'+ str(num_design) + '_time' 
        A_file = open(file_, 'r')
        a = pickle.load(A_file)
        A_file.close()

        kona_times = a[0]
        snopt_times = a[1]

        kona_mean_all[k] = np.mean(kona_times)
        kona_std_all[k] = np.std(kona_times)

        snopt_mean_all[k] = np.mean(snopt_times)
        snopt_std_all[k] = np.std(snopt_times)

    # ---------------------------------------------
    # plot the data
    # ms = markersize
    # mfc = markerfacecolor     mec = 'k'
    # mew = markeredgewidth
    axis_fs = 12 # axis title font size
    axis_lw = 1.0 # line width used for axis box, legend, and major ticks
    label_fs = 11 # axis labels' font size


    # set figure size in inches, and crete a single set of axes
    fig = plt.figure(figsize=(7,4), facecolor='w')
    ax = fig.add_subplot(111)

    # plot the data
    # mfc = markerfacecolor
    # mec = markeredgecolor
    # ms = markersize
    # mew = markeredgewidth

    if pic_color == 1:
        kona_plt = ax.errorbar(num_designs, kona_mean_all, kona_std_all, fmt='r-o', elinewidth=1.5, 
            linewidth=1.5, mfc='r', ms=4.0, mew=1.5, mec='r', capsize=5)

        snopt_plt = ax.errorbar(num_designs, snopt_mean_all, snopt_std_all, fmt='b-s', elinewidth=1.5,  
            linewidth=1.5, mfc='b', ms=4.0, mew=1.5, mec='b', capsize=5)
        # elinewidth  
    else:
        kona_plt = ax.errorbar(num_designs, kona_mean_all, kona_std_all, fmt='k-o', elinewidth=1.5, 
            linewidth=1.5, mfc=(0.35,0.35,0.35), ms=4.0, mew=1.5, mec='k', capsize=5)

        snopt_plt = ax.errorbar(num_designs, snopt_mean_all, snopt_std_all, fmt='k-s', elinewidth=1.5,  
            linewidth=1.5, mfc=(0.35,0.35,0.35), ms=4.0, mew=1.5, mec='k', capsize=5)

    # Tweak the appeareance of the axes
    ax.axis([min(num_designs)-30, max(num_designs)+30, 0, max(snopt_mean_all+snopt_std_all)+1])  # axes ranges
    ax.set_position([0.12, 0.13, 0.86, 0.83]) # position relative to figure edges
    ax.set_xlabel('Number of Design Variables', fontsize=axis_fs, weight='bold')
    ax.set_ylabel('CPU time', fontsize=axis_fs, weight='bold', \
                  labelpad=12)
    ax.grid(which='major', axis='y', linestyle='--')
    ax.set_axisbelow(True) # grid lines are plotted below
    rect = ax.patch # a Rectangle instance
    #ax.yaxis.set_ticks(np.arange(-1, max(wrong_sols)+1.5, 1))
    #rect.set_facecolor('white')
    #rect.set_ls('dashed')
    rect.set_linewidth(axis_lw)
    rect.set_edgecolor('k')

    # ticks on bottom and left only
    ax.xaxis.tick_bottom() # use ticks on bottom only
    ax.yaxis.tick_left()
    for line in ax.xaxis.get_ticklines():
        line.set_markersize(6) # length of the tick
        line.set_markeredgewidth(axis_lw) # thickness of the tick
    for line in ax.yaxis.get_ticklines():
        line.set_markersize(6) # length of the tick
        line.set_markeredgewidth(axis_lw) # thickness of the tick
    for label in ax.xaxis.get_ticklabels():
        label.set_fontsize(label_fs)
    for label in ax.yaxis.get_ticklabels():
        label.set_fontsize(label_fs)


    # define and format the minor ticks
    # ax.xaxis.set_ticks(np.arange(0,max(init_norms)+3,1),minor=True)
    ax.xaxis.set_tick_params(which='minor', length=3, width=2.0*axis_lw/3.0)
    # ax.yaxis.set_ticks(np.arange(-1, max(wrong_sols)+2,1),minor=True)
    # ax.set_yticklabels([i for i in wrong_sols])
    ax.yaxis.set_tick_params(which='minor', length=3, width=2.0*axis_lw/3.0)
        #print ax.xaxis.get_ticklines(minor=True)

    # turn off tick on right and upper edges; this is now down above
    #for tick in ax.xaxis.get_major_ticks():
    #    tick.tick2On = False
    #for tick in ax.yaxis.get_major_ticks():
    #    tick.tick2On = False


    leg = ax.legend([kona_plt, snopt_plt], ['Homotopy RSNK', 'SNOPT'], \
                    loc=(0.02, 0.80), numpoints=1, prop={'size':6},  borderpad=0.75, handlelength=4)
    rect = leg.get_frame()
    rect.set_linewidth(axis_lw)
    for t in leg.get_texts():
        t.set_fontsize(12)    
    plt.show()
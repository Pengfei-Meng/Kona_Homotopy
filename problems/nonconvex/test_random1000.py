import numpy as np
import unittest
import pprint
import os, pickle
import matplotlib.pyplot as plt
import kona
from kona import Optimizer 
from kona.algorithms import PredictorCorrectorCnstrCond, Verifier
from nonconvex import NONCONVEX
import time
import pdb
from pyoptsparse import Optimization, OPT
from kona.linalg.matrices.hessian import LimitedMemoryBFGS
import argparse

""" 1000 randomly generated initial points for 
    1000 randomly generated nonconvex quadratic problems
"""

# np.random.seed(1) 

parser = argparse.ArgumentParser()
parser.add_argument("--output", help='Output directory', type=str, default='./random')
parser.add_argument("--task", help='what to do', choices=['opt','post'], default='opt')
args = parser.parse_args()


outdir = args.output    # './random'
if not os.path.isdir(outdir):
    os.mkdir(outdir)

print args.task

post_dir = './random_1000'

lb = -2
ub = 2

num_design = 100
num_case = 1000


if args.task == 'opt': 

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
        print 'Starting case %d'%i
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

    #---- write init_norms, wrong_sols ---- # 
    file_ =  outdir + '/design'     
    A_file = open(file_, 'w')
    pickle.dump([init_norms, wrong_sols], A_file)
    A_file.close()


if args.task=='post':
    # ------------------ Make Plots ------------------
    file_ =  post_dir + '/design' 
    A_file = open(file_, 'r')
    a = pickle.load(A_file)
    A_file.close()

    init_norms = a[0]
    wrong_sols = a[1]

    sols_mean = np.mean(wrong_sols)
    sols_std = np.std(wrong_sols)

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
    # ms = markersize
    # mfc = markerfacecolor
    # mew = markeredgewidth
    dynamic = ax.plot(init_norms, wrong_sols, 'ko', linewidth=1.5, ms=8.0, \
                  mfc=(0.35,0.35,0.35), mew=1.5, mec='k')
    #qn = ax.plot(ndv, qn/flow_cost, '-k^', linewidth=2.0, ms=8.0, mfc='w', mew=1.5, \
    #         color=(0.35, 0.35, 0.35), mec=(0.35, 0.35, 0.35))

    # Tweak the appeareance of the axes
    ax.axis([0, max(init_norms)+3, -1, max(wrong_sols)+1])  # axes ranges
    ax.set_position([0.12, 0.13, 0.86, 0.83]) # position relative to figure edges
    ax.set_xlabel('Norm of Initial Starting Point', fontsize=axis_fs, weight='bold')
    ax.set_ylabel('Number of Stationery Points', fontsize=axis_fs, weight='bold', \
                  labelpad=12)
    ax.grid(which='major', axis='y', linestyle='--')
    ax.set_axisbelow(True) # grid lines are plotted below
    rect = ax.patch # a Rectangle instance
    ax.yaxis.set_ticks(np.arange(-1, max(wrong_sols)+1.5, 1))
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
    ax.xaxis.set_ticks(np.arange(0,max(init_norms)+3,1),minor=True)
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

    # # plot and tweak the legend
    # leg = ax.legend(('fixed tol.'), loc=(0.02,0.75), numpoints=1, \
    #                 borderpad=0.75, \
    #                 handlelength=4) # handlelength controls the width of the legend
    # rect = leg.get_frame()
    # rect.set_linewidth(axis_lw)
    # for t in leg.get_texts():
    #     t.set_fontsize(12)    # the legend text fontsize

    # --------------- Text ---------------
    # ------------- Counting -------------
    bbox_props = dict(boxstyle="round", fc="w", ec="0.5", alpha=0.9)



    for k in np.arange(0, max(wrong_sols)+1, 1):

        cout = sum(wrong_sols == k)
        text = "{0:.0f}%".format(cout*1.0/num_case * 100) + ", or " + str(cout) 
        ax.text(6.5, k, text, ha="center", va="center", size=12,
                bbox=bbox_props)        

    # text = "Mean: .1f}".format(sols_mean) + '\n' + " STD: {%.1f}".format(sols_std)
    text = "Mean: %.2f"%(sols_mean) + '\n' + " STD: %.2f"%(sols_std)
    ax.text(3, 2.5, text, ha="center", va="center", size=12,
        bbox=bbox_props)  

    plt.show()
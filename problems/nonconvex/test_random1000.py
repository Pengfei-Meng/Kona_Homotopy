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
    remember to comment out the np.random.seed(0) in NONCONVEX problem.
    python test_random1000.py --output 'temp' --task 'opt' --num_case 100
    python test_random1000.py --output 'temp' --task 'post'
 
                 bad    83%    82%    77%    bad     87%     81%     62%
init_step:       0.01   0.01   0.01   0.01   0.01    0.01   0.01    0.01
nominal dist:    0.1    0.1    0.1    0.1    0.01    0.5    1       1
nominal angle:    1      5      10     20     5      5      5       10

                  83%    76%    83%
init_step:        0.05   0.1    0.1
nominal dist:     0.5    1      0.5 
nominal angle:    5      10     5
"""

# np.random.seed(1) 

parser = argparse.ArgumentParser()
parser.add_argument("--output", help='Output directory', type=str, default='./temp')
parser.add_argument("--task", help='what to do', choices=['opt','post'], default='opt')
parser.add_argument("--num_case", type=int, default=10)
parser.add_argument("--krylov_tol", type=float, default=1e-2)
args = parser.parse_args()

num_case = args.num_case
krylov_tol = args.krylov_tol

outdir = args.output    
if not os.path.isdir(outdir):
    os.mkdir(outdir)

print args.task

post_dir = outdir      #'./random_1000'

lb = -2
ub = 2

num_design = 100


if args.task == 'opt': 

    optOptions = {'Print file': outdir + '/SNOPT_print.out',
                  'Summary file': outdir + '/SNOPT_summary.out',
                  'Problem Type':'Minimize',
                  }
    storeHistory=False


    # Optimizer
    optns = {
        'max_iter' : 300,
        'opt_tol' : 1e-7,
        'feas_tol' : 1e-7,        
        'info_file' : outdir+'/kona_info.dat',
        'hist_file' : outdir+'/kona_hist.dat',

        'homotopy' : {
            'init_homotopy_parameter' : 1.0, 
            'inner_tol' : 0.1,                          
            'inner_maxiter' : 2,                    
            'init_step' : 0.01,                     
            'nominal_dist' : 0.5,                    
            'nominal_angle' : 5*np.pi/180.,          
            'max_factor' : 50.0,                  
            'min_factor' : 0.001,                   
            'dmu_max' : -0.0005,        # -0.0005
            'dmu_min' : -0.9,      
            'mu_correction' : 1.0,  
            'use_frac_to_bound' : True,
            'mu_pc_on' : 1.0,      
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
            'subspace_size' : 40,                                    
            'check_res'     : False,
            'rel_tol'       : krylov_tol,        
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
    wrong_sols_kona = np.zeros(num_case)
    wrong_sols_snopt = np.zeros(num_case)

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

        diff_kona = sum(abs(x_kona - x_true))
        wrong_sols_kona[i] = diff_kona
        print 'Kona wrong points %d'%diff_kona


        #------------- SNOPT Optimization ---------------#

        # def objfunc(xdict):

        #     x = xdict['xvars']
        #     funcs = {}
        #     funcs['obj'] = 0.5*np.dot(x.T,  solver.D*x)

        #     conval = x 
        #     funcs['con'] = conval
        #     fail = False
            
        #     return funcs, fail

        # def sens(xdict, funcs):
        #     x = xdict['xvars']
        #     funcsSens = {}            
        #     funcsSens['obj'] = {'xvars': solver.D*x }
        #     funcsSens['con'] = {'xvars': np.eye(num_design) }
        #     fail = False

        #     return funcsSens, fail


        # optProb = Optimization('nonconvex', objfunc)

        # # Design Variables
        # optProb.addVarGroup('xvars', num_design, value=init_x)

        # # Constraints
        # lower = -np.ones(num_design)
        # upper = np.ones(num_design)
        # optProb.addConGroup('con', num_design, lower = lower, upper = upper)

        # # Objective
        # optProb.addObj('obj')

        # # Optimizer
        # opt = OPT('snopt', options=optOptions)

        # # Solution
        # if storeHistory:
        #     histFileName = '%s.hst' % (optName.lower())
        # else:
        #     histFileName = None

        # sol = opt(optProb, sens=sens, storeHistory=histFileName)  
        
        # # Check Solution
        # pyopt_obj = sol.objectives['obj'].value
        # pyopt_x = np.array(map(lambda x:  sol.variables['xvars'][x].value, xrange(num_design)))

        # x_snopt = np.rint(abs(pyopt_x))

        # diff_snopt = sum(abs(x_snopt - x_true))
        # wrong_sols_snopt[i] = diff_snopt
        # print 'SNOPT wrong points %d'%diff_snopt

        # ------------ Studying Wrong Solutions ------------
        # diff_sol_snopt = abs(x_snopt - x_true)
        # wrong_snopt = 'wrong snopt: ' + str(x_snopt[diff_sol_snopt > 0])


        # ------------ Studying Wrong Solutions ------------
        sep_str = '----------- Case %d ------------'%i
        diff_sol = abs(x_kona - x_true)
        wrong_D = 'solver.D: ' +  str(solver.D[diff_sol > 0])
        wrong_initx = 'init x: ' + str(init_x[diff_sol > 0])
        wrong_kona = 'wrong kona: ' + str(x_kona[diff_sol > 0])
        wrong_true = 'wrong true: ' +  str(x_true[diff_sol > 0])
        # wrong_snopt = 'wrong snopt: ' + str(x_snopt[diff_sol > 0])
        
        with open(outdir+'/kona_optns.txt', 'a') as file:
            if diff_kona > 0: 
                pprint.pprint(sep_str, file)
                pprint.pprint(wrong_D, file)
                pprint.pprint(wrong_initx, file)
                pprint.pprint(wrong_true, file)
                pprint.pprint(wrong_kona, file)

            # if diff_snopt > 0:
                # pprint.pprint(wrong_snopt, file)


        # # ------------- output ------------ #
        # sep_str = '----------- Case %d ------------'%i
        # D_str = 'solver.D: ' + str(solver.D)
        # init_xs = 'init x: ' + str(init_x)
        # # kona_sol = 'kona_solution: ' + str(kona_x)
        # x_kona = 'kona_solution: ' + str(x_kona)
        # x_snopt = 'snopt_solution: ' + str(x_snopt)
        # true_sol = 'true solution: ' + str(x_true)
        # err_diff = 'Kona number of wrong solutions: ' + str(diff_kona)
        # err_diff_sn = 'SNOPT number of wrong solutions: ' + str(diff_snopt)

        # with open(outdir+'/kona_optns.txt', 'a') as file:
        #     pprint.pprint(sep_str, file)
        #     pprint.pprint(D_str, file)
        #     pprint.pprint(init_xs, file)
        #     pprint.pprint(true_sol, file)
        #     pprint.pprint(x_kona, file)
        #     pprint.pprint(err_diff, file)
        #     pprint.pprint(x_snopt, file)            
        #     pprint.pprint(err_diff_sn, file)


    #---- write init_norms, wrong_sols_kona ---- # 
    file_ =  outdir + '/design' + str(krylov_tol)    
    A_file = open(file_, 'w')
    pickle.dump([init_norms, wrong_sols_kona, wrong_sols_snopt], A_file)
    A_file.close()


if args.task=='post':
    # ------------------ Make Plots ------------------
    file_ =  post_dir + '/design' + str(krylov_tol)    
    A_file = open(file_, 'r')
    a = pickle.load(A_file)
    A_file.close()

    init_norms = a[0]
    wrong_sols_kona = a[1]
    wrong_sols_snopt = a[2]

    sols_mean_kona = np.mean(wrong_sols_kona)
    sols_std_kona = np.std(wrong_sols_kona)

    sols_mean_snopt = np.mean(wrong_sols_snopt)
    sols_std_snopt = np.std(wrong_sols_snopt)


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
    dynamic = ax.plot(init_norms, wrong_sols_kona, 'ko', linewidth=1.5, ms=8.0, \
                  mfc=(0.35,0.35,0.35), mew=1.5, mec='k')
    #qn = ax.plot(ndv, qn/flow_cost, '-k^', linewidth=2.0, ms=8.0, mfc='w', mew=1.5, \
    #         color=(0.35, 0.35, 0.35), mec=(0.35, 0.35, 0.35))

    # Tweak the appeareance of the axes
    ax.axis([0, max(init_norms)+3, -1, max(wrong_sols_kona)+1])  # axes ranges
    ax.set_position([0.12, 0.13, 0.86, 0.83]) # position relative to figure edges
    ax.set_xlabel('Norm of Initial Starting Point', fontsize=axis_fs, weight='bold')
    ax.set_ylabel('Number of Stationery Points', fontsize=axis_fs, weight='bold', \
                  labelpad=12)
    ax.grid(which='major', axis='y', linestyle='--')
    ax.set_axisbelow(True) # grid lines are plotted below
    rect = ax.patch # a Rectangle instance
    ax.yaxis.set_ticks(np.arange(-1, max(wrong_sols_kona)+1.5, 1))
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
    # ax.yaxis.set_ticks(np.arange(-1, max(wrong_sols_kona)+2,1),minor=True)
    # ax.set_yticklabels([i for i in wrong_sols_kona])
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



    for k in np.arange(0, max(wrong_sols_kona)+1, 1):

        cout = sum(wrong_sols_kona == k)
        text = "{0:.1f}%".format(cout*1.0/num_case * 100) + ", or " + str(cout) 
        ax.text(6.5, k, text, ha="center", va="center", size=12,
                bbox=bbox_props)        

    # text = "Mean: .1f}".format(sols_mean) + '\n' + " STD: {%.1f}".format(sols_std)
    text = "Mean: %.2f"%(sols_mean_kona) + '\n' + " STD: %.2f"%(sols_std_kona)
    ax.text(3, 2.5, text, ha="center", va="center", size=12,
        bbox=bbox_props)  

    plt.show()

    fig_name = post_dir  + '/nonconvex_1000.eps' 
    fig.savefig(fig_name, format='eps', dpi=1200)

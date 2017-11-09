import numpy as np
import unittest
import pprint
import os

import kona
from kona import Optimizer 
from kona.algorithms import PredictorCorrectorCnstrCond, Verifier
# from kona.examples import Constructed_SVDA
from construct_svdA import Constructed_SVDA
import time
import pdb
from pyoptsparse import Optimization, OPT
from kona.linalg.matrices.hessian import LimitedMemoryBFGS


class InequalityTestCase(unittest.TestCase):

    def setUp(self):

        self.outdir = './output2/sm_W'
        if not os.path.isdir(self.outdir):
            os.mkdir(self.outdir)

        size_prob = 100

        self.num_design = size_prob
        self.num_ineq = size_prob
        np.random.seed(0) 
        self.init_x = np.random.rand(size_prob)    

        self.solver = Constructed_SVDA(self.num_design, self.num_ineq, self.init_x, self.outdir)

    def kona_optimize(self):

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
                'init_step' : 60.0,                # 100-500 : 60 100 100 100 150 
                'nominal_dist' : 10.0,              # 100-500 : 10 10 20 30  30
                'nominal_angle' : 20.0*np.pi/180.,  # 100-500 : 20 20 20 30  30
                'max_factor' : 30.0,                  
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
                'beta'         : 1.0, 
                'cmin'         : 1e-1,   # negative value, cut-off ineffective; 
            }, 

            'rsnk' : {
                'precond'       : 'svd_pc_cmu',    #None, #,  #'svd_pc_cmu',                  
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
        # import pdb; pdb.set_trace()

    def objfunc(self, xdict):
        self.iteration += 1
        self.fun_obj_counter += 1

        x = xdict['xvars']
        funcs = {}
        funcs['obj'] = 0.5 * np.dot(x.T, np.dot(self.solver.Q, x)) + np.dot(self.solver.g, x) 

        conval = np.dot(self.solver.A,x) - self.solver.b
        funcs['con'] = conval
        fail = False

        return funcs, fail

    def sens(self, xdict, funcs):

        x = xdict['xvars']
        funcsSens = {}
        funcsSens['obj'] = {'xvars': np.dot(x.T, self.solver.Q) + self.solver.g }
        funcsSens['con'] = {'xvars': self.solver.A }
        fail = False

        # ---------- recording ----------
        self.iteration += 1
        self.sens_counter += 1
        self.endTime_sn = time.clock()
        self.duration_sn = self.endTime_sn - self.startTime_sn
        self.totalTime_sn += self.duration_sn
        self.startTime_sn = self.endTime_sn

        timing = '  {0:3d}        {1:4.2f}        {2:4.2f}        {3:4.6g}     \n'.format(
            self.sens_counter, self.duration_sn, self.totalTime_sn,  funcs['obj'] )
        file = open(self.outdir+'/SNOPT_timings.dat', 'a')
        file.write(timing)
        file.close()


        return funcsSens, fail

    def optimize(self, optName, optOptions={}, storeHistory=False):

        # Optimization Object
        optProb = Optimization('SVD_Construct Problem', self.objfunc)

        # Design Variables
        value = self.init_x
        optProb.addVarGroup('xvars', self.num_design, value=value)

        # Constraints
        lower = np.zeros(self.num_ineq)
        upper = [None]*self.num_ineq
        optProb.addConGroup('con', self.num_ineq, lower = lower, upper = upper)

        # Objective
        optProb.addObj('obj')

        # Check optimization problem:
        # print(optProb)

        # Optimizer
        try:
            opt = OPT(optName, options=optOptions)
        except:
            raise unittest.SkipTest('Optimizer not available:', optName)

        # Solution
        if storeHistory:
            histFileName = '%s_svdConstruct.hst' % (optName.lower())
        else:
            histFileName = None

        sol = opt(optProb, sens=self.sens, storeHistory=histFileName)   # 

        
        # Check Solution
        self.pyopt_obj = sol.objectives['obj'].value
        self.pyopt_x = np.array(map(lambda x:  sol.variables['xvars'][x].value, xrange(self.num_design)))
        


    def test_snopt(self):

        self.iteration = 0
        self.fun_obj_counter = 0
        self.sens_counter = 0

        self.startTime_sn = time.clock()
        self.totalTime_sn = 0
        self.endTime_sn = 0
        file = open(self.outdir+'/SNOPT_timings.dat', 'w')
        file.write('# SNOPT iteration timing history\n')
        titles = '# {0:s}    {1:s}    {2:s}    {3:s}  \n'.format(
            'Iter', 'Time (s)', 'Total Time (s)', 'Objective')
        file.write(titles)
        file.close()

        optOptions = {'Print file':self.outdir + '/SNOPT_print.out',
                      'Summary file':self.outdir + '/SNOPT_summary.out',
                      'Problem Type':'Minimize',
                      }

        self.optimize('snopt', optOptions)
        

        # ------ Kona Opt --------

        self.solver.iterations = 0
        self.solver.duration = 0.
        self.solver.totalTime = 0.
        self.solver.startTime = 0.
        self.solver.startTime = time.clock()
        file = open(self.outdir+'/kona_timings.dat', 'w')
        file.write('# Constructed_SVDA iteration timing history\n')
        titles = '# {0:s}    {1:s}    {2:s}    {3:s}    {4:s}   {5:s}   {6:s}\n'.format(
            'Iter', 'Time (s)', 'Total Time (s)', 'Objective', 'max(abs(-S*Lam))', 'negative S', 'postive Lam' )
        file.write(titles)
        file.close()

        self.kona_optimize()
        
        diff = max( abs( (self.kona_x - self.pyopt_x)/np.linalg.norm(self.pyopt_x) ) )


        err_diff = 'Kona and SNOPT solution X maximum relative difference, ' + str(diff)
        kona_obj = 'Kona objective value at the solution, ' + str(self.kona_obj)
        pyopt_obj = 'SNOPT objective value at the solution, ' + str(self.pyopt_obj)
        pos_dual = 'Positive dual, ' + str(self.solver.curr_dual[self.solver.curr_dual > 1e-5])
        neg_slack = 'Negative Slack, ' + str(self.solver.curr_slack[self.solver.curr_slack < -1e-5])

        with open(self.outdir+'/kona_optns.txt', 'a') as file:
            pprint.pprint(err_diff, file)
            pprint.pprint(kona_obj, file)
            pprint.pprint(pyopt_obj, file)
            pprint.pprint(pos_dual, file)
            pprint.pprint(neg_slack, file)


        print err_diff

        print 'kona_obj %f, '%(self.kona_obj)
        print 'pyopt_obj %f, '%(self.pyopt_obj)



        # print 'kona_x', self.kona_x
        # print 'pyopt_x', self.pyopt_x
        # print 'max A', max(self.solver.A.max(axis=0))
        # print 'min A', min(self.solver.A.min(axis=0))

        # self.optimize('slsqp')
        # diff = max( abs(self.kona_x - self.pyopt_x)/np.linalg.norm(self.pyopt_x) )

        # print 'SLSQP relative difference, ', diff
        # print 'kona_obj %f, time %f'%(self.kona_obj, self.kona_time)
        # print 'pyopt_obj %f, time %f'%(self.pyopt_obj, self.pyopt_time)

    # def test_slsqp(self):
    #     self.optimize('slsqp')
    #     self.kona_optimize()

    #     diff = max( (self.kona_x - self.pyopt_x)/np.linalg.norm(self.pyopt_x) )

    #     print 'SLSQP relative difference, ', diff
    #     print 'kona_obj %f, time %f'%(self.kona_obj, self.kona_time)
    #     print 'pyopt_obj %f, time %f'%(self.pyopt_obj, self.pyopt_time)

    # def test_nlpqlp(self):
    #     self.optimize('nlpqlp')

    # def test_fsqp(self):
    #     self.optimize('fsqp')

    # def test_ipopt(self):
    #     self.optimize('ipopt')


    # def test_conmin(self):
    #     self.optimize('conmin')

    # def test_psqp(self):
    #     self.optimize('psqp')

        #    optOptions = {
        #               'Major feasibility tolerance':1.0e-6,
        #               'Major optimality tolerance':1.00e-6, 
        #               'Major print level':1,
        #               'Minor print level':1,        
        #               'Major iterations limit':500,
        #               'Minor iterations limit':500,
        #               'Major step limit':.01,
        #               'Nonderivative linesearch':None,
        #               'Function precision':1.0e-6,
        #               'Print file':self.outdir + '/SNOPT_print.out',
        #               'Summary file':self.outdir + '/SNOPT_summary.out',
        #               'Problem Type':'Minimize',
        #               'Timing level': 3,
        #              }


if __name__ == "__main__":
    unittest.main()
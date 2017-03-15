import numpy as np
import unittest

import kona
from kona import Optimizer 
from kona.algorithms import PredictorCorrectorCnstrCond, Verifier
from kona.examples import Constructed_SVDA
import timeit
import pdb
# from __future__ import print_function
from pyoptsparse import Optimization, OPT


class InequalityTestCase(unittest.TestCase):

    def setUp(self):

        self.num_design = 500
        self.num_ineq = 500
        self.init_x = np.zeros(500)   #np.random.rand(20)    # np.zeros(300)

        start = timeit.timeit()
        self.solver = Constructed_SVDA(self.num_design, self.num_ineq, self.init_x)
        end = timeit.timeit()
        self.setup_time = end - start

    def kona_optimize(self):

        prefix = './temp/'
    
        # Optimizer
        optns = {
            'max_iter' : 200,
            'opt_tol' : 1e-7,
            'feas_tol' : 1e-7,        
            'info_file' : prefix+'/kona_info.dat',
            'hist_file' : prefix+'/kona_hist.dat',

            'homotopy' : {
                'init_homotopy_parameter' : 1.0, 
                'inner_tol' : 0.1,
                'inner_maxiter' : 5,
                'init_step' : 0.05,        
                'nominal_dist' : 1.0,            
                'nominal_angle' : 8.0*np.pi/180., 
                'max_factor' : 30.0,                  
                'min_factor' : 0.5,                   
                'dmu_max' : -0.0005,       
                'dmu_min' : -0.9,      
                'mu_correction' : 0.1,  
            }, 

            'rsnk' : {
                'precond'       : 'svd_pc',   # 'approx_adjoint', # None,  #  
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
                'krylov_file'   : prefix+'/kona_krylov.dat',
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
                'out_file'       : prefix+'/kona_verify.dat',
            },
        }

        
        # algorithm = kona.algorithms.Verifier
        algorithm = kona.algorithms.PredictorCorrectorCnstrCond
        optimizer = kona.Optimizer(self.solver, algorithm, optns)
        optimizer.solve()

        self.kona_obj = self.solver.eval_obj(self.solver.curr_design, self.solver.curr_state)
        self.kona_x = self.solver.curr_design


    def objfunc(self, xdict):
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

        sol = opt(optProb, sens=self.sens, storeHistory=histFileName)

        
        # Check Solution
        self.pyopt_obj = sol.objectives['obj'].value
        self.pyopt_x = np.array(map(lambda x:  sol.variables['xvars'][x].value, xrange(self.num_design)))
        


    def test_snopt(self):

        pyopt_start = timeit.timeit()
        self.optimize('snopt')
        pyopt_end = timeit.timeit()
        self.pyopt_time = pyopt_end - pyopt_start

        kona_start = timeit.timeit()
        self.kona_optimize()
        kona_end = timeit.timeit()
        self.kona_time = kona_end - kona_start

        diff = max( abs( (self.kona_x - self.pyopt_x)/np.linalg.norm(self.pyopt_x) ) )

        print 'SNOPT  relative difference, ', diff
        print 'kona_obj %f, '%(self.kona_obj)
        print 'pyopt_obj %f, '%(self.pyopt_obj)
        print 'setup_time %f, kona_time %f, pyopt_time %f'%(self.setup_time, self.kona_time, self.pyopt_time)
        # print 'kona_x', self.kona_x
        # print 'pyopt_x', self.pyopt_x

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

if __name__ == "__main__":
    unittest.main()
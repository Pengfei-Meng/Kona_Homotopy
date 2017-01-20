import numpy as np
from kona.user import UserSolver
import pdb
# import cutermgr


class KONA_CUTER(UserSolver):

    def __init__(self, prob_name='BT11'):

        if cutermgr.isCached(prob_name): 
            self.prob=cutermgr.importProblem(prob_name)
        else:
            cutermgr.prepareProblem(prob_name, efirst=True, nvfirst=True)            
            self.prob=cutermgr.importProblem(prob_name)

        info=self.prob.getinfo()
        self.init_x = info['x']

        num_design = info['n']
        num_state = 0
        num_eq = sum(info['equatn'])

        # no. of inequality constraints excluding bounds/2
        self.bl = info['bl']
        self.bu = info['bu']
        self.cl = info['cl']
        self.cu = info['cu']

        # special treatment for bound constraints
        # if self.bl ~ -1e20 or self.bu ~ 1e20 
        # then this bound constraint doesn't exists
        num_inef_bl = 0
        num_inef_bu = 0

        # effective lower bound
        if sum(abs(self.bl) > 1e6) > 0: 
            num_inef_bl = sum(abs(self.bl) > 1e10)
            print '%d ineffective lower bound on design ' %num_inef_bl

        if sum(abs(self.bu) > 1e6) > 0: 
            num_inef_bu = sum(abs(self.bu) > 1e10)
            print '%d ineffective upper bound on design ' %num_inef_bu

        # effective upper bound
        num_bnd = 2*num_design - num_inef_bl - num_inef_bu      
        num_ineq= 2*(info['m'] - num_eq) + num_bnd

        self.num_bnd = num_bnd
        self.m_ineq = info['m'] - num_eq  

        self.bl_eff = abs(self.bl) < 1e6
        self.bu_eff = abs(self.bu) < 1e6
        # self.bnd_eff = np.concatenate(self.bl_eff, self.bu_eff)

        assert sum(abs(self.cl) > 1e6) == 0, 'self.cl -infinity, check'
        assert sum(abs(self.cu) > 1e6) == 0, 'self.cu infinity, check'

        super(KONA_CUTER, self).__init__(
            num_design, num_state, num_eq, num_ineq)

    def eval_obj(self, at_design, at_state):
        result = self.prob.obj(at_design)
        return np.asscalar(result)

    def eval_residual(self, at_design, at_state, store_here):
        store_here.data[:] = 0.

    def eval_dFdX(self, at_design, at_state):
        f,g = self.prob.obj(at_design, True)
        return g

    def eval_eq_cnstr(self, at_design, at_state):
        if self.num_eq == 0:       # efirst = True
           c_e = np.zeros(self.num_eq)
        else:
            c = self.prob.cons(at_design)
            c_e = c[:self.num_eq]
        return c_e

    def eval_ineq_cnstr(self, at_design, at_state):

        if self.num_bnd > 0:   # effective bound constraints exist
            c_bnd_bl = at_design[self.bl_eff] - self.bl[self.bl_eff]
            c_bnd_bu = -at_design[self.bu_eff] + self.bu[self.bu_eff]
            c_bnd = np.concatenate([c_bnd_bl, c_bnd_bu])

        if self.m_ineq > 0:    # non bound inequality constraints exist
            c = self.prob.cons(at_design)
            c_ineq_orig = c[self.num_eq:]
            c_ineq_2orig = np.concatenate([c_ineq_orig - self.cl,
                                          -c_ineq_orig + self.cu])

        if self.num_bnd > 0:  # bound constr exists
            if self.m_ineq > 0:
                c_ineq = np.concatenate([c_ineq_2orig, c_bnd])
            else:   # no inequ constrs
                c_ineq = c_bnd
        else:
            if self.m_ineq > 0:
                c_ineq = c_ineq_2orig

        return c_ineq
        
    def multiply_dCEQdX(self, at_design, at_state, in_vec):
        if self.num_eq == 0:
            return np.zeros_like(in_vec)
        else:
            assert len(in_vec) == self.num_design, "Incorrect in_vec size!"
            f, g = self.prob.cons(at_design, True)
            g_eq_orig = g[:self.num_eq, :]
            return g_eq_orig.dot(in_vec)

    def multiply_dCEQdX_T(self, at_design, at_state, in_vec):
        if self.num_eq == 0:
            return np.zeros_like(in_vec)
        else:
            assert len(in_vec) == self.num_eq, "Incorrect in_vec size"
            f, g = self.prob.cons(at_design, True)
            g_eq_orig = g[:self.num_eq, :]
            g_eq = g_eq_orig.transpose()
            return g_eq.dot(in_vec)

    def multiply_dCINdX(self, at_design, at_state, in_vec):
        assert len(in_vec) == self.num_design, "Incorrect in_vec size!"
        if self.num_bnd > 0:   # effective bound constraints exist
            out_bnd_bl = in_vec[self.bl_eff]
            out_bnd_bu = -in_vec[self.bu_eff]  
            out_bnd = np.concatenate([out_bnd_bl, out_bnd_bu])

        if self.m_ineq > 0:    # non bound inequality constraints exist
            f, g = self.prob.cons(at_design, True)
            g_ineq_orig = g[self.num_eq:, :]
            out_g_ineq = g_ineq_orig.dot(in_vec)
            out_2ineq = np.concatenate([out_g_ineq, -out_g_ineq])

        if self.num_bnd > 0:  # bound constr exists
            if self.m_ineq > 0:
                c_ineq = np.concatenate([out_2ineq, out_bnd])
            else:   # no inequ constrs
                c_ineq = out_bnd
        else:
            if self.m_ineq > 0:
                c_ineq = out_2ineq

        return c_ineq


    def multiply_dCINdX_T(self, at_design, at_state, in_vec):

        assert len(in_vec) == self.num_ineq, "Incorrect in_vec size!"


        if self.m_ineq > 0:    # non bound inequality constraints exist
            f, g = self.prob.cons(at_design, True)
            g_ineq_orig = g[self.num_eq:, :]
            g_ineq = g_ineq_orig.transpose()   
            out_1ineq = g_ineq.dot(in_vec[:self.m_ineq])
            out_2ineq = (-g_ineq).dot(in_vec[self.m_ineq:(2*self.m_ineq)])

        if self.num_bnd > 0:   # effective bound constraints exist
            in_vec_bnd = in_vec[2*self.m_ineq:]
            in_bnd_bl = in_vec_bnd[:sum(self.bl_eff)]
            in_bnd_bu = in_vec_bnd[sum(self.bl_eff):] 

            out_bnd_bl = in_bnd_bl[self.bl_eff]
            out_bnd_bu = -in_bnd_bu[self.bu_eff]  
            
        if self.num_bnd > 0:  # bound constr exists
            if self.m_ineq > 0:
                out_ineq =  out_1ineq + out_2ineq + out_bnd_bl +  out_bnd_bu
            else:   # no inequ constrs
                out_ineq = out_bnd_bl +  out_bnd_bu
        else:
            if self.m_ineq > 0:
                out_ineq = out_1ineq + out_2ineq

        return out_ineq


    def init_design(self):
        return self.init_x

    def current_solution(self, num_iter, curr_design, curr_state, curr_adj,
                         curr_eq, curr_ineq, curr_slack):
        self.curr_design = curr_design
        self.curr_state = curr_state


"""
Kona wrapper for Cuter Optimization test problems

The following info is copied from PyCUTEr wrapper: cutermgr.py 

# Arguments to cutermgr.prepareProblem()
# sifParams: sifParams={'NN': 10}   
#            sifOptions=['-param', 'NN=10']
# # Specifying variable and constraint ordering
# nvfirst: Nonlinear variables before linear variables
# efirst: Equality constraints before inequality constraints
# lfirst: Linear constraints can be placed before nonlinear

The manager module (``cutermgr``) offers the following functions:

* :func:`clearCache` -- remove a compiled problem from cache
* :func:`prepareProblem` -- compile a problem, place it in the cache, 
  and return a reference to the imported problem module
* :func:`importProblem` -- import an already compiled problem from cache 
  and return a reference to its module

Every problem module has several functions that access the corresponding problem's CUTEr tools:

* :func:`getinfo` -- get problem description
* :func:`varnames` -- get names of problem's variables
* :func:`connames` -- get names of problem's constraints
* :func:`objcons` -- objective and constraints
* :func:`obj` -- objective and objective gradient
* :func:`cons` -- constraints and constraints gradients/Jacobian
* :func:`lagjac` -- gradient of objective/Lagrangian and constraints Jacobian
* :func:`jprod` -- product of constraints Jacobian with a vector
* :func:`hess` -- Hessian of objective/Lagrangian
* :func:`ihess` -- Hessian of objective/constraint
* :func:`hprod` -- product of Hessian of objective/Lagrangian with a vector
* :func:`gradhess` -- gradient and Hessian of objective (if m=0) or
  gradient of objective/Lagrangian, Jacobian, and Hessian of Lagrangian (if m > 0)
* :func:`scons` -- constraints and sparse Jacobian of constraints
* :func:`slagjac` -- gradient of objective/Lagrangian and sparse Jacobian
* :func:`sphess` -- sparse Hessian of objective/Lagrangian
* :func:`isphess` -- sparse Hessian of objective/constraint
* :func:`gradsphess` -- gradient and sparse Hessian of objective (if m=0) or
  gradient of objective/Lagrangian, sparse Jacobian, and sparse Hessian of Lagrangian (if m > 0)
* :func:`report` -- get usage statistics


**Problem information**

The problem information dictionary is returned by the :func:`getinfo` problem 
interface function. The dictionary has the following entries

* ``name`` -- problem name
* ``n`` -- number of variables
* ``m`` -- number of constraints (excluding bounds)
* ``x`` -- initial point (1D array of length n)
* ``bl`` -- 1D array of length n with lower bounds on variables
* ``bu`` -- 1D array of length n with upper bounds on variables
* ``nnzh`` -- number of nonzero elements in the diagonal and upper triangle of sparse 
  Hessian
* ``vartype`` -- 1D integer array of length n storing variable type
  0=real, 1=boolean (0 or 1), 2=integer
* ``nvfirst`` -- boolean flag indicating that nonlinear variables were placed before 
  linear variables 
* ``sifparams`` -- parameters passed to sifdecode with the sifParams argument to 
  :func:`prepareProblem`. ``None`` if no parameters were given
* ``sifoptions`` -- additional options passed to sifdecode with the sifOptions 
  argument to :func:`prepareProblem`. ``None`` if no additional options were given.

Additional entries are available if the problem has constraints (m>0):

* ``nnzj`` -- number of nonzero elements in sparse Jacobian of constraints
* ``v`` -- 1D array of length m with initial values of Lagrange multipliers
* ``cl`` -- 1D array of length m with lower bounds on constraint functions
* ``cu`` -- 1D array of length m with upper bounds on constraint functions
* ``equatn`` -- 1D boolean array of length m indicating whether a constraint is an 
  equality constraint
* ``linear`` -- 1D boolean array of length m indicating whether a constraint is a 
  linear constraint
* ``efirst`` -- boolean flag indicating that equality constraints were places 
  before inequality constraints
* ``lfirst`` -- boolean flag indicating that linear constraints were placed before 
  nonlinear constraints

The usage statistics dictionary is returned by the report() problem interface 
function. The dictionary has the following entries

* ``f`` -- number of objective evaluations
* ``g`` -- number of objective gradient evaluations
* ``H`` -- number of objective Hessian evaluations
* ``Hprod`` -- number of Hessian multiplications with a vector
* ``tsetup`` -- CPU time used in setup
* ``trun`` -- CPU time used in run

For constrained problems the following additional members are available

* ``c`` -- number of constraint evaluations
* ``cg`` -- number of constraint gradient evaluations
* ``cH`` -- number of constraint Hessian evaluations 

"""

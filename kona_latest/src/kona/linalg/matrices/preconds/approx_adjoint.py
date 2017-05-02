import numpy as np 
import scipy as sp
from kona.options import get_opt
from kona.linalg.matrices.hessian.basic import BaseHessian
import pdb, time
from kona.linalg.vectors.composite import ReducedKKTVector
from kona.linalg.vectors.composite import CompositePrimalVector
from kona.linalg.vectors.composite import CompositeDualVector
import pickle

class APPROXADJOINT(BaseHessian):
    """
    Specially for the ASO problem, with eq, ineq constraints both present
    This object is a preconditioner for the KKT system using approximate adjoints. 
    Explicit matrices, together with approximate adjointed constrained Jacobian, 
    are explicitly factorized to carry out the precondition work. 
    The precondition system is :

    .. math::
        \\begin{bmatrix}
        \\mathcal{W} && 0 && \tilda{A_h}^T && \tilda{A_g}^T \\\\
        \\ 0   &&   -\Gamma_g  &&  0  &&  -S \\\\
        \\ \tilda{A_h}  &&   0  &&  0  &&  0 \\\\
        \\ \tilda{A_g}  &&   -I  &&  0  &&  0 
        \\end{bmatrix}
        \\begin{bmatrix}
        \\ v_x \\\\
        \\ v_s \\\\
        \\ v_h \\\\
        \\ v_g 
        \\end{bmatrix}
        =
        \\begin{bmatrix}
        \\ u_x \\\\
        \\ u_s \\\\
        \\ u_h \\\\
        \\ u_g 
        \\end{bmatrix}
    """
    update_mat = False
    def __init__(self, vector_factories):    

        super(APPROXADJOINT, self).__init__(vector_factories, None)
        
        self.primal_factory.request_num_vectors(5)
        self.state_factory.request_num_vectors(2)
        if self.eq_factory is not None:
            self.eq_factory.request_num_vectors(3)
        if self.ineq_factory is not None:
            self.ineq_factory.request_num_vectors(5)

        self.W = LagrangianHessian( vector_factories )
        self.Ag = TotalConstraintJacobian( vector_factories )

        self._allocated = False

    def _generate_primal(self):
        if self.ineq_factory is None:
            return self.primal_factory.generate()
        else:
            prim = self.primal_factory.generate()
            dual_ineq = self.ineq_factory.generate()        
            return CompositePrimalVector(prim, dual_ineq)

    def _generate_dual(self):

        if self.ineq_factory is not None:
            if self.eq_factory is not None:
                dual_eq = self.eq_factory.generate()
                dual_ineq = self.ineq_factory.generate()
                out = CompositeDualVector(dual_eq, dual_ineq)
            else:    
                out = self.ineq_factory.generate()
        else:
            out = self.eq_factory.generate()
        return  out

    def _generate_kkt(self):
        primal = self._generate_primal()
        dual = self._generate_dual()
        return ReducedKKTVector(primal, dual)

    def linearize(self, X, state, adjoint, mu=0.0):

        self.W.linearize(X, state, adjoint)
        self.Ag.linearize(X.primal.design, state)

        self.at_design = X.primal.design.base.data
        self.at_slack = X.primal.slack.base.data
        if self.eq_factory is not None:
            self.at_dual_eq = X.dual.eq.base.data
            self.at_dual_ineq = X.dual.ineq.base.data
        else:
            self.at_dual_ineq = X.dual.base.data
        self.at_slack_kona = X.primal.slack
        self.mu = mu

        self.num_design = len(self.at_design)
        self.num_ineq = len(self.at_dual_ineq)

        self.W_full = np.eye(self.num_design)
        self.A_full = np.zeros((self.num_ineq, self.num_design))
        self.A_full[:128, :] = np.eye(self.num_design)
        self.A_full[128:128*2, :] = -1.0*np.eye(self.num_design)
        # self.A_full[128*2:, :] = np.eye(self.num_design)


        if self.update_mat is True:
            print 'approx_precond.W_full .A_full being calculated'
            # inversely retrieving Lagrangian Hessian or Constraint Jacobian
            in_design = self.primal_factory.generate()
            out_design = self.primal_factory.generate()
            out_dual = self.ineq_factory.generate()

            self.W_full = np.zeros((self.num_design, self.num_design))
            self.A_full = np.zeros((self.num_ineq, self.num_design))

            # loop over design variables and start assembling the matrices
            for i in xrange(self.num_design):
                # set the input vector so that we only pluck out one column of the matrix
                in_design.equals(0.0)
                in_design.base.data[i] = 1.
                # # perform the Lagrangian Hessian product and store
                self.W.approx.multiply_W(in_design, out_design)
                self.W_full[:, i] = out_design.base.data
                # perform the Constraint Jacobian product and store
                self.Ag.approx.product(in_design, out_dual)
                self.A_full[:, i] = out_dual.base.data
                # self.A_full[128*2:, :] = out_dual.base.data[128*2:]
    
    def solve(self, in_vec, out_vec):  
        # in_vec  : to be preconditioned
        # out_vec : after preconditioned
        # note: you cannot change in_vec!!!!!!!         Working 

        # specifically for Graeme's Problem
        v_x = in_vec.primal.design.base.data
        v_s = in_vec.primal.slack.base.data
        v_g = in_vec.dual.base.data

        rhs_full = np.hstack([v_x, v_s, v_g])

        # # ----------------- The Full KKT Matrix -------------------
        # KKT_full = np.vstack([np.hstack([self.W_full,  np.zeros((self.num_design, self.num_ineq)),  self.A_full.transpose()]), 
        #                       np.hstack([np.zeros((self.num_ineq, self.num_design)),  -np.diag(self.at_dual_ineq), -np.diag(self.at_slack)]),
        #                       np.hstack([self.A_full, -np.eye(self.num_ineq),  np.zeros((self.num_ineq, self.num_ineq))]) ])

        KKT_full = np.vstack([np.hstack([self.W_full,  np.zeros((self.num_design, self.num_ineq)),  self.A_full.transpose()]), 
                              np.hstack([np.zeros((self.num_ineq, self.num_design)),  -np.diag(self.at_dual_ineq*self.at_slack), -np.diag(self.at_slack)]),
                              np.hstack([self.A_full, -np.diag(self.at_slack),  np.zeros((self.num_ineq, self.num_ineq))]) ])

        # eyes_h = np.hstack([ np.ones(self.num_design), np.ones(self.num_ineq), -np.ones(self.num_ineq) ])    
        eyes_h = np.hstack([ np.ones(self.num_design), self.at_slack, -np.ones(self.num_ineq) ])   
        homo_I = np.diag(eyes_h)        

        #------------------------------------------------------------------         
        KKT = (1-self.mu)*KKT_full + self.mu*homo_I

        start = time.time()
        p_full = sp.linalg.lu_solve(sp.linalg.lu_factor(KKT), rhs_full)
        end = time.time()
        # print 'lu_solve in approx_adjoint time spent in seconds: ', end-start

        p_x = p_full[:self.num_design]
        p_s = p_full[self.num_design:self.num_design + self.num_ineq]     
        p_g = p_full[self.num_design + self.num_ineq:]

        out_vec.primal.design.base.data = p_x
        out_vec.primal.slack.base.data = p_s
        out_vec.dual.base.data = p_g

        self.update_mat = False

    def solve_reduce(self, in_vec, out_vec):
        # 1st Type System Reduction, with mu // Working! 

        v_x = in_vec.primal.design.base.data
        v_s = in_vec.primal.slack.base.data
        v_g = in_vec.dual.base.data

        # 1) 
        # LAM = -(1.0-self.mu)*self.at_dual_ineq + self.mu*np.ones(len(self.at_dual_ineq))
        # GAM = -(1.0-self.mu)**2 * (1.0/LAM) * self.at_slack - self.mu*np.ones(self.at_slack.shape)
        # Scaled Slack
        LAM = -(1.0-self.mu)*self.at_dual_ineq*self.at_slack + self.mu*np.ones(len(self.at_dual_ineq))
        GAM = -(1.0-self.mu)**2 * self.at_slack * (1.0/LAM) * self.at_slack - self.mu*np.ones(len(self.at_slack))
        
        UV1 = np.dot( np.diag(1.0/GAM), self.A_full )
        UV = -(1.0-self.mu)**2 * np.dot( self.A_full.transpose(), UV1) 

        W = (1.0-self.mu) * self.W_full + self.mu*np.eye(self.num_design) + UV


        # 2) rhs_x
        # rhs_x_1 =  1.0/GAM * (v_g + (1.0-self.mu) * 1.0/LAM * v_s) 
        # Scaled Slack
        rhs_x_1 =  1.0/GAM * (v_g + (1.0-self.mu) * self.at_slack * 1.0/LAM * v_s) 
        rhs_x = v_x - np.dot( self.A_full.transpose(), rhs_x_1 )

        p_x = sp.linalg.lu_solve(sp.linalg.lu_factor(W), rhs_x)

        # 3) 
        # rhs_g_1 = v_g + (1.0-self.mu) * 1./LAM * v_s - np.dot( self.A_full, p_x)
        rhs_g_1 = v_g + (1.0-self.mu) * self.at_slack * 1./LAM * v_s - np.dot( self.A_full, p_x)
        p_g = 1.0/GAM * rhs_g_1 
        
        p_s = 1.0/LAM * ( (1.0-self.mu) * self.at_slack * p_g + v_s )     

        out_vec.primal.design.base.data = p_x
        out_vec.primal.slack.base.data = p_s
        out_vec.dual.base.data = p_g 

       

from kona.linalg.matrices.hessian import TotalConstraintJacobian
from kona.linalg.matrices.common import IdentityMatrix
from kona.linalg.matrices.hessian import LagrangianHessian
from kona.linalg.vectors.common import DesignVector, StateVector
from kona.linalg.vectors.common import DualVectorEQ, DualVectorINEQ
from kona.linalg.vectors.composite import CompositeDualVector
from kona.linalg.matrices.common import dRdX, dRdU, dCdX, dCdU
from kona.linalg.matrices.common import dCdX_total_linear

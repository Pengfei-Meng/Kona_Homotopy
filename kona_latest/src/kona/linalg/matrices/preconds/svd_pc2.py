import numpy as np 
import scipy as sp
import pdb
from kona.options import get_opt
from kona.linalg.matrices.preconds import LowRankSVD
from kona.linalg.matrices.hessian.basic import BaseHessian
from kona.linalg.solvers.krylov import FGMRES
from kona.linalg.matrices.common import IdentityMatrix
from kona.linalg.vectors.common import DualVectorEQ, DualVectorINEQ
from kona.linalg.vectors.composite import CompositePrimalVector
from kona.linalg.vectors.composite import ReducedKKTVector
from kona.linalg.matrices.hessian import TotalConstraintJacobian
from kona.linalg.matrices.hessian import LagrangianHessian
from kona.linalg.matrices.hessian import LimitedMemoryBFGS
import matplotlib.pylab as pylt
import scipy.sparse as sps

class SVDPC(BaseHessian):
    """
    On top of svd_pc.py 
    SVD PC is updated to incorporate when mu not 0
    Hessian block is approximated by BFGS 
    SVD approximation is applied on Ag W^{-1} Ag^T 
    Sherman-Morrison is used to solve the reduced inequ multipliers block equation
    See Overleaf Section 6.6 for the formulas 
    """

    def __init__(self, vector_factories, optns={}):    

        super(SVDPC, self).__init__(vector_factories, None)

        self.primal_factory.request_num_vectors(20)
        self.state_factory.request_num_vectors(2)
        if self.eq_factory is not None:
            self.eq_factory.request_num_vectors(3)
        if self.ineq_factory is not None:
            self.ineq_factory.request_num_vectors(5)

        svd_optns = {'lanczos_size': 32}
        self.max_stored = get_opt(optns, 10, 'max_stored')  
        bfgs_optns = {'max_stored': self.max_stored}


        self.svd_AWA_mu = LowRankSVD(
            self.awa_mat_vec_mu, self.ineq_factory, None, None, svd_optns)

        self.Ag = TotalConstraintJacobian( vector_factories )
        self.W_hat = LimitedMemoryBFGS(self.primal_factory, bfgs_optns)  

        self.eye = IdentityMatrix()
        self.eye_precond = self.eye.product
        self._allocated = False

    def awa_mat_vec_mu(self, in_vec, out_vec):
        self.Ag.T.approx.product(in_vec, self.design_work)      
        self.design_work.times(1.0 - self.mu)

        self.W_hat.solve(self.design_work, self.design_work0)

        self.Ag.approx.product(self.design_work0, out_vec)
        out_vec.times(1.0 - self.mu)


    def linearize(self, X, state, adjoint, mu, dLdX_homo, dLdX_homo_oldual, inner_iters):

        assert isinstance(X.primal, CompositePrimalVector), \
            "SVDPC() linearize >> X.primal must be of CompositePrimalVector type!"
        assert isinstance(X.dual, DualVectorINEQ),  \
            "SVDPC() linearize >> X.dual must be of DualVectorINEQ type!"

        if not self._allocated:
            self.design_work0 = self.primal_factory.generate()
            self.design_work = self.primal_factory.generate()
            self.design_work2 = self.primal_factory.generate()
            self.design_work3 = self.primal_factory.generate()

            self.slack_work = self.ineq_factory.generate()
            self.kkt_work = self._generate_kkt()

            self.dual_work1 = self.ineq_factory.generate()
            self.dual_work2 = self.ineq_factory.generate()
            self.dual_work3 = self.ineq_factory.generate()

            # approximate BFGS Hessian
            self.dldx_old = self.primal_factory.generate()
            self.dldx = self.primal_factory.generate()
            self.design_old = self.primal_factory.generate()
            # ------------------------

            self._allocated = True

        # ------------ Extract Data ------------
        self.at_design = X.primal.design
        self.at_slack = X.primal.slack
        self.at_dual_ineq = X.dual
        self.mu = mu

        self.num_design = len(X.primal.design.base.data)
        self.num_ineq = len(X.dual.base.data)

        # self.at_design_data = X.primal.design.base.data
        self.at_slack_data = X.primal.slack.base.data
        if self.eq_factory is not None:
            self.at_dual_eq_data = X.dual.eq.base.data
            self.at_dual_ineq_data = X.dual.ineq.base.data
        else:
            self.at_dual_ineq_data = X.dual.base.data


        self.Ag.linearize(X.primal.design, state)

        # ------------- SVD on Ag W^{-1} Ag^T ---------------------
        # ------------- Hessian LBFGS approximation ---------------
        self.W_hat.norm_init = 1.0  

        if inner_iters == 0:
            self.W_hat.restart()

        else:
            self.design_old.minus(X.primal.design)
            self.design_old.times(-1.0)

            # self.dldx.equals(dLdX_homo_oldual.primal.design)
            # self.dldx_old.minus(self.dldx)
            # self.dldx_old.times(-1.0)

            self.dldx.equals(dLdX_homo.primal.design)
            self.dldx.minus(dLdX_homo_oldual.primal.design)

            self.W_hat.add_correction(self.design_old, self.dldx)

        self.design_old.equals(X.primal.design)


        if self.mu < 0.9:
            self.svd_AWA_mu.linearize()

            self.awa_S = self.svd_AWA_mu.S
            self.awa_U = np.zeros((self.num_ineq, len( self.svd_AWA_mu.U ) ))
            self.awa_V = np.zeros((self.num_ineq, len( self.svd_AWA_mu.V ) ))

            for j in xrange(self.awa_S.shape[0]):
                self.awa_U[:, j] = self.svd_AWA_mu.U[j].base.data
                self.awa_V[:, j] = self.svd_AWA_mu.V[j].base.data


    def solve(self, rhs_vec, pcd_vec):    # BFGS W,   SVD on A W^{-1} A^T, 2nd Type with mu

        if self.mu >= 0.9:
            pcd_vec.equals(rhs_vec)
        else:
            u_x = rhs_vec.primal.design.base.data
            u_s = rhs_vec.primal.slack.base.data
            u_g = rhs_vec.dual.base.data

            # ------------ data used in Sherman-Morrison inverse -------------
            self.lam_aug = -(1.0-self.mu)*self.at_dual_ineq_data + self.mu*np.ones(self.at_dual_ineq_data.shape)
            self.lam_aug_inv = 1./self.lam_aug

            self.W_hat.solve(rhs_vec.primal.design, self.design_work0)


            self.Ag.approx.product(self.design_work0, self.dual_work1)
            self.dual_work1.times(1.0 - self.mu)

            rhs_vg = - u_g - (1.0 - self.mu)*self.lam_aug_inv * u_s + self.dual_work1.base.data

            # ----------------------------------------
            self.A_61 = (1.0 - self.mu)**2 * self.lam_aug_inv * self.at_slack_data + self.mu*np.ones(self.at_dual_ineq_data.shape)

            self.A_61_inv = 1.0/self.A_61
            
            self.Gamma_Nstar = np.dot(self.awa_S, self.awa_V.transpose()) 

            core_mat = np.eye(self.awa_S.shape[0]) + np.dot(self.Gamma_Nstar, np.dot(np.diag(self.A_61_inv),self.awa_U))
            core_inv = np.linalg.inv(core_mat)

            # ------------- multiplying ---------------
            work_1 = self.A_61_inv * rhs_vg
            work_2 = np.dot(self.Gamma_Nstar, work_1)
            work_3 = np.dot(core_inv, work_2)
            work_4 = np.dot(self.awa_U, work_3)
            work_5 = -self.A_61_inv * work_4

            p_g = self.A_61_inv*rhs_vg + work_5
            pcd_vec.dual.base.data = p_g

            # ------------- next look for p_s, p_x 

            self.Ag.T.approx.product(pcd_vec.dual, self.design_work)
            self.design_work.times(self.mu - 1.0)

            self.design_work2.base.data = self.design_work.base.data + u_x 

            self.W_hat.solve(self.design_work2, pcd_vec.primal.design)
            

            Lambda_g_p_s = (1-self.mu) * self.at_slack_data * p_g  + u_s 

            p_s = self.lam_aug_inv * Lambda_g_p_s 
            
            pcd_vec.primal.slack.base.data = p_s

            

    def solve_I(self, rhs_vec, pcd_vec):
        u_x = rhs_vec.primal.design.base.data
        u_s = rhs_vec.primal.slack.base.data
        u_g = rhs_vec.dual.base.data

        self.svd_Ag.approx_fwd_prod(rhs_vec.primal.design, self.dual_work1)

        rhs_vg = - self.at_dual_ineq_data * u_g + u_s + self.at_dual_ineq_data * self.dual_work1.base.data
        
        # ------------ data used in Sherman-Morrison inverse -------------
        self.slack_inv = 1./self.at_slack_data

        self.sigma = - self.slack_inv * self.at_dual_ineq_data    # S_inv * Lambda_g

        self.M_Gamma = np.dot(self.U, self.S)

        core_mat = np.eye(self.S.shape[0]) + np.dot(self.M_Gamma.transpose(),np.dot(np.diag(self.sigma),self.M_Gamma))
        core_inv = np.linalg.inv(core_mat)


        # ------------- multiplying ---------------
        work_1 = - self.slack_inv * rhs_vg
        work_2 = np.dot(self.M_Gamma.transpose(), work_1)
        work_3 = np.dot(core_inv, work_2)
        work_4 = np.dot(self.M_Gamma, work_3)
        work_5 = -self.sigma * work_4

        p_g = - self.slack_inv*rhs_vg + work_5
        pcd_vec.dual.base.data = p_g


        self.svd_Ag.approx_rev_prod(pcd_vec.dual, self.design_work)
        p_x = - self.design_work.base.data + u_x 
        pcd_vec.primal.design.base.data = p_x
        

        Lambda_g_p_s = - u_s - self.at_slack_data * p_g
        Lambda_g_inv = 1./self.at_dual_ineq_data

        p_s = Lambda_g_p_s * Lambda_g_inv
        
        pcd_vec.primal.slack.base.data = p_s
        

    def _generate_kkt(self):
        prim = self.primal_factory.generate()
        slak = self.ineq_factory.generate()        
        dual = self.ineq_factory.generate()
        return ReducedKKTVector(CompositePrimalVector(prim, slak), dual)

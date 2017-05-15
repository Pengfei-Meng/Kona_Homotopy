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

class SVDPC_STRESS(BaseHessian):
    """
    On top of svd_pc2.py 
    Specially designed for Structural problem, with lower, upper bound constraints, stress constraints
    use 1st type of system reduction with mu, apply SVD approximation on AsT_SigS_As
    this way separate bound constraints and stress constraint Jacobian
    It has been separtedly proved effective in  ~/postprocess/solve_kkt.py
    """

    def __init__(self, vector_factories, optns={}):    

        super(SVDPC_STRESS, self).__init__(vector_factories, None)

        self.primal_factory.request_num_vectors(20)
        self.state_factory.request_num_vectors(2)
        if self.eq_factory is not None:
            self.eq_factory.request_num_vectors(3)
        if self.ineq_factory is not None:
            self.ineq_factory.request_num_vectors(5)
        
        svd_optns = {'lanczos_size': get_opt(optns, 40, 'lanczos_size')}  
        bfgs_optns = {'max_stored': get_opt(optns, 10, 'bfgs_max_stored')}

        self.Ag = TotalConstraintJacobian( vector_factories )
        self.svd_AsT_SigS_As_mu = LowRankSVD(
            self.asa_mat_vec_mu, self.primal_factory, None, None, svd_optns)

        self._allocated = False

    def asa_mat_vec_mu(self, in_vec, out_vec):
        self.Ag.product(in_vec, self.dual_work1)
        self.dual_work1.times(1.0 - self.mu)

        work_0 = self.sig_aug_inv_stress * self.dual_work1.base.data[-self.num_design:]

        self.dual_work2.equals(0.0)
        self.dual_work2.base.data[-self.num_design:] = work_0

        self.Ag.T.product(self.dual_work2, out_vec)
        out_vec.times(1.0 - self.mu)


    def linearize(self, X, state, adjoint, mu):

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


        # ------ Data to use --------------
        self.lam_aug = -(1.0-self.mu)*self.at_dual_ineq_data*self.at_slack_data + self.mu*self.at_slack_data
        self.lam_aug_inv = 1./self.lam_aug

        # I''Lam'^{-1} S' - I'
        self.sig_aug = (1.0 - self.mu)**2 *self.at_slack_data* self.lam_aug_inv * self.at_slack_data + self.mu*np.ones(self.at_dual_ineq_data.shape)
        self.sig_aug_inv = 1.0/self.sig_aug

        self.sig_aug_inv_lower = self.sig_aug_inv[: self.num_design]
        self.sig_aug_inv_upper = self.sig_aug_inv[self.num_design : 2*self.num_design ]
        self.sig_aug_inv_stress = self.sig_aug_inv[2*self.num_design : ]

        # ------- LBFGS approx on (1-mu)W + mu*I -------- 
        self.Ag.linearize(X.primal.design, state)

        # -------- SVD on  svd_AsT_SigS_As_mu --------
        # if self.mu < 0.5:
        self.svd_AsT_SigS_As_mu.linearize()
        self.asa_S = self.svd_AsT_SigS_As_mu.S
        self.asa_U = np.zeros((self.num_design, len( self.svd_AsT_SigS_As_mu.U ) ))
        self.asa_V = np.zeros((self.num_design, len( self.svd_AsT_SigS_As_mu.V ) ))

        for j in xrange(self.asa_S.shape[0]):
            self.asa_U[:, j] = self.svd_AsT_SigS_As_mu.U[j].base.data
            self.asa_V[:, j] = self.svd_AsT_SigS_As_mu.V[j].base.data

        self.svd_ASA = np.dot(self.asa_U, np.dot(self.asa_S, self.asa_V.transpose()))


    def solve(self, rhs_vec, pcd_vec):    

        # using scaled slack version,  Lambda_aug, I'' contains Slack component
        u_x = rhs_vec.primal.design.base.data
        u_s = rhs_vec.primal.slack.base.data
        u_g = rhs_vec.dual.base.data         

        # rhs_vx
        rhs_vx_1 = u_g + (1-self.mu) * self.at_slack_data * self.lam_aug_inv * u_s 
        rhs_vx_2 = self.sig_aug_inv * rhs_vx_1

        self.dual_work1.base.data = rhs_vx_2
        self.Ag.T.product(self.dual_work1, self.design_work)

        rhs_vx = u_x + self.design_work.base.data

        # LHS  v_x, svd on whole AsT_SigS_As    # 0.1 for tiny case;  0.01 for small case
        # if self.mu < 1e-6:
        #     fac = 0.005
        # else:
        #     fac = 0.01
        fac = 0.025
        W_approx = fac*np.ones(self.num_design)

        LHS = np.diag( W_approx + self.sig_aug_inv_lower + self.sig_aug_inv_upper ) + self.svd_ASA 
        v_x = sp.linalg.lu_solve(sp.linalg.lu_factor(LHS), rhs_vx) 

        # solve v_g
        self.design_work2.base.data = v_x
        self.Ag.product(self.design_work2, self.dual_work2)
        self.dual_work2.times(1.0 - self.mu)

        rhs_vg = u_g + (1-self.mu)*self.at_slack_data * self.lam_aug_inv * u_s - self.dual_work2.base.data
        v_g = - self.sig_aug_inv * rhs_vg

        # solve v_s 
        v_s = self.lam_aug_inv * ((1-self.mu)*self.at_slack_data * v_g + u_s )

        pcd_vec.primal.design.base.data = v_x
        pcd_vec.primal.slack.base.data = v_s
        pcd_vec.dual.base.data = v_g


    def _generate_kkt(self):
        prim = self.primal_factory.generate()
        slak = self.ineq_factory.generate()        
        dual = self.ineq_factory.generate()
        return ReducedKKTVector(CompositePrimalVector(prim, slak), dual)

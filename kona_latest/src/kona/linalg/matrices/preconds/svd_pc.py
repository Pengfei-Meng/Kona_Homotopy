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

class SVDPC(BaseHessian):

    def __init__(self, vector_factories):    

        super(SVDPC, self).__init__(vector_factories, None)
        
        self.primal_factory.request_num_vectors(5)
        self.state_factory.request_num_vectors(2)
        if self.eq_factory is not None:
            self.eq_factory.request_num_vectors(3)
        if self.ineq_factory is not None:
            self.ineq_factory.request_num_vectors(5)

        self.Ag = TotalConstraintJacobian( vector_factories )

        svd_optns = {'lanczos_size': 5}
        self.svd_Ag = LowRankSVD(
            self.fwd_mat_vec, self.primal_factory, self.rev_mat_vec, self.ineq_factory, svd_optns)

        # krylov solver settings
        krylov_optns = {
            'krylov_file'   : 'svdpc_krylov.dat',
            'subspace_size' : 7,
            'check_res'     : False, 
            'rel_tol'       : 1e-2
        }
        self.krylov = FGMRES(self.primal_factory, krylov_optns,
                             eq_factory=self.eq_factory, ineq_factory=self.ineq_factory)

        self.eye = IdentityMatrix()
        self.eye_precond = self.eye.product
        self._allocated = False

    def fwd_mat_vec(self, in_vec, out_vec):
        self.Ag.approx.product(in_vec, out_vec)

    def rev_mat_vec(self, in_vec, out_vec):
        self.Ag.T.approx.product(in_vec, out_vec)

    def linearize(self, X, state, adjoint, mu=0.0):

        assert isinstance(X.primal, CompositePrimalVector), \
            "SVDPC() linearize >> X.primal must be of CompositePrimalVector type!"
        assert isinstance(X.dual, DualVectorINEQ),  \
            "SVDPC() linearize >> X.dual must be of DualVectorINEQ type!"

        if not self._allocated:
            self.design_work = self.primal_factory.generate()
            self.slack_work = self.ineq_factory.generate()
            self.kkt_work = self._generate_kkt()
            self._allocated = True

        self.at_design = X.primal.design
        self.at_slack = X.primal.slack
        self.at_dual_ineq = X.dual
        self.mu = mu

        self.Ag.linearize(X.primal.design, state)
        self.svd_Ag.linearize()

        # return self.svd_Ag.S

        # # for direct solve part
        # -----------------------
        self.num_design = len(X.primal.design.base.data)
        self.num_ineq = len(X.dual.base.data)


        # self.at_design_data = X.primal.design.base.data
        self.at_slack_data = X.primal.slack.base.data
        if self.eq_factory is not None:
            self.at_dual_eq_data = X.dual.eq.base.data
            self.at_dual_ineq_data = X.dual.ineq.base.data
        else:
            self.at_dual_ineq_data = X.dual.base.data

        self.W_eye = np.eye(self.num_design)
        self.A_full = np.zeros((self.num_ineq, self.num_design))

        # --------- peeling off SVD_Ag  U, V -----------
        self.S = self.svd_Ag.S
        self.U = np.zeros((self.num_ineq, len( self.svd_Ag.U ) ))
        self.V = np.zeros((self.num_design, len( self.svd_Ag.V ) ))

        for j in xrange(self.S.shape[0]):
            self.U[:, j] = self.svd_Ag.U[j].base.data
            self.V[:, j] = self.svd_Ag.V[j].base.data

        self.A_full = np.dot( self.U,  np.dot(self.S, self.V.transpose()) )

        if self.mu == 0.0:
            print 'dual_data: ', self.at_dual_ineq_data
            print 'slack data: ', self.at_slack_data


    def solve(self, rhs_vec, pcd_vec):

        v_x = rhs_vec.primal.design.base.data
        v_s = rhs_vec.primal.slack.base.data
        v_g = rhs_vec.dual.base.data

        rhs_full = np.hstack([v_x, v_s, v_g])

        # # ----------------- The Full KKT Matrix -------------------
        KKT_full = np.vstack([np.hstack([self.W_eye,  np.zeros((self.num_design, self.num_ineq)),  self.A_full.transpose()]), 
                              np.hstack([np.zeros((self.num_ineq, self.num_design)),  -np.diag(self.at_dual_ineq_data), -np.diag(self.at_slack_data)]),
                              np.hstack([self.A_full, -np.eye(self.num_ineq),  np.zeros((self.num_ineq, self.num_ineq))]) ])


        eyes_h = np.hstack([ np.ones(self.num_design), np.ones(self.num_ineq), -np.ones(self.num_ineq) ])    
        homo_I = np.diag(eyes_h)        

        KKT = (1-self.mu)*KKT_full + self.mu*homo_I
        p_full = sp.linalg.lu_solve(sp.linalg.lu_factor(KKT), rhs_full)

        p_x = p_full[:self.num_design]
        p_s = p_full[self.num_design:self.num_design + self.num_ineq]   
        p_g = p_full[self.num_design + self.num_ineq:]

        pcd_vec.primal.design.base.data = p_x
        pcd_vec.primal.slack.base.data = p_s
        pcd_vec.dual.base.data = p_g


    def solve_krylov(self, rhs_vec, pcd_vec): 

        self.krylov.outer_iters = 100000
        self.krylov.inner_iters = 100000
        self.krylov.mu = 100000
        self.krylov.step = 'SVDPC'
        self.krylov.solve(self._mat_vec, rhs_vec, pcd_vec, self.eye_precond)

        # # &&&&&&&&&&&&&& scaled 
        # pcd_vec.primal.slack.times(self.at_slack)
        
    def _mat_vec(self, in_vec, out_vec):
        self._kkt_product(in_vec, out_vec)

        out_vec.times(1. - self.mu)

        self.kkt_work.equals(in_vec)
        self.kkt_work.times(self.mu)

        # # &&&&&&&&&&&&&& scaled 
        # self.kkt_work.primal.slack.times(self.at_slack)
        # ------------------------------

        out_vec.primal.plus(self.kkt_work.primal)
        out_vec.dual.minus(self.kkt_work.dual)

    def _kkt_product(self, in_vec, out_vec):
        # the approximate KKT mat-vec product, with W = Identity, Ag = SVD 
        # expedient coding for now:  only for Graeme's structure problem
        # with only inequality constraints, no equality constraints
        # do some aliasing to make the code cleanier

        assert isinstance(in_vec.primal, CompositePrimalVector), \
            "SVDPC() _kkt_product >> in_vec.primal must be of CompositePrimalVector type!"

        assert isinstance(in_vec.dual, DualVectorINEQ),  \
            "SVDPC() _kkt_product >> in_vec.dual must be of DualVectorINEQ type!"

        in_design = in_vec.primal.design
        in_slack = in_vec.primal.slack
        out_design = out_vec.primal.design
        out_slack = out_vec.primal.slack

        in_dual_ineq = in_vec.dual
        out_dual_ineq = out_vec.dual

        # design block
        out_design.equals(in_design)
        self.svd_Ag.approx_rev_prod(in_dual_ineq, self.design_work)
        out_design.plus(self.design_work)
        
        # slack block
        out_slack.equals(in_slack)
        out_slack.times(self.at_dual_ineq)

        # # &&&&&&&&&&&&&& scaled 
        # out_slack.times(self.at_slack)
        # ---------------------------------
        
        self.slack_work.equals(in_dual_ineq)
        self.slack_work.times(self.at_slack)
        out_slack.plus(self.slack_work)
        out_slack.times(-1.0)

        # # ineq_dual block
        self.svd_Ag.approx_fwd_prod(in_design, out_dual_ineq)
        self.slack_work = in_slack

        # # # &&&&&&&&&&&&&& scaled 
        # self.slack_work.times(self.at_slack)
        # ---------------------------------

        out_dual_ineq.minus(self.slack_work)

        # # ----- adding the theta_r block --------
        # self.slack_work.equals(in_dual_ineq)
        # self.slack_work.times(min(self.singular_vals))
        # out_dual_ineq.minus(self.slack_work)

    def _generate_kkt(self):
        prim = self.primal_factory.generate()
        slak = self.ineq_factory.generate()        
        dual = self.ineq_factory.generate()
        return ReducedKKTVector(CompositePrimalVector(prim, slak), dual)

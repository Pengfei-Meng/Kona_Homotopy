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
import matplotlib.pylab as pylt
import scipy.sparse as sps

class SVDPC(BaseHessian):

    def __init__(self, vector_factories):    

        super(SVDPC, self).__init__(vector_factories, None)
        
        self.primal_factory.request_num_vectors(20)
        self.state_factory.request_num_vectors(2)
        if self.eq_factory is not None:
            self.eq_factory.request_num_vectors(3)
        if self.ineq_factory is not None:
            self.ineq_factory.request_num_vectors(5)

        self.Ag = TotalConstraintJacobian( vector_factories )

        svd_optns = {'lanczos_size': 5}
        self.svd_Ag = LowRankSVD(
            self.fwd_mat_vec, self.primal_factory, self.rev_mat_vec, self.ineq_factory, svd_optns)

        # #---------- Hessian ---------
        # self.W = LagrangianHessian( vector_factories )
        # self.svd_W = LowRankSVD(
        #     self.W_mat_vec, self.primal_factory)
        #----------------------------

        # krylov solver settings
        krylov_optns = {
            'krylov_file'   : 'svdpc_krylov.dat',
            'subspace_size' : 25,
            'check_res'     : False, 
            'rel_tol'       : 1e-2
        }
        self.krylov = FGMRES(self.primal_factory, krylov_optns,
                             eq_factory=self.eq_factory, ineq_factory=self.ineq_factory)

        self.eye = IdentityMatrix()
        self.eye_precond = self.eye.product
        self._allocated = False

    def W_mat_vec(self, in_vec, out_vec):
        self.W.approx.multiply_W(in_vec, out_vec)

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

            self.dual_work1 = self.ineq_factory.generate()

            self._allocated = True

        self.at_design = X.primal.design
        self.at_slack = X.primal.slack
        self.at_dual_ineq = X.dual
        self.mu = mu

        self.Ag.linearize(X.primal.design, state)
        self.svd_Ag.linearize()

        # # # for direct solve part
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

        # # ------------------ Hessian ----------------
        # self.W.linearize(X, state, adjoint)
        # self.svd_W.linearize()
        # self.W_explicit = np.zeros((self.num_design, self.num_design))

        # self.W_S = self.svd_W.S
        # self.W_U = np.zeros((self.num_design, len( self.svd_W.U ) ))
        # self.W_V = np.zeros((self.num_design, len( self.svd_W.V ) ))

        # for j in xrange(self.W_S.shape[0]):
        #     self.W_U[:,j] = self.svd_W.U[j].base.data
        #     self.W_V[:,j] = self.svd_W.V[j].base.data

        # self.W_explicit = np.dot( self.W_U,  np.dot(self.W_S, self.W_V.transpose()) ) + 0.01*self.W_eye

        # self.sigma = - self.at_dual_ineq_data/self.at_slack_data

        # if np.any(abs(self.sigma) > 1000 ):
        #     print 'max dual_data: ', max(abs(self.at_dual_ineq_data))
        #     print 'min dual_data: ', min(abs(self.at_dual_ineq_data))
        #     print 'max slack data: ', max(abs(self.at_slack_data))
        #     print 'min slack data: ', min(abs(self.at_slack_data))
            # print 'A_full condition number ', np.linalg.cond(self.A_full)
            # print 'A_full rank:', np.linalg.matrix_rank(self.A_full)

    def solve(self, rhs_vec, pcd_vec):
        u_x = rhs_vec.primal.design.base.data
        u_s = rhs_vec.primal.slack.base.data
        u_g = rhs_vec.dual.base.data

        self.svd_Ag.approx_fwd_prod(rhs_vec.primal.design, self.dual_work1)

        rhs_vg = - self.at_dual_ineq_data * u_g + u_s + self.at_dual_ineq_data * self.dual_work1.base.data
        
        # ------------ data used in Sherman-Morrison inverse -------------
        self.slack_inv = 1./self.at_slack_data
        # index_slack = abs(self.slack_inv) > 100000 
        # self.slack_inv[ index_slack ] = 0.0

        self.sigma = - self.slack_inv * self.at_dual_ineq_data    # S_inv * Lambda_g

        self.M_Gamma = np.dot(self.U, self.S)

        core_mat = np.eye(self.S.shape[0]) + np.dot(self.M_Gamma.transpose(),np.dot(np.diag(self.sigma),self.M_Gamma))
        core_inv = np.linalg.inv(core_mat)

        # p_full = sp.linalg.lu_solve(sp.linalg.lu_factor(KKT), rhs_full)

        # ------------- multiplying ---------------
        work_1 = - self.slack_inv * rhs_vg
        work_2 = np.dot(self.M_Gamma.transpose(), work_1)
        work_3 = np.dot(core_inv, work_2)
        work_4 = np.dot(self.M_Gamma, work_3)
        work_5 = -self.sigma * work_4

        p_g = - self.slack_inv*rhs_vg + work_5
        # p_g[ index_slack ] = u_g[ index_slack ] 
        pcd_vec.dual.base.data = p_g


        self.svd_Ag.approx_rev_prod(pcd_vec.dual, self.design_work)
        p_x = - self.design_work.base.data + u_x 
        pcd_vec.primal.design.base.data = p_x
        

        Lambda_g_p_s = - u_s - self.at_slack_data * p_g
        Lambda_g_inv = 1./self.at_dual_ineq_data
        # index = abs(Lambda_g_inv) > 100000 
        # Lambda_g_inv[ index ] = 0.0
        p_s = Lambda_g_p_s * Lambda_g_inv
        # p_s[ index ] = u_s[index]
        
        pcd_vec.primal.slack.base.data = p_s
        


            
    def solve_lu(self, rhs_vec, pcd_vec):

        v_x = rhs_vec.primal.design.base.data
        v_s = rhs_vec.primal.slack.base.data
        v_g = rhs_vec.dual.base.data

        rhs_full = np.hstack([v_x, v_s, v_g])

        # # ----------------- The Full KKT Matrix -------------------
        KKT_full = np.vstack([np.hstack([self.W_eye,  np.zeros((self.num_design, self.num_ineq)),  self.A_full.transpose()]), 
                              np.hstack([np.zeros((self.num_ineq, self.num_design)),  -np.diag(self.at_dual_ineq_data), -np.diag(self.at_slack_data)]),
                              np.hstack([ self.A_full, -np.eye(self.num_ineq),  np.zeros((self.num_ineq, self.num_ineq)) ]) ])  
        #            -min(np.diag(self.S))*np.eye(self.num_ineq) 

        # if self.mu < 1e-6:
        #     print 'KKT_full condition number ', np.linalg.cond(KKT_full)
        #     print 'KKT_full rank ', np.linalg.matrix_rank(KKT_full)
        #     print 'A_full condition number ', np.linalg.cond(self.A_full)
        #     print 'A_full rank:', np.linalg.matrix_rank(self.A_full)

            # fig2 = pylt.figure()
            # M = sps.csr_matrix(KKT_full)
            # pylt.spy(M, precision=1e-5, marker='o', markersize=2)
            # pylt.title('sparsity pattern for Jacobian')
            # pylt.show()


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

        # ----- adding the theta_r block --------
        # self.slack_work.equals(in_dual_ineq)
        # self.slack_work.times(min(np.diag(self.svd_Ag.S)))     
        # out_dual_ineq.minus(self.slack_work)

    def _generate_kkt(self):
        prim = self.primal_factory.generate()
        slak = self.ineq_factory.generate()        
        dual = self.ineq_factory.generate()
        return ReducedKKTVector(CompositePrimalVector(prim, slak), dual)

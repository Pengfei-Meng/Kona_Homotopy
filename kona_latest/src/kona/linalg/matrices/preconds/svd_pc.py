import numpy as np 
import scipy as sp
from kona.options import get_opt
from kona.linalg.matrices.preconds import LowRankSVD
from kona.linalg.matrices.hessian.basic import BaseHessian

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

        svd_optns = {'lanczos_size': 10}
        self.svd = LowRankSVD(
            fwd_mat_vec, self.pf, rev_mat_vec, self.df, svd_optns)


    def fwd_mat_vec(self, in_vec, out_vec):
        self.Ag.product(in_vec, out_vec)

    def rev_mat_vec(self, in_vec, out_vec):
        self.Ag.T.product(in_vec, out_vec)

    def linearize(self, X, state, adjoint, mu=0.0):

        self.Ag.linearize(X.primal.design, state)
        self.svd.linearize()

    def solve(self, in_vec, out_vec): 
        






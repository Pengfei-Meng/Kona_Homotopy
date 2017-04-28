import numpy as np 

from kona.linalg.vectors.common import DualVectorEQ, DualVectorINEQ
from kona.linalg.matrices.hessian import TotalConstraintJacobian
from kona.linalg.vectors.composite import CompositePrimalVector
from kona.linalg.matrices.hessian.basic import BaseHessian

class IterSolver(BaseHessian):

    def __init__(self, vector_factories, optns={}):
        super(IterSolver, self).__init__(vector_factories, optns)
        # get references to individual factories
        self.primal_factory.request_num_vectors(50)
        self.state_factory.request_num_vectors(2)
        if self.eq_factory is not None:
            self.eq_factory.request_num_vectors(3)
        if self.ineq_factory is not None:
            self.ineq_factory.request_num_vectors(50)
        
        self.A = TotalConstraintJacobian( vector_factories )
        self._allocated = False

    def linearize(self, X, state, adjoint):

        self.at_slack_data = X.primal.slack.base.data
        if self.eq_factory is not None:
            self.at_dual_eq_data = X.dual.eq.base.data
            self.at_dual_ineq_data = X.dual.ineq.base.data
        else:
            self.at_dual_ineq_data = X.dual.base.data

        if not self._allocated:
            self.design_work = self.primal_factory.generate()
            self.dual_work1 = self.ineq_factory.generate()
            self.dual_work2 = self.ineq_factory.generate()

        self.A.linearize(X.primal.design, state)


    def solve(self, in_vec, out_vec):

        in_design = in_vec.primal.design
        in_slack = in_vec.primal.slack
        in_dual = in_vec.dual

        out_design = out_vec.primal.design
        out_slack = out_vec.primal.slack
        out_dual = out_vec.dual

        # equality: add later

        # step 2 
        out_dual.base.data = - in_slack.base.data * 1.0/self.at_slack_data

        # step 3
        out_design.equals(in_design)

        self.A.T.product(out_dual, self.design_work)
        out_design.minus(self.design_work)

        # step 4
        out_slack.equals(in_dual)

        self.A.product(out_design, self.dual_work1)

        out_slack.minus(self.dual_work1)

























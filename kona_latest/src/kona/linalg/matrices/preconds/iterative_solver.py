import numpy as np 

from kona.linalg.vectors.common import PrimalVector, StateVector, DualVector
from kona.linalg.matrices.hessian import TotalConstraintJacobian
from kona.linalg.vectors.composite import CompositePrimalVector
from kona.linalg.matrices.hessian.basic import BaseHessian

class IterSolver(BaseHessian):

    def __init__(self, vector_factories, optns={}):
        super(IterSolver, self).__init__(vector_factories, optns)
        # get references to individual factories
        self.primal_factory = None
        self.state_factory = None
        self.dual_factory = None
        for factory in self.vec_fac:
            if factory._vec_type is PrimalVector:
                self.primal_factory = factory
            elif factory._vec_type is StateVector:
                self.state_factory = factory
            elif factory._vec_type is DualVector:
                self.dual_factory = factory
        
        self.A = TotalConstraintJacobian( vector_factories )
        self._allocated = False

    def linearize(self, X, state):

        if isinstance(X._primal, CompositePrimalVector):
            self.at_design = X._primal._design
            self.at_slack = X._primal._slack
        else:
            raise ValueError('X._primal must be a Composite Primal Vector \
                for using Iterative Solver! ')

        if not self._allocated:
            self.design_work = self.primal_factory.generate()
            self.dual_work1 = self.dual_factory.generate()
            self.dual_equ = self.dual_factory.generate()
            self.dual_inequ = self.dual_factory.generate()
            self.dual_work2 = self.dual_factory.generate()


            self.slack_work = self.dual_factory.generate()
            self.slack_term_inv = self.dual_factory.generate()

        self.A.linearize(X, state)

        if self.at_slack is not None:
            self.slack_work.equals(self.at_slack)
            self.slack_work.times(-1.0)

            self.slack_term_inv.exp(self.slack_work)
            self.slack_term_inv.times(-1.)
            self.slack_term_inv.restrict()

    def solve(self, in_vec, out_vec):

        in_design = in_vec._primal._design
        in_slack = in_vec._primal._slack
        in_dual = in_vec._dual

        out_design = out_vec._primal._design
        out_slack = out_vec._primal._slack
        out_dual = out_vec._dual


        # step 1
        self.dual_equ.equals(in_dual)
        self.dual_equ.restrict()
        self.dual_equ.minus(in_dual)
        self.dual_equ.times(-1.0)

        out_dual.equals(self.dual_equ)

        # step 2 
        self.dual_inequ.equals(in_slack)
        self.dual_inequ.times(self.slack_term_inv)
        out_dual.plus(self.dual_inequ)

        # step 3
        out_design.equals(in_design)

        self.A.T.product(out_dual, self.design_work)
        out_design.minus(self.design_work)

        # step 4
        out_slack.equals(in_dual)
        # out_slack.restrict()

        self.A.product(out_design, self.dual_work1)
        # self.dual_work1.restrict()

        out_slack.minus(self.dual_work1)
        out_slack.times(self.slack_term_inv)

























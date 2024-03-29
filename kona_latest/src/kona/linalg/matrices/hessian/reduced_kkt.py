from kona.linalg.matrices.hessian.basic import BaseHessian

class ReducedKKTMatrix(BaseHessian):
    """
    Reduced approximation of the KKT matrix using a 2nd order adjoint
    formulation.

    The KKT system is defined as:

    .. math::
        \\begin{bmatrix}
        \\nabla_x^2 \\mathcal{L} && 0 && \\nabla_x c_{eq}^T && \\nabla_x c_{ineq}^T \\\\
        0 && \\Sigma && 0 && I \\\\
        \\nabla_x c_{eq} && 0 && 0 && 0 \\\\
        \\nabla_x c_{ineq} && I && 0 && 0
        \\end{bmatrix}
        \\begin{bmatrix}
        \\Delta x \\\\
        \\Delta s \\\\
        \\Delta \\lambda_{eq} \\\\
        \\Delta \\lambda_{ineq}
        \\end{bmatrix}
        =
        \\begin{bmatrix}
        -\\nabla_x \\mathcal{f} - \\lambda_{eq}^T \\nabla_x c_{eq} -\\lambda_{eq}^T \\nabla_x c_{eq} \\\\
        \\lambda^T \\mu * S^{-1}e \\\\
        - c_{eq} \\\\
        - c_{ineq} + s
        \\end{bmatrix}

    where :math:`\\mathcal{L}` is the Lagrangian defined as:

    .. math::
        \\mathcal{L}(x, u(x), \lambda) = F(x, u(x)) +
        \\lambda_{eq}^T c_{eq}(x, u(x)) +
        \\lambda_{ineq}^T \\left[c_{ineq}(x, u(x)) - s\\right] +
        \\frac{1}{2}\\mu\\sum_{i=1}^{n_{ineq}}ln(s_i)

    Inequality constrained are handled via the slack variables :math:`s` and
    the logarithmic barrier term enforcing non-negativity.

    .. note::

        More information on this 2nd order adjoint formulation can be found
        `in this paper <http://arc.aiaa.org/doi/abs/10.2514/6.2015-1945>`.

    Attributes
    ----------
    product_fac : float
    product_tol : float
    lamb : float
    scale : float
    grad_scale : float
    ceq_scale : float
    dynamic_tol : boolean
    krylov : KrylovSolver
        A krylov solver object used to solve the system defined by this matrix.
    dRdX, dRdU, dCdX, dCdU : KonaMatrix
        Various abstract jacobians used in calculating the mat-vec product.
    """
    def __init__(self, vector_factories, optns=None):
        super(ReducedKKTMatrix, self).__init__(vector_factories, optns)

        # read reduced options
        self.product_fac = get_opt(self.optns, 0.001, 'product_fac')
        self.product_tol = 1.0
        self.lamb = get_opt(self.optns, 0.0, 'lambda')
        self.scale = get_opt(self.optns, 1.0, 'scale')
        self.grad_scale = get_opt(self.optns, 1.0, 'grad_scale')
        self.feas_scale = get_opt(self.optns, 1.0, 'feas_scale')
        self.dynamic_tol = get_opt(self.optns, False, 'dynamic_tol')
        self.symmetric = get_opt(self.optns, False, 'symmetric')

        # set empty solver handle
        self.krylov = None

        # reset the linearization flag
        self._allocated = False

        # request vector memory for future allocation
        self.primal_factory.request_num_vectors(4)
        self.state_factory.request_num_vectors(10)
        if self.eq_factory is not None:
            self.eq_factory.request_num_vectors(3)
        if self.ineq_factory is not None:
            self.ineq_factory.request_num_vectors(4)

        # initialize abtract jacobians
        self.dRdX = dRdX()
        self.dRdU = dRdU()
        self.dCdX = dCdX()
        self.dCdU = dCdU()
        self.dCINdX = dCINdX()

        self._approx = False

    @property
    def approx(self):
        self._approx = True
        return self

    def _linear_solve(self, rhs_vec, solution, rel_tol=1e-8):
        self.dRdU.linearize(self.at_design, self.at_state)
        if self._approx:
            self.dRdU.precond(rhs_vec, solution)
        else:
            self.dRdU.solve(rhs_vec, solution, rel_tol=rel_tol)

    def _adjoint_solve(self, rhs_vec, solution, rel_tol=1e-8):
        self.dRdU.linearize(self.at_design, self.at_state)
        if self._approx:
            self.dRdU.T.precond(rhs_vec, solution)
        else:
            self.dRdU.T.solve(rhs_vec, solution, rel_tol=rel_tol)

    def set_krylov_solver(self, krylov_solver):
        if isinstance(krylov_solver, KrylovSolver):
            self.krylov = krylov_solver
        else:
            raise TypeError('Solver is not a valid KrylovSolver')

    def linearize(self, at_kkt, at_state, at_adjoint,
                  obj_scale=1.0, cnstr_scale=1.0):
        """
        Linearize the KKT matrix at the given KKT, state, adjoint and barrier
        point. This method does not perform any factorizations or matrix
        operations.

        Parameters
        ----------
        at_kkt : ReducedKKTVector
            KKT vector at which the product is evaluated
        at_state : StateVector
            State point at which the product is evaluated.
        at_adjoint : StateVector
            1st order adjoint variables at which the product is evaluated.
        obj_scale : float, optional
            Factor by which the objective component of the product is scaled.
        cnstr_scale : float, optional
            Factor by which the constraint component of the product is scaled.
        """
        # if this is the first ever linearization...
        if not self._allocated:

            # generate state vectors
            self.adjoint_res = self.state_factory.generate()
            self.w_adj = self.state_factory.generate()
            self.lambda_adj = self.state_factory.generate()
            self.state_work = []
            for i in xrange(4):
                self.state_work.append(self.state_factory.generate())

            # generate primal vectors
            self.pert_design = self.primal_factory.generate()
            self.reduced_grad = self.primal_factory.generate()
            self.primal_work = self.primal_factory.generate()

            # generate dual vectors
            # if isinstance(at_kkt.dual, CompositeDualVector):
            if self.ineq_factory is not None:
                if self.eq_factory is not None:
                    dual_eq = self.eq_factory.generate()
                    dual_ineq = self.ineq_factory.generate()
                    self.dual_work = CompositeDualVector(dual_eq, dual_ineq)
                else:
                    self.dual_work = self.ineq_factory.generate()
            else:
                if self.eq_factory is not None:
                    self.dual_work = self.eq_factory.generate()


            self.slack_block = None
            if self.ineq_factory is not None:
                self.slack_block = self.ineq_factory.generate()

            if self.eq_factory is not None:
                self.reduced_work_eq = self.eq_factory.generate()  

            if self.ineq_factory is not None:  
                self.reduced_work_ineq = self.ineq_factory.generate()
                self.reduced_work_ineq2 = self.ineq_factory.generate()
                self.in_dual_ineq = self.ineq_factory.generate()
            self.reduced_work_design = self.primal_factory.generate()

            self._allocated = True

        # store the linearization point
        self.at_dual = at_kkt.dual
        if isinstance(at_kkt.primal, CompositePrimalVector):
            self.at_design = at_kkt.primal.design
            self.at_slack = at_kkt.primal.slack
            if isinstance(self.at_dual, CompositeDualVector):
                self.at_dual_ineq = self.at_dual.ineq
            else:
                self.at_dual_ineq = self.at_dual
        else:
            self.at_design = at_kkt.primal
            self.at_slack = None
            self.at_dual_ineq = None
        self.design_norm = self.at_design.norm2
        self.at_state = at_state
        self.state_norm = self.at_state.norm2
        self.at_adjoint = at_adjoint
        self.at_dual = at_kkt.dual

        # store scales
        self.obj_scale = obj_scale
        self.cnstr_scale = cnstr_scale

        # pre compute the slack block
        # if self.slack_block is not None:
            # self.slack_block.equals(self.at_slack)
            # self.slack_block.pow(-1.)
            # self.slack_block.times(self.at_dual_ineq)

        # compute adjoint residual at the linearization
        self.dual_work.equals_constraints(self.at_design, self.at_state)
        self.adjoint_res.equals_objective_partial(
            self.at_design, self.at_state, scale=self.obj_scale)
        self.dRdU.linearize(self.at_design, self.at_state)
        self.dRdU.T.product(self.at_adjoint, self.state_work[0])
        self.adjoint_res.plus(self.state_work[0])
        self.dCdU.linearize(self.at_design, self.at_state)
        self.dCdU.T.product(self.at_dual, self.state_work[0], state_work=self.state_work[3])
        self.state_work[0].times(self.cnstr_scale)
        self.adjoint_res.plus(self.state_work[0])

        # compute reduced gradient at the linearization
        self.reduced_grad.equals_objective_partial(
            self.at_design, self.at_state, scale=self.obj_scale)
        self.dRdX.linearize(self.at_design, self.at_state)
        self.dRdX.T.product(self.at_adjoint, self.primal_work)
        self.reduced_grad.plus(self.primal_work)
        self.dCdX.linearize(self.at_design, self.at_state)
        self.dCdX.T.product(self.at_dual, self.primal_work)
        self.primal_work.times(self.cnstr_scale)
        self.reduced_grad.plus(self.primal_work)

        self.dCINdX.linearize(self.at_design, self.at_state)


    def product(self, in_vec, out_vec):
        """
        Matrix-vector product for the reduced KKT system.

        Parameters
        ----------
        in_vec : ReducedKKTVector
            Vector to be multiplied with the KKT matrix.
        out_vec : ReducedKKTVector
            Result of the operation.
        """
        # type check given vectors
        if not isinstance(in_vec, ReducedKKTVector):
            raise TypeError('Multiplying vector is not a ReducedKKTVector')
        if not isinstance(out_vec, ReducedKKTVector):
            raise TypeError('Result vector is not a ReducedKKTVector')

        # clear output vector
        out_vec.equals(0.0)

        # do some aliasing to make the code cleanier
        in_dual = in_vec.dual
        out_dual = out_vec.dual
        if isinstance(in_vec.primal, CompositePrimalVector):
            if self.at_slack is None:
                raise TypeError('No slack variables defined!')
            in_design = in_vec.primal.design
            in_slack = in_vec.primal.slack
            out_design = out_vec.primal.design
            out_slack = out_vec.primal.slack
            if isinstance(in_dual, CompositeDualVector):
                in_dual_ineq = in_dual.ineq
                out_dual_ineq = out_dual.ineq
            else:
                in_dual_ineq = in_dual
                out_dual_ineq = out_dual
        else:
            in_design = in_vec.primal
            in_slack = None
            out_design = out_vec.primal
            out_slack = None
            in_dual_ineq = None
            out_dual_ineq = None

        # calculate appropriate FD perturbation for design
        epsilon_fd = calc_epsilon(self.design_norm, in_design.norm2)

        # assemble RHS for first adjoint system
        self.dRdX.linearize(self.at_design, self.at_state)
        self.dRdX.product(in_design, self.state_work[0])
        self.state_work[0].times(-1.0)

        # perform the adjoint solution
        self.w_adj.equals(0.0)
        rel_tol = self.product_tol * \
            self.product_fac/max(self.state_work[0].norm2, EPS)
        # rel_tol = 1e-12
        # if self.w_adj._memory.solver.get_rank() == 0:
        #     print 'first second adjoint solve in forming KKT mat_vec product'

        self._linear_solve(self.state_work[0], self.w_adj, rel_tol=rel_tol)
        # if self.w_adj._memory.solver.get_rank() == 0:
        #     print 'first second adjoint solve completed'
        # find the adjoint perturbation by solving the linearized dual equation
        self.pert_design.equals_ax_p_by(
            1.0, self.at_design, epsilon_fd, in_design)
        self.state_work[2].equals_ax_p_by(
            1.0, self.at_state, epsilon_fd, self.w_adj)

        # first part of LHS: evaluate the adjoint equation residual at
        # perturbed design and state
        self.state_work[0].equals_objective_partial(
            self.pert_design, self.state_work[2], scale=self.obj_scale)
        pert_state = self.state_work[2] # aliasing for readability
        self.dRdU.linearize(self.pert_design, pert_state)
        self.dRdU.T.product(self.at_adjoint, self.state_work[1])
        self.state_work[0].plus(self.state_work[1])
        self.dCdU.linearize(self.pert_design, pert_state)
        self.dCdU.T.product(self.at_dual, self.state_work[1], state_work=self.state_work[3])
        self.state_work[1].times(self.cnstr_scale)
        self.state_work[0].plus(self.state_work[1])

        # at this point state_work[0] should contain the perturbed adjoint
        # residual, so take difference with unperturbed adjoint residual
        self.state_work[0].minus(self.adjoint_res)
        self.state_work[0].divide_by(epsilon_fd)

        # multiply by -1 to move to RHS
        self.state_work[0].times(-1.0)

        # second part of LHS: (dC/dU) * in_vec.dual
        self.dCdU.linearize(self.at_design, self.at_state)
        self.dCdU.T.product(in_dual, self.state_work[1], state_work=self.state_work[3])
        self.state_work[1].times(self.cnstr_scale)

        # assemble final RHS
        self.state_work[0].minus(self.state_work[1])

        # perform the adjoint solution
        self.lambda_adj.equals(0.0)
        rel_tol = self.product_tol * \
            self.product_fac/max(self.state_work[0].norm2, EPS)
        # rel_tol = 1e-12
        # if self.lambda_adj._memory.solver.get_rank() == 0:
        #     print 'second Second adjoint solve in forming KKT mat_vec product'
        self._adjoint_solve(
            self.state_work[0], self.lambda_adj, rel_tol=rel_tol)

        # if self.lambda_adj._memory.solver.get_rank() == 0:
        #     print 'second Second adjoint solve completed'

        # evaluate first order optimality conditions at perturbed design, state
        # and adjoint:
        # g = df/dX + lag_mult*dC/dX + (adjoint + eps_fd*lambda_adj)*dR/dX
        self.state_work[1].equals_ax_p_by(
            1.0, self.at_adjoint, epsilon_fd, self.lambda_adj)
        pert_adjoint = self.state_work[1] # aliasing for readability
        out_design.equals_objective_partial(
            self.pert_design, pert_state, scale=self.obj_scale)
        self.dRdX.linearize(self.pert_design, pert_state)
        self.dRdX.T.product(pert_adjoint, self.primal_work)
        out_design.plus(self.primal_work)
        self.dCdX.linearize(self.pert_design, pert_state)
        self.dCdX.T.product(self.at_dual, self.primal_work)
        self.primal_work.times(self.cnstr_scale)
        out_design.plus(self.primal_work)

        # take difference with unperturbed conditions
        out_design.times(self.grad_scale)
        out_design.minus(self.reduced_grad)
        out_design.divide_by(epsilon_fd)

        # the dual part needs no FD
        self.dCdX.linearize(self.at_design, self.at_state)
        self.dCdX.T.product(in_dual, self.primal_work)
        self.primal_work.times(self.cnstr_scale)
        out_design.plus(self.primal_work)

        # evaluate dual part of product:
        # C = dC/dX*in_vec + dC/dU*w_adj
        self.dCdX.linearize(self.at_design, self.at_state)
        self.dCdX.product(in_design, out_vec.dual)
        out_vec.dual.times(self.cnstr_scale)
        self.dCdU.linearize(self.at_design, self.at_state)
        self.dCdU.product(self.w_adj, self.dual_work)
        self.dual_work.times(self.cnstr_scale)
        out_dual.plus(self.dual_work)
        out_dual.times(self.feas_scale)

        # add globalization if necessary
        if self.lamb > EPS:
            out_design.equals_ax_p_by(
                1., out_design, self.lamb*self.scale, in_design)

        if self.symmetric is False:   # Unsymmetric case
            # # add the slack term to the dual component
            if in_slack is not None:
                # set slack output
                # out_slack = -diag(at_dual_ineq) * in_slack - diag(at_slack) * in_dual_ineq
                out_slack.equals(in_slack)
                out_slack.times(self.at_dual_ineq)
                self.slack_block.equals(in_dual_ineq)
                self.slack_block.times(self.at_slack)
                out_slack.plus(self.slack_block)
                out_slack.times(-1.0)
                # add the slack contribution to dual component
                # out_dual_ineq -= in_slack
                out_dual_ineq.minus(in_slack)

        # reset the approx and transpose flags at the end
        self._approx = False

        # testing on scaled Slack block KKT product. If not work, just use the above commented block
        if self.symmetric is True: 
            if in_slack is not None:
                # set slack output
                # out_slack = -diag(at_dual_ineq.*at_slack) * in_slack - diag(at_slack) * in_dual_ineq
                out_slack.equals(in_slack)
                out_slack.times(self.at_dual_ineq)
                out_slack.times(self.at_slack)

                self.slack_block.equals(in_dual_ineq)
                self.slack_block.times(self.at_slack)
                out_slack.plus(self.slack_block)
                out_slack.times(-1.0)

                self.slack_block.equals(in_slack)
                self.slack_block.times(self.at_slack)
                out_dual_ineq.minus(self.slack_block)     

    def product_reduced(self, in_vec, out_vec):
        """
        Matrix-vector product for the slack, ineq_dual eliminated reduced KKT system .

        Parameters
        ----------
        in_vec : CompositeVector, with primal: DesignVector, dual: DualVectorEQ
            Vector to be multiplied with the KKT matrix.
        out_vec : CompositeVector, with primal: DesignVector, dual: DualVectorEQ
            Result of the operation.
        """       
        assert isinstance(in_vec.primal, DesignVector), \
            "ReducedKKTMatrix() product_reduced >> in_vec.primal must be of DesignVector type!"

        assert isinstance(in_vec.dual, DualVectorEQ), \
            "ReducedKKTMatrix() product_reduced >> in_vec.dual must be of DualVectorEQ type!"

        assert isinstance(out_vec.primal, DesignVector), \
            "ReducedKKTMatrix() product_reduced >> out_vec.primal must be of DesignVector type!"

        assert isinstance(out_vec.dual, DualVectorEQ), \
            "ReducedKKTMatrix() product_reduced >> out_vec.dual must be of DualVectorEQ type!"     

        # clear output vector
        out_vec.equals(0.0)

        # do some aliasing to make the code cleanier
        in_design = in_vec.primal
        out_design = out_vec.primal
        in_dual_eq = in_vec.dual
        out_dual = out_vec.dual

        # in_dual_ineq ---------------------
        self.dCINdX.linearize(self.at_design, self.at_state)   
        self.dCINdX.product(in_design, self.reduced_work_ineq)
        self.reduced_work_ineq2.equals_constraints(self.at_design, self.at_state)  
        self.reduced_work_ineq2.plus(self.reduced_work_ineq)
        self.reduced_work_ineq2.times(-1.0)

        self.reduced_work_ineq.equals(0.0)
        self.reduced_work_ineq.equals(self.at_slack)
        self.reduced_work_ineq.pow(-1.0)
        self.reduced_work_ineq.times(self.at_dual_ineq)
        self.reduced_work_ineq2.times(self.reduced_work_ineq)

        self.in_dual_ineq.equals(self.reduced_work_ineq2)

        # self.dual_work is very important here, in_dual_eq, in_dual_ineq 
        self.dual_work.eq.equals(in_dual_eq)
        self.dual_work.ineq.equals(self.in_dual_ineq)
        # ------------------------------

        # calculate appropriate FD perturbation for design
        epsilon_fd = calc_epsilon(self.design_norm, in_design.norm2)

        # assemble RHS for first adjoint system
        self.dRdX.linearize(self.at_design, self.at_state)
        self.dRdX.product(in_design, self.state_work[0])
        self.state_work[0].times(-1.0)

        # perform the adjoint solution
        self.w_adj.equals(0.0)
        rel_tol = self.product_tol * \
            self.product_fac/max(self.state_work[0].norm2, EPS)
        # rel_tol = 1e-12
        self._linear_solve(self.state_work[0], self.w_adj, rel_tol=rel_tol)

        # find the adjoint perturbation by solving the linearized dual equation
        self.pert_design.equals_ax_p_by(
            1.0, self.at_design, epsilon_fd, in_design)
        self.state_work[2].equals_ax_p_by(
            1.0, self.at_state, epsilon_fd, self.w_adj)

        # first part of LHS: evaluate the adjoint equation residual at
        # perturbed design and state
        self.state_work[0].equals_objective_partial(
            self.pert_design, self.state_work[2], scale=self.obj_scale)
        pert_state = self.state_work[2] # aliasing for readability
        self.dRdU.linearize(self.pert_design, pert_state)
        self.dRdU.T.product(self.at_adjoint, self.state_work[1])
        self.state_work[0].plus(self.state_work[1])
        self.dCdU.linearize(self.pert_design, pert_state)
        self.dCdU.T.product(self.at_dual, self.state_work[1], state_work=self.state_work[3])
        self.state_work[1].times(self.cnstr_scale)
        self.state_work[0].plus(self.state_work[1])

        # at this point state_work[0] should contain the perturbed adjoint
        # residual, so take difference with unperturbed adjoint residual
        self.state_work[0].minus(self.adjoint_res)
        self.state_work[0].divide_by(epsilon_fd)

        # multiply by -1 to move to RHS
        self.state_work[0].times(-1.0)

        # second part of LHS: (dC/dU) * in_vec.dual   # self.dual_work contains eq, ineq in_dual
        self.dCdU.linearize(self.at_design, self.at_state)
        self.dCdU.T.product(self.dual_work, self.state_work[1], state_work=self.state_work[3])
        self.state_work[1].times(self.cnstr_scale)

        # assemble final RHS
        self.state_work[0].minus(self.state_work[1])

        # perform the adjoint solution
        self.lambda_adj.equals(0.0)
        rel_tol = self.product_tol * \
            self.product_fac/max(self.state_work[0].norm2, EPS)
        # rel_tol = 1e-12
        self._adjoint_solve(
            self.state_work[0], self.lambda_adj, rel_tol=rel_tol)

        # evaluate first order optimality conditions at perturbed design, state
        # and adjoint:
        # g = df/dX + lag_mult*dC/dX + (adjoint + eps_fd*lambda_adj)*dR/dX
        self.state_work[1].equals_ax_p_by(
            1.0, self.at_adjoint, epsilon_fd, self.lambda_adj)
        pert_adjoint = self.state_work[1] # aliasing for readability
        out_design.equals_objective_partial(
            self.pert_design, pert_state, scale=self.obj_scale)
        self.dRdX.linearize(self.pert_design, pert_state)
        self.dRdX.T.product(pert_adjoint, self.primal_work)
        out_design.plus(self.primal_work)
        self.dCdX.linearize(self.pert_design, pert_state)
        self.dCdX.T.product(self.at_dual, self.primal_work)
        self.primal_work.times(self.cnstr_scale)
        out_design.plus(self.primal_work)

        # take difference with unperturbed conditions
        out_design.times(self.grad_scale)
        out_design.minus(self.reduced_grad)
        out_design.divide_by(epsilon_fd)

        # the dual part needs no FD
        self.dCdX.linearize(self.at_design, self.at_state)
        self.dCdX.T.product(in_dual_eq, self.primal_work)
        self.primal_work.times(self.cnstr_scale)
        out_design.plus(self.primal_work)

        # evaluate dual part of product:
        # C = dC/dX*in_vec + dC/dU*w_adj
        self.dCdX.linearize(self.at_design, self.at_state)
        self.dCdX.product(in_design, out_dual)
        out_dual.times(self.cnstr_scale)
        self.dCdU.linearize(self.at_design, self.at_state)
        # self.dCdU.product(self.w_adj, self.dual_work)
        # self.dual_work.times(self.cnstr_scale)
        # out_dual.plus(self.dual_work)
        # out_dual.times(self.feas_scale)

        #------------- this part is modified ------------
        self.dCdU.product(self.w_adj, self.reduced_work_eq)
        self.reduced_work_eq.times(self.cnstr_scale)
        out_dual.plus(self.reduced_work_eq)
        out_dual.times(self.feas_scale)
        # print 'reduced_kkt 1st out_design.norm2', out_design.norm2
        if out_dual._memory.solver.get_rank() == 0:
            print 'min(abs(self.at_slack.base.data))', min(abs(self.at_slack.base.data))

        # add globalization if necessary
        if self.lamb > EPS:
            out_design.equals_ax_p_by(
                1., out_design, self.lamb*self.scale, in_design)

        ###  add the second block
        self.dCINdX.product(in_design, self.reduced_work_ineq)

        self.reduced_work_ineq2.equals(self.at_slack)
        self.reduced_work_ineq2.pow(-1.0)
        self.reduced_work_ineq2.times(self.at_dual_ineq)

        if out_dual._memory.solver.get_rank() == 0:
            print 'max(abs(S^-1 * Lam_g))', max(abs(self.reduced_work_ineq2.base.data))
            # print 'self.at_dual_ineq.base.data', self.at_dual_ineq.base.data
            # print 'self.at_slack.base.data', self.at_slack.base.data

        self.reduced_work_ineq.times(self.reduced_work_ineq2)


        self.dCINdX.T.product(self.reduced_work_ineq, self.reduced_work_design)

        out_design.minus(self.reduced_work_design)

        if out_dual._memory.solver.get_rank() == 0:
            print 'reduced_kkt 2nd out_design.norm2', out_design.norm2
        # print 'reduced_kkt out_dual.norm2', out_dual.norm2


# imports here to prevent circular errors
from numbers import Number
from kona.options import get_opt
from kona.linalg.vectors.common import DesignVector, StateVector
from kona.linalg.vectors.common import DualVectorEQ, DualVectorINEQ
from kona.linalg.vectors.composite import ReducedKKTVector
from kona.linalg.vectors.composite import CompositePrimalVector
from kona.linalg.vectors.composite import CompositeDualVector
from kona.linalg.matrices.common import dRdX, dRdU, dCdX, dCdU, dCINdX
from kona.linalg.solvers.krylov.basic import KrylovSolver
from kona.linalg.solvers.util import calc_epsilon, EPS
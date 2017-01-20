import numpy as np
import pickle

from kona.user import BaseVector, BaseAllocator, UserSolver

from fstopo.linalg import MGVec, fgmres_solve
from fstopo.material import *
import fstopo.sparse
import fstopo.kona

class FSTopoConstraints(BaseVector):

    def __init__(self, design_vec1, design_vec2, stress_vec):
        self.x_lower = design_vec1
        self.x_upper = design_vec2
        self.stress = stress_vec
        self.normalized = False

    def plus(self, vector):
        self.x_lower.x += vector.x_lower.x
        self.x_upper.x += vector.x_upper.x
        self.stress.x += vector.stress.x

    def times_scalar(self, value):
        self.x_lower.x *= value
        self.x_upper.x *= value
        self.stress.x *= value

    def times_vector(self, vector):
        self.x_lower.x = self.x_lower.x*vector.x_lower.x
        self.x_upper.x = self.x_upper.x*vector.x_upper.x
        self.stress.x = self.stress.x*vector.stress.x

    def equals_value(self, value):
        self.x_lower.x[:] = value
        self.x_upper.x[:] = value
        self.stress.x[:] = value

    def equals_vector(self, vector):
        self.x_lower.copy(vector.x_lower)
        self.x_upper.copy(vector.x_upper)
        self.stress.copy(vector.stress)

    def equals_ax_p_by(self, a, x, b, y):
        self.x_lower.x = a*x.x_lower.x + b*y.x_lower.x
        self.x_upper.x = a*x.x_upper.x + b*y.x_upper.x
        self.stress.x = a*x.stress.x + b*y.stress.x

    def exp(self, vector):
        self.x_lower.x = np.exp(vector.x_lower.x)
        self.x_upper.x = np.exp(vector.x_upper.x)
        self.stress.x = np.exp(vector.stress.x)

    def log(self, vector):
        self.x_lower.x = np.log(vector.x_lower.x)
        self.x_upper.x = np.log(vector.x_upper.x)
        self.stress.x = np.log(vector.stress.x)

    def pow(self, power):
        self.x_lower.x = self.x_lower.x**power
        self.x_upper.x = self.x_upper.x**power
        self.stress.x = self.stress.x**power

    def inner(self, vector):
        if self.normalized:
            length = \
                self.x_lower.x.shape[0]*self.x_lower.x.shape[1] + \
                self.x_upper.x.shape[0]*self.x_upper.x.shape[1] + \
                self.stress.x.shape[0]*self.stress.x.shape[1]
            return \
                (1./length) * (
                    self.x_lower.dot(vector.x_lower) +
                    self.x_upper.dot(vector.x_upper) +
                    self.stress.dot(vector.stress))
        else:
            return \
                self.x_lower.dot(vector.x_lower) + \
                self.x_upper.dot(vector.x_upper) + \
                self.stress.dot(vector.stress)

    @property
    def infty(self):
        return max(
            self.x_lower.infty(), self.x_upper.infty(), self.stress.infty())

class FSTopoVector(BaseVector):

    def __init__(self, data_vec):
        self.data = data_vec
        self.normalized = False

    def plus(self, vector):
        self.data.x += vector.data.x

    def times_scalar(self, value):
        self.data.x *= value

    def times_vector(self, vector):
        self.data.x = self.data.x*vector.data.x

    def equals_value(self, value):
        self.data.x[:] = value

    def equals_vector(self, vector):
        self.data.copy(vector.data)

    def equals_ax_p_by(self, a, x, b, y):
        self.data.x = a*x.data.x + b*y.data.x

    def exp(self, vector):
        self.data.x = np.exp(vector.data.x)

    def log(self, vector):
        self.data.x = np.log(vector.data.x)

    def pow(self, power):
        self.data.x = self.data.x**power

    def inner(self, vector):
        if self.normalized:
            length = self.data.x.shape[0]*self.data.x.shape[1]
            return \
                (1./length)*self.data.dot(vector.data)
        else:
            return self.data.dot(vector.data)

    @property
    def infty(self):
        return self.data.infty()


class FSTopoAllocator(BaseAllocator):

    def __init__(self, wrapper, problem):
        self.wrapper = wrapper
        self.prob = problem

    def alloc_primal(self, count):
        out = []
        for i in xrange(count):
            out.append(FSTopoVector(self.prob.createDesignVec()))
        return out

    def alloc_state(self, count):
        out = []
        for i in xrange(count):
            out.append(FSTopoVector(self.prob.createSolutionVec()))
        return out

    def alloc_dual(self, count):
        out = []
        for i in xrange(count):
            out.append(
                FSTopoConstraints(
                    self.prob.createDesignVec(),
                    self.prob.createDesignVec(),
                    MGVec(n=self.wrapper.num_stress, nb=1)
                )
            )
        return out

class FSTopoSolver(UserSolver):

    def __init__(self, prob, force, x, lower=0.1, upper=10.0,
                 num_aggr=0, ks_rho=50., cnstr_scale=True, prefix='.'):
        # General setup
        self.prob = prob
        self.force = force
        self.x_min = lower
        self.x_max = upper
        self.init_design_vector = x
        self.prefix = prefix

        # KS aggregation setup
        self.num_aggr = num_aggr
        self.ks_rho = ks_rho
        if self.num_aggr == 0:
            self.num_stress = self.prob.conn.shape[0]
        else:
            self.num_stress = self.num_aggr
        self.cnstr_scale = cnstr_scale

        # internal work vectors
        self.at_state = self.prob.createSolutionVec()
        self.state_work = self.prob.createSolutionVec()
        self.dual_work = FSTopoConstraints(
            self.prob.createDesignVec(),
            self.prob.createDesignVec(),
            MGVec(n=self.num_stress, nb=1)
        )

        # Kona related stuff
        self.allocator = FSTopoAllocator(self, self.prob)
        self.num_primal = x.x.shape[0]*x.x.shape[1]
        self.num_state = force.x.shape[0]*force.x.shape[1]
        self.num_dual = 2*self.num_primal + self.num_stress

        # FD check parameters
        self.use_FD = False
        self.eps_FD = 1e-8

    def get_rank(self):
        return 0

    def enforce_bounds(self, design_vec):
        # loop over design variables
        lower_enforced = False
        upper_enforced = False
        for i in xrange(design_vec.data.x.shape[0]):
            for j in xrange(design_vec.data.x.shape[1]):
                # enforce lower bound
                if design_vec.data.x[i, j] < self.x_min[i, j]:
                    design_vec.data.x[i, j] = self.x_min[i, j]
                    lower_enforced = True
                # enforce upper bound
                if design_vec.data.x[i, j] > self.x_max[i, j]:
                    design_vec.data.x[i, j] = self.x_max[i, j]
                    upper_enforced = True
        if lower_enforced:
            print 'Lower bound enforced!'
        if upper_enforced:
            print 'Upper bound enforced!'

    def restrict_dual(self, dual_vec):
        pass

    def init_design(self, store_here):
        store_here.data.copy(self.init_design_vector)

    def current_solution(self, curr_design, curr_state, curr_adj, curr_dual,
                         num_iter, curr_slack):
        self.curr_design = curr_design
        min_design = min(self.curr_design.data.x)
        max_design = max(self.curr_design.data.x)
        if self.prob.ptype == 1:
            min_thick = 1./max_design
            max_thick = 1./min_design
            output = 'max thickness = %f\n'%max_thick + \
                     'min thickness = %f\n'%min_thick + \
                     '\n' + \
                     'max design    = %f\n'%max_design + \
                     'min design    = %f\n'%min_design
        elif self.prob.ptype == 3:
            min_thick = min_design
            max_thick = max_design
            output = 'max thickness = %f\n'%max_thick + \
                     'min thickness = %f\n'%min_thick
        self.eval_constraints(curr_design, curr_state, self.dual_work)
        output += '\n' + \
                  'max stress    = %e\n'%max(self.dual_work.stress.x) + \
                  'min stress    = %e\n'%min(self.dual_work.stress.x)
        self.prob.writeSolution(
            curr_state.data, curr_adj.data, curr_design.data,
            filename=self.prefix + '/iter_%i.dat'%num_iter)

        design_file = open('design', 'w')
        pickle.dump(
            curr_design.data.x, design_file)
        design_file.close()
        output += '\nDesign saved...\n'
        dual_file = open('dual', 'w')
        pickle.dump(
            [curr_dual.x_lower.x, curr_dual.x_upper.x, curr_dual.stress.x],
            dual_file)
        dual_file.close()
        output += '\nDual saved...\n'
        slack_file = open('slack', 'w')
        pickle.dump(
            [curr_slack.x_lower.x, curr_slack.x_upper.x, curr_slack.stress.x],
            slack_file)
        slack_file.close()
        output += '\nSlack saved...\n'

        return output

    ############################################################################
    # OBJECTIVE ROUTINES
    ############################################################################

    def eval_obj(self, at_design, at_state):
        mass = self.prob.computeMass(at_design.data)
        return (mass, 0)

    def eval_dFdX(self, at_design, at_state, store_here):
        self.prob.computeMassDeriv(at_design.data, store_here.data)

    def eval_dFdU(self, at_design, at_state, store_here):
        store_here.data.zero()

    ############################################################################
    # RESIDUAL ROUTINES
    ############################################################################

    def eval_residual(self, at_design, at_state, store_here):
        # factor the stiffness matrix
        kmat = self.prob.getKMat()
        self.prob.assembleKMat(at_design.data, kmat)
        kmat.factor()
        # assemble the residual
        # R(x, u) = K(x)*u - f
        store_here.data.zero()
        kmat.mult(at_state.data, store_here.data)
        store_here.data.axpy(-1., self.force)

    def multiply_dRdX(self, at_design, at_state, in_vec, out_vec):
        if self.use_FD:
            state_work = self.allocator.alloc_state(2)
            primal_work = self.allocator.alloc_primal(1)[0]
            self.eval_residual(at_design, at_state, state_work[0])
            primal_work.equals_vector(in_vec)
            primal_work.times_scalar(self.eps_FD)
            primal_work.plus(at_design)
            self.eval_residual(primal_work, at_state, state_work[1])
            out_vec.equals_ax_p_by(1., state_work[1], -1., state_work[0])
            out_vec.times_scalar(1./self.eps_FD)
        else:
            out_vec.data.zero()
            self.at_state.copy(at_state.data)
            fstopo.sparse.applyvecbcs(
                self.prob.bcnodes, self.prob.bcvars.T,
                self.at_state.x.T)
            self.prob.multAMatAdd(
                at_design.data, self.at_state, in_vec.data, out_vec.data)

    def multiply_dRdU(self, at_design, at_state, in_vec, out_vec):
        if self.use_FD:
            state_work = self.allocator.alloc_state(3)
            state_work[2].equals_vector(in_vec)
            state_work[2].times_scalar(self.eps_FD)
            state_work[2].plus(at_state)
            self.eval_residual(at_design, at_state, state_work[0])
            self.eval_residual(at_design, state_work[2], state_work[1])
            out_vec.equals_ax_p_by(1., state_work[1], -1., state_work[0])
            out_vec.times_scalar(1./self.eps_FD)
        else:
            # factor the stiffness matrix
            kmat = self.prob.getKMat()
            self.prob.assembleKMat(at_design.data, kmat)
            kmat.factor()
            # perform multiplication
            out_vec.data.zero()
            kmat.mult(in_vec.data, out_vec.data)

    def multiply_dRdX_T(self, at_design, at_state, in_vec, out_vec):
        if self.use_FD:
            out_vec.data.zero()
            state_work = self.allocator.alloc_state(3)
            self.eval_residual(at_design, at_state, state_work[0])
            primal_work = self.allocator.alloc_primal(1)[0]
            # loop over elements
            for i in xrange(at_design.data.x.shape[0]):
                # perturb the state variable
                primal_work.equals_vector(at_design)
                primal_work.data.x[i, :] += self.eps_FD
                # evaluate constraints at perturbed state
                self.eval_residual(
                    primal_work, at_state, state_work[1])
                # calculate the FD approximation
                state_work[2].equals_ax_p_by(
                    1., state_work[1], -1., state_work[0])
                state_work[2].times_scalar(1./self.eps_FD)
                # calculate the product
                out_vec.data.x[i][:] = state_work[2].inner(in_vec)
        else:
            out_vec.data.zero()
            self.at_state.copy(at_state.data)
            fstopo.sparse.applyvecbcs(
                self.prob.bcnodes, self.prob.bcvars.T,
                self.at_state.x.T)
            self.state_work.copy(in_vec.data)
            self.prob.multAMatTransposeAdd(
                at_design.data, self.at_state, self.state_work, out_vec.data)

    def multiply_dRdU_T(self, at_design, at_state, in_vec, out_vec):
        if self.use_FD:
            out_vec.data.zero()
            state_work = self.allocator.alloc_state(4)
            self.eval_residual(at_design, at_state, state_work[0])
            # loop over elements
            for i in xrange(self.prob.conn.shape[0]):
                # loop over nodes in the element
                for k in xrange(4):
                    # loop over degrees of freedom
                    for j in xrange(2):
                        # set up a perturbation vector
                        state_work[1].equals_vector(at_state)
                        state_work[1].data.x[self.prob.conn[i, k]-1, j] += \
                            self.eps_FD
                        # evaluate constraints at perturbed state
                        self.eval_residual(
                            at_design, state_work[1], state_work[2])
                        # calculate the FD approximation
                        state_work[3].equals_ax_p_by(
                            1., state_work[2], -1., state_work[0])
                        state_work[3].times_scalar(1./self.eps_FD)
                        # calculate the product
                        out_vec.data.x[self.prob.conn[i, k]-1, j] = \
                            state_work[3].inner(in_vec)
        else:
            self.multiply_dRdU(at_design, at_state, in_vec, out_vec)

    ############################################################################
    # CONSTRAINT ROUTINES
    ############################################################################

    def eval_constraints(self, at_design, at_state, store_here):
        # evaluate the minimum and maximum thickness constraints
        store_here.x_lower.x[:] = at_design.data.x[:] - self.x_min
        store_here.x_upper.x[:] = self.x_max - at_design.data.x[:]
        # apply BCs to the state point
        store_here.stress.zero()
        self.at_state.copy(at_state.data)
        fstopo.sparse.applyvecbcs(
            self.prob.bcnodes, self.prob.bcvars.T,
            self.at_state.x.T)
        # evaluate stresses
        if self.num_aggr == 0:
            # if there is no aggregation, use smoothed direct stresses
            fstopo.kona.computestressconstraints(
                self.prob.xi, self.prob.eta, self.prob.conn.T,
                self.prob.X.T, self.at_state.x.T,
                self.prob.tconn.T, self.prob.tweights.T,
                self.prob.h.T, self.prob.G.T,
                store_here.stress.x.T)
        else:
            # otherwise use aggregated stresses
            fstopo.kona.computeaggregate(
                self.prob.xi, self.prob.eta, self.prob.conn.T,
                self.prob.X.T, self.at_state.x.T,
                self.prob.tconn.T, self.prob.tweights.T,
                self.prob.h.T, self.prob.G.T, self.ks_rho,
                store_here.stress.x.T)
        if self.cnstr_scale:
            store_here.stress.scale(1./self.num_stress)

    def multiply_dCdX(self, at_design, at_state, in_vec, out_vec):
        out_vec.x_lower.copy(in_vec.data)
        out_vec.x_upper.copy(in_vec.data)
        out_vec.x_upper.scale(-1.)
        out_vec.stress.zero()
        if self.cnstr_scale:
            out_vec.stress.scale(1./self.num_stress)

    def multiply_dCdU(self, at_design, at_state, in_vec, out_vec):
        if self.use_FD:
            out_vec.equals_value(0.0)
            dual_work = self.allocator.alloc_dual(3)
            self.eval_constraints(at_design, at_state, dual_work[0])
            state_work = self.allocator.alloc_state(1)[0]
            self.state_work.copy(in_vec.data)
            fstopo.sparse.applyvecbcs(
                self.prob.bcnodes, self.prob.bcvars.T,
                self.state_work.x.T)
            # loop over state variables
            for i in xrange(self.state_work.x.shape[0]):
                # loop over degrees of freedom
                for j in xrange(self.state_work.x.shape[1]):
                    # calculate the perturbed states
                    state_work.equals_vector(at_state)
                    state_work.data.x[i, j] += self.eps_FD
                    # evaluate constraints at perturbed state
                    self.eval_constraints(
                        at_design, state_work, dual_work[1])
                    # calculate the FD approximation
                    dual_work[2].equals_ax_p_by(
                        1., dual_work[1], -1., dual_work[0])
                    dual_work[2].times_scalar(1./self.eps_FD)
                    # calculate the product contribution
                    dual_work[2].times_scalar(self.state_work.x[i, j])
                    # add to the product
                    out_vec.plus(dual_work[2])
        else:
            out_vec.equals_value(0.0)
            self.at_state.copy(at_state.data)
            fstopo.sparse.applyvecbcs(
                self.prob.bcnodes, self.prob.bcvars.T,
                self.at_state.x.T)
            self.state_work.copy(in_vec.data)
            fstopo.sparse.applyvecbcs(
                self.prob.bcnodes, self.prob.bcvars.T,
                self.state_work.x.T)
            if self.num_aggr == 0:
                fstopo.kona.stressconstraintjacobianproduct(
                    self.prob.xi, self.prob.eta, self.prob.conn.T,
                    self.prob.X.T, self.at_state.x.T,
                    self.prob.tconn.T, self.prob.tweights.T,
                    self.prob.h.T, self.prob.G.T,
                    self.state_work.x.T, out_vec.stress.x.T)
            else:
                fstopo.kona.computeaggregateproduct(
                    self.prob.xi, self.prob.eta, self.prob.conn.T,
                    self.prob.X.T, self.at_state.x.T,
                    self.prob.tconn.T, self.prob.tweights.T,
                    self.prob.h.T, self.prob.G.T, self.ks_rho,
                    self.state_work.x.T, out_vec.stress.x.T)
        if self.cnstr_scale:
            out_vec.stress.scale(1./self.num_stress)

    def multiply_dCdX_T(self, at_design, at_state, in_vec, out_vec):
        out_vec.data.x = in_vec.x_lower.x - in_vec.x_upper.x

    def multiply_dCdU_T(self, at_design, at_state, in_vec, out_vec):
        if self.use_FD:
            out_vec.equals_value(0.0)
            dual_work = self.allocator.alloc_dual(3)
            self.eval_constraints(at_design, at_state, dual_work[0])
            state_work = self.allocator.alloc_state(1)[0]
            # loop over elements
            for i in xrange(self.prob.conn.shape[0]):
                # loop over nodes in the element
                for k in xrange(self.prob.conn.shape[1]):
                    # loop over degrees of freedom
                    for j in xrange(out_vec.data.x.shape[1]):
                        # calculate the perturbed states
                        state_work.equals_vector(at_state)
                        state_work.data.x[self.prob.conn[i, k]-1, j] += \
                            self.eps_FD
                        # evaluate constraints at perturbed state
                        self.eval_constraints(
                            at_design, state_work, dual_work[1])
                        # calculate the FD approximation
                        dual_work[2].equals_ax_p_by(
                            1., dual_work[1], -1., dual_work[0])
                        dual_work[2].times_scalar(1./self.eps_FD)
                        # calculate the product
                        out_vec.data.x[self.prob.conn[i, k]-1, j] = \
                            dual_work[2].inner(in_vec)
            # apply BCs to the outgoing vector
            fstopo.sparse.applyvecbcs(
                self.prob.bcnodes, self.prob.bcvars.T,
                out_vec.data.x.T)
        else:
            out_vec.equals_value(0.0)
            self.at_state.copy(at_state.data)
            fstopo.sparse.applyvecbcs(
                self.prob.bcnodes, self.prob.bcvars.T,
                self.at_state.x.T)
            self.dual_work.equals_vector(in_vec)
            if self.cnstr_scale:
                self.dual_work.stress.scale(1./self.num_stress)
            if self.num_aggr == 0:
                fstopo.kona.stressconstraintjacobiantransproduct(
                    self.prob.xi, self.prob.eta, self.prob.conn.T,
                    self.prob.X.T, self.at_state.x.T,
                    self.prob.tconn.T, self.prob.tweights.T,
                    self.prob.h.T, self.prob.G.T,
                    self.dual_work.stress.x.T, out_vec.data.x.T)
            else:
                fstopo.kona.computeaggregatetransproduct(
                    self.prob.xi, self.prob.eta, self.prob.conn.T,
                    self.prob.X.T, self.at_state.x.T,
                    self.prob.tconn.T, self.prob.tweights.T,
                    self.prob.h.T, self.prob.G.T, self.ks_rho,
                    self.dual_work.stress.x.T, out_vec.data.x.T)
            fstopo.sparse.applyvecbcs(
                self.prob.bcnodes, self.prob.bcvars.T,
                out_vec.data.x.T)

    ############################################################################
    # SOLVE ROUTINES
    ############################################################################

    def factor_linear_system(self, at_design, at_state):
        pass

    def apply_precond(self, at_design, at_state, in_vec, out_vec):
        # factor the stiffness matrix
        kmat = self.prob.getKMat()
        self.prob.assembleKMat(at_design.data, kmat)
        kmat.factor()
        out_vec.data.zero()
        kmat.applyPc(in_vec.data, out_vec.data)

    def apply_precond_T(self, at_design, at_state, in_vec, out_vec):
        self.apply_precond(at_design, at_state, in_vec, out_vec)

    def solve_nonlinear(self, at_design, result):
        if min(at_design.data.x[:]) < 0.:
            return -1

        # Compute the initial state variable values
        kmat = self.prob.getKMat()
        self.prob.assembleKMat(at_design.data, kmat)
        kmat.factor()

        # Solve the system of equations
        result.data.zero()
        niters = fgmres_solve(
            kmat, self.force, result.data, rtol=1e-20, atol=1e-7)

        # Apply BCs to the Solution
        fstopo.sparse.applyvecbcs(
            kmat.bcnodes[0], kmat.bcvars[0].T,
            result.data.x.T)

        # if the fgmres solver maxxed out, we didn't converge
        if niters == 30:
            converged = False
        else:
            converged = True

        cost = niters
        if converged:
            # You can return the number of preconditioner calls here.
            # This is used by Kona to track computational cost.
            return cost
        else:
            # Must return negative cost to Kona when your system solve fails to
            # converge. This is important because it helps Kona determine when
            # it needs to back-track on the optimization.
            return -cost

    def solve_linear(self, at_design, at_state, rhs_vec, rel_tol, result):
        # factor the stiffness matrix
        kmat = self.prob.getKMat()
        self.prob.assembleKMat(at_design.data, kmat)
        kmat.factor()
        # apply the BCs to the RHS vector
        self.state_work.copy(rhs_vec.data)
        fstopo.sparse.applyvecbcs(
            self.prob.bcnodes, self.prob.bcvars.T,
            self.state_work.x.T)
        # zero out the solution
        result.data.zero()
        # check RHS vector to make sure it's not all Zero
        norm = np.sqrt(self.state_work.dot(self.state_work))
        if norm == 0.:
            return 0
        else:
            cost = fgmres_solve(
                kmat, self.state_work, result.data, rtol=rel_tol, atol=1e-7)
            # apply BCs to the solution
            fstopo.sparse.applyvecbcs(
                self.prob.bcnodes, self.prob.bcvars.T,
                result.data.x.T)
            return cost

    def solve_adjoint(self, at_design, at_state, rhs_vec, rel_tol, result):
        return self.solve_linear(at_design, at_state, rhs_vec, rel_tol, result)

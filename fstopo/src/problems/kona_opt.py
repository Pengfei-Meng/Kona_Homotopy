import numpy as np
import pickle, time
import pdb
from kona.user import BaseVector, UserSolver

from fstopo.linalg import MGVec, fgmres_solve
from fstopo.material import *
import fstopo.sparse
import fstopo.kona
import copy

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
        # self.allocator = FSTopoAllocator(self, self.prob)
        self.num_design = x.x.shape[0]*x.x.shape[1]
        self.num_state = force.x.shape[0]*force.x.shape[1]
        self.num_ineq = 2*self.num_design + self.num_stress
        self.num_eq = 0

        super(FSTopoSolver, self).__init__(num_design=self.num_design, 
            num_state=self.num_state, num_eq=0, num_ineq=self.num_ineq)

        # FD check parameters
        self.use_FD = False
        self.eps_FD = 1e-8

        # internal optimization bookkeeping
        self.iterations = 0
        self.totalTime = 0.
        self.startTime = time.clock()

        file = open(self.prefix+'/kona_timings.dat', 'w')
        file.write('# FSTOPO Homotopy iteration timing history\n')
        titles = '# {0:s}    {1:s}    {2:s}    {3:s}    {4:s}   {5:s}   {6:s}\n'.format(
            'Iter', 'Time (s)', 'Total Time (s)', 'Objective Val', 'max(abs(-S*Lam))', 'negative S', 'postive Lam')
        file.write(titles)
        file.close()


    def get_rank(self):
        return 0

    def allocate_state(self, num_vecs):
        return [FSTopoVector(self.prob.createSolutionVec()) for i in range(num_vecs)]

    def enforce_bounds(self, design_vec):
        # print 'enforce_bounds being called!! '
        # loop over design variables
        lower_enforced = False
        upper_enforced = False
        
        design_vec[ design_vec < 0.1 ] = self.x_min
        design_vec[ design_vec > 10.0 ] = self.x_max

        # for i in xrange(len(design_vec)):
        #     # enforce lower bound
        #     if design_vec[i] < self.x_min[i]:
        #         # print 'lower bound', i
        #         design_vec[i] = self.x_min[i] 
        #         lower_enforced = True
        #     # enforce upper bound
        #     if design_vec[i] > self.x_max[i]:
        #         # print 'upper bound', i
        #         design_vec[i] = self.x_max[i]
        #         upper_enforced = True
        #if lower_enforced:
            # print 'Lower bound enforced!'
        #if upper_enforced:
            # print 'Upper bound enforced!'

    def restrict_dual(self, dual_vec):
        pass

    def init_design(self):
        return self.init_design_vector.x.flatten()

    def init_slack(self):
        at_design = self.init_design()
        state_work = self.allocate_state(1)[0]

        cost = self.solve_nonlinear(at_design, state_work)
        at_slack = self.eval_ineq_cnstr(at_design, state_work)
        # at_slack, cost = 0.1*np.ones(self.num_ineq), 0
        return (at_slack, cost)

    def current_solution(self, num_iter, curr_design, curr_state, curr_adj, curr_eq,
            curr_ineq,curr_slack):
                         
        self.curr_design = curr_design
        self.curr_ineq = curr_ineq
        self.curr_adj = curr_adj
        self.curr_slack = curr_slack

        min_design = min(self.curr_design)
        max_design = max(self.curr_design)
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
    
        dual_work = self.eval_ineq_cnstr(curr_design, curr_state)
        dual_work_stress = dual_work[-self.num_stress:]
        output += '\n' + \
                  'max stress    = %e\n'%max(dual_work_stress) + \
                  'min stress    = %e\n'%min(dual_work_stress)
        # self.prob.writeSolution(
        #     curr_state.data, curr_adj.data, MGVec(curr_design[:,np.newaxis]),
        #     filename=self.prefix + '/iter_%i.dat'%num_iter)
        self.prob.writeSolution(
            curr_state.data, curr_adj.data, MGVec(curr_design[:,np.newaxis]),
            filename=self.prefix + '/iter_last.dat')


        # ----------------------------------------------------------
        # time the iteration
        self.endTime = time.clock()
        duration = self.endTime - self.startTime
        self.totalTime += duration
        self.startTime = self.endTime

        objVal, _ = self.eval_obj(curr_design, curr_state)

        # max_constr_violation = min(self.eval_ineq_cnstr(curr_design, curr_state))
        slack_lamda = max(abs(curr_slack*curr_ineq))
        slack_lamda_l = abs(curr_slack*curr_ineq)
        neg_S = sum(curr_slack < -1e-5)
        pos_Lam = sum(curr_ineq > 1e-5)


        # checking strict complementariy condition
        indx = (slack_lamda_l < 1e-5) & ( abs(curr_slack) < 1e-3 ) & ( abs(curr_ineq) < 1e-3 )
        if sum(indx) > 0:
            print 'Strict Complementarity not satisfied! Inner iteration: ',  num_iter
            print 'slack: ', curr_slack[indx]
            print 'dual: ', curr_ineq[indx]

        # else:
        #     print 'Strict Complementarity Satisfied! '


        # write timing to file
        timing = '  {0:3d}        {1:4.2f}        {2:4.2f}        {3:4.6g}      {4:4.6f}   {5:3d}   {6:3d}\n'.format(
            num_iter, duration, self.totalTime, objVal, slack_lamda, neg_S, pos_Lam)
        file = open(self.prefix + '/kona_timings.dat', 'a')
        file.write(timing)
        file.close()

        # design_file = open(self.prefix + '/design_%i'%num_iter, 'w')
        # pickle.dump(
        #     curr_design, design_file)
        # design_file.close()
        # output += '\nDesign saved...\n'
        # dual_file = open(self.prefix + '/dual_%i'%num_iter, 'w')
        # pickle.dump(
        #     curr_ineq,
        #     dual_file)
        # dual_file.close()
        # output += '\nDual saved...\n'
        # slack_file = open(self.prefix + '/slack_%i'%num_iter, 'w')
        # pickle.dump(
        #     curr_slack,
        #     slack_file)
        # slack_file.close()
        # output += '\nSlack saved...\n'

        return output

    ############################################################################
    # OBJECTIVE ROUTINES
    ############################################################################

    def eval_obj(self, at_design, at_state):
        
        mass = self.prob.computeMass(MGVec(at_design[:,np.newaxis]))

        return (mass, 0)

    def eval_dFdX(self, at_design, at_state):
        designV = MGVec(at_design[:,np.newaxis])
        storeV = MGVec()
        self.prob.computeMassDeriv(designV, storeV)

        return storeV.x.flatten()

    def eval_dFdU(self, at_design, at_state, store_here):
        store_here.data.zero()

    ############################################################################
    # RESIDUAL ROUTINES
    ############################################################################

    def eval_residual(self, at_design, at_state, store_here):
        # factor the stiffness matrix
        kmat = self.prob.getKMat()
        self.prob.assembleKMat(MGVec(at_design[:,np.newaxis]), kmat)
        kmat.factor()
        # assemble the residual
        # R(x, u) = K(x)*u - f
        store_here.data.zero()
        kmat.mult(at_state.data, store_here.data)
        store_here.data.axpy(-1., self.force)

    def multiply_dRdX(self, at_design, at_state, in_vec, out_vec):
        designV = MGVec(at_design[:,np.newaxis])
        inV = MGVec(in_vec[:,np.newaxis])
        # print 'multiply_dRdX called!! '

        if self.use_FD:
            state_work = self.allocate_state(2)
            # primal_work = np.zeros_like(at_design)
            self.eval_residual(at_design, at_state, state_work[0])
            primal_work = copy.deepcopy(in_vec)
            primal_work *= self.eps_FD
            primal_work += at_design
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
                designV, self.at_state, inV, out_vec.data)
            
        # print 'multiply_dRdX finished!! '

    def multiply_dRdU(self, at_design, at_state, in_vec, out_vec):
        designV = MGVec(at_design[:,np.newaxis])

        if self.use_FD:
            state_work = self.allocate_state(3)
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
            self.prob.assembleKMat(designV, kmat)
            kmat.factor()
            # perform multiplication
            out_vec.data.zero()
            kmat.mult(in_vec.data, out_vec.data)

    def multiply_dRdX_T(self, at_design, at_state, in_vec):

        designV = MGVec(at_design[:,np.newaxis])
        out_vec = MGVec(n=len(at_design), nb=1)
        # out_vec.zero()

        if self.use_FD:
            state_work = self.allocate_state(3)
            self.eval_residual(at_design, at_state, state_work[0])
            # primal_work = np.zeros_like(at_design)
            # loop over elements
            for i in xrange(len(at_design)):
                # perturb the state variable
                primal_work = copy.deepcopy(at_design)
                primal_work[i, :] += self.eps_FD
                # evaluate constraints at perturbed state
                self.eval_residual(
                    primal_work, at_state, state_work[1])
                # calculate the FD approximation
                state_work[2].equals_ax_p_by(
                    1., state_work[1], -1., state_work[0])
                state_work[2].times_scalar(1./self.eps_FD)
                # calculate the product
                out_vec = state_work[2].inner(in_vec)
        else:
            self.at_state.copy(at_state.data)
            fstopo.sparse.applyvecbcs(
                self.prob.bcnodes, self.prob.bcvars.T,
                self.at_state.x.T)
            self.state_work.copy(in_vec.data)
            self.prob.multAMatTransposeAdd(
                designV, self.at_state, self.state_work, out_vec)

        return out_vec.x.flatten()

    def multiply_dRdU_T(self, at_design, at_state, in_vec, out_vec):
        if self.use_FD:
            out_vec.data.zero()
            state_work = self.allocate_state(4)
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

    def eval_ineq_cnstr(self, at_design, at_state):

        x_lower =  MGVec(at_design[:,np.newaxis] - self.x_min) 
        x_upper =  MGVec(self.x_max - at_design[:,np.newaxis])
        stress = MGVec(n=len(at_design), nb=1)

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
                stress.x.T)
        else:
            # otherwise use aggregated stresses
            fstopo.kona.computeaggregate(
                self.prob.xi, self.prob.eta, self.prob.conn.T,
                self.prob.X.T, self.at_state.x.T,
                self.prob.tconn.T, self.prob.tweights.T,
                self.prob.h.T, self.prob.G.T, self.ks_rho,
                stress.x.T)
        if self.cnstr_scale:
            stress.scale(1./self.num_stress)

        return np.concatenate((x_lower.x.flatten(), x_upper.x.flatten(), stress.x.flatten()))

    def multiply_dCINdX(self, at_design, at_state, in_vec):

        x_lower = in_vec
        x_upper = -in_vec
        stress = np.zeros_like(in_vec)
        if self.cnstr_scale:
            stress *= 1./self.num_stress
        return np.concatenate((x_lower, x_upper, stress))


    def multiply_dCINdU(self, at_design, at_state, in_vec):

        x_lower = np.zeros_like(at_design[:,np.newaxis])
        x_upper = np.zeros_like(at_design[:,np.newaxis])
        stress  = np.zeros_like(at_design[:,np.newaxis])

        if self.use_FD:
            out_vec.equals_value(0.0)
            # dual_work = self.allocator.alloc_dual(3)
            dual_work0 = self.eval_ineq_cnstr(at_design, at_state)
            state_work = self.allocate_state(1)[0]
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
                    dual_work1 = self.eval_ineq_cnstr(at_design, state_work)
                       
                    # calculate the FD approximation
                    dual_work2 = dual_work1 - dual_work0
                    dual_work2 *= 1./self.eps_FD
                    # calculate the product contribution
                    dual_work2 *= (self.state_work.x[i, j])
                    # add to the product
                    x_lower += dual_work2[:len(at_design)]
                    x_upper += dual_work2[len(at_design):2*len(at_design)]
                    stress += dual_work2[-self.num_stress:]
        else:
            
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
                    self.state_work.x.T, stress.T)
            else:
                fstopo.kona.computeaggregateproduct(
                    self.prob.xi, self.prob.eta, self.prob.conn.T,
                    self.prob.X.T, self.at_state.x.T,
                    self.prob.tconn.T, self.prob.tweights.T,
                    self.prob.h.T, self.prob.G.T, self.ks_rho,
                    self.state_work.x.T, stress.T)
        if self.cnstr_scale:
            stress *= 1./self.num_stress

        return np.concatenate((x_lower.flatten(), x_upper.flatten(), stress.flatten()))

    def multiply_dCINdX_T(self, at_design, at_state, in_vec):
        out_vec = in_vec[:len(at_design)] - in_vec[len(at_design):2*len(at_design)]
        return out_vec

    def multiply_dCINdU_T(self, at_design, at_state, in_vec, out_vec):
        in_x_lower = in_vec[:len(at_design)]
        in_x_upper = in_vec[len(at_design):2*len(at_design)]
        in_stress = in_vec[2*len(at_design):]
        if self.use_FD:
            out_vec.equals_value(0.0)

            dual_work0 = self.eval_ineq_cnstr(at_design, at_state)
            state_work = self.allocate_state(1)[0]
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
                        dual_work1 = self.eval_ineq_cnstr(at_design, state_work)
                        dual_work2 = dual_work1 - dual_work0
                        # calculate the FD approximation
                        dual_work2 *= 1./self.eps_FD
                        # calculate the product
                        out_vec.data.x[self.prob.conn[i, k]-1, j] = \
                            dual_work2.dot(in_vec)
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
            # self.dual_work.equals_vector(in_vec)
            if self.cnstr_scale:
                in_stress*=1./self.num_stress
            if self.num_aggr == 0:
                fstopo.kona.stressconstraintjacobiantransproduct(
                    self.prob.xi, self.prob.eta, self.prob.conn.T,
                    self.prob.X.T, self.at_state.x.T,
                    self.prob.tconn.T, self.prob.tweights.T,
                    self.prob.h.T, self.prob.G.T,
                    in_stress.T, out_vec.data.x.T)
            else:
                fstopo.kona.computeaggregatetransproduct(
                    self.prob.xi, self.prob.eta, self.prob.conn.T,
                    self.prob.X.T, self.at_state.x.T,
                    self.prob.tconn.T, self.prob.tweights.T,
                    self.prob.h.T, self.prob.G.T, self.ks_rho,
                    in_stress.T, out_vec.data.x.T)
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
        self.prob.assembleKMat(MGVec(at_design[:,np.newaxis]), kmat)
        kmat.factor()
        out_vec.data.zero()
        kmat.applyPc(in_vec.data, out_vec.data)

    def apply_precond_T(self, at_design, at_state, in_vec, out_vec):
        self.apply_precond(at_design, at_state, in_vec, out_vec)

    def solve_nonlinear(self, at_design, result):
        if min(at_design[:]) < 0.:
            return -1
         
        # Compute the initial state variable values
        MGv = MGVec(at_design[:,np.newaxis])

        kmat = self.prob.getKMat()

        self.prob.assembleKMat(MGv, kmat)
        kmat.factor()

        # Solve the system of equations
        result.data.zero()
        niters = fgmres_solve(
            kmat, self.force, result.data, rtol=1e-16, atol=1e-8)    #rtol=1e-20, atol=1e-7

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
            return cost      # 9
        else:
            # Must return negative cost to Kona when your system solve fails to
            # converge. This is important because it helps Kona determine when
            # it needs to back-track on the optimization.
            return -cost

    def solve_linear(self, at_design, at_state, rhs_vec, rel_tol, result):
        # factor the stiffness matrix

        kmat = self.prob.getKMat()
        self.prob.assembleKMat(MGVec(at_design[:,np.newaxis]), kmat)
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
            # print 'rhs norm is 0, so cost is zero for solve_linear'
            return 0
        else:
            cost = fgmres_solve(
                kmat, self.state_work, result.data, rtol=rel_tol, atol=1e-7)
            # apply BCs to the solution
            fstopo.sparse.applyvecbcs(
                self.prob.bcnodes, self.prob.bcvars.T,
                result.data.x.T)

            # print 'cost for solve_linear: ', cost
            return cost       # 6


    def solve_adjoint(self, at_design, at_state, rhs_vec, tol, result):
        return self.solve_linear(at_design, at_state, rhs_vec, tol, result)

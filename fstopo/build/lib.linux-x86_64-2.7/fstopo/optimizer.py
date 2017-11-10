import sys
import numpy as np
from mpi4py import MPI

# The sparse matrix routines
import fstopo.sparse as sparse

from fstopo.linalg import fgmres, fgmres_solve, KKTVec

class FullSpaceOpt:
    def __init__(self, prob):
        '''
        Set up a basic multigrid method for a problem on a rectangular
        domain
        '''

        # Record all the problem data
        self.prob = prob

        # Set the minimum value of rho to use
        self.rho_min = 0.0

        # Set the barrier parameter
        self.barrier = 1000.0
        self.stress_barrier_scale = 1.0
        self.barrier_target = 1.0

        # Set the preconditioner type
        self.pc_type = 'Biros'

        # Set default parameter values
        self.fgmres_subspace_size = 80
        self.fgmres_min_iters = -1
        self.pc_gmres_subspace_size = 0
        self.opt_tol = 1e-5
        self.barrier_fraction = 0.25
        self.max_newton_iters = 100
        self.max_newton_tol = 1e-2
        self.max_line_iters = 8
        self.c1 = 1e-4

        # Parameters for verifying the implementation
        self.kkt_test_iter = -1 # Perform checks at this iteration
        self.line_search_check = False

        # Get the matrices that we'll use later
        self.kmat = self.prob.getKMat()
        self.dmat = self.prob.getDMat()
        self.dmat_exact = self.dmat.duplicate()

        # Flag to indicate whether algormithm will switch to use
        # the exact Hessian at some point
        self.use_exact_hessian = False

        # Flag to indicate whether the exact Hessian is in use
        self.exact_hessian = False

        return

    def optimize(self, x, u, lb, ub, force, lambd=None,
                 filename='opt_hist.dat', prefix='results'):
        '''
        Perform a full-space optimization.
        '''

        fp = None
        if filename is not None:
            fp = open(filename, 'w')
            fp.write('Variables =\n')

        # Get the initial design point
        self.x = x
        self.lambd = lambd
        self.u = u
        self.lb = lb
        self.ub = ub
        self.f = force

        # Create the design vector and initialize the lower/upper bounds
        self.xtemp = self.prob.createDesignVec()
        self.ddiag = self.prob.createDesignVec()

        # Create the vectors for everything else
        self.psi = self.prob.createSolutionVec()
        self.rtemp = self.prob.createSolutionVec()
        self.utemp = self.prob.createSolutionVec()

        if self.lambd is not None:
            self.wtemp1 = self.prob.createWeightVec()
            self.wtemp2 = self.prob.createWeightVec()

        # Assemble and factor the stiffness matrix
        self.prob.assembleKMat(self.x, self.kmat)
        self.kmat.factor()

        # Compute an initial estimate of the Lagrange multipliers
        self.utemp.zero()
        self.prob.addLogStressSumDeriv(self.x, self.u,
                                       self.stress_barrier_scale*self.barrier,
                                       self.xtemp, self.utemp)
        self.utemp.scale(-1.0)
        fgmres_solve(self.kmat, self.utemp, self.psi,
                     m=self.fgmres_subspace_size, rtol=1e-12, print_flag=0)

        # Create the vectors
        res = KKTVec(self.x, self.lambd, self.u, self.psi)
        step = res.duplicate()
        temp = res.duplicate()

        # Allocate the full Krylov subspace
        W = [res.duplicate()]
        Z = []
        for i in xrange(self.fgmres_subspace_size):
            W.append(res.duplicate())
            Z.append(res.duplicate())

        # Copy over the subspace for the dipslacement components
        Wu = []
        Zu = []
        for w in W:
            Wu.append(w.u)
        for z in Z:
            Zu.append(z.u)

        # Set up a small subspace for u
        if self.pc_gmres_subspace_size > 0:
            self.Wd = [self.utemp.duplicate()]
            self.Zd = []
            for i in xrange(self.pc_gmres_subspace_size):
                self.Wd.append(self.utemp.duplicate())
                self.Zd.append(self.utemp.duplicate())

        # Write out the header to the screen
        s = '%5s %10s %10s %10s %10s %10s %10s %10s %10s %10s '%(
            'Iter', 'mass', 'alpha', 'alpha_max', 'mu',
            '||Rx||', '||Rlambd||', '||Ru||', '||Rpsi||', '||Ropt||')
        s += '%6s %4s %9s'%('GMRES', 'diff', 'nktol')
        print s
        if fp:
            fp.write(s + '\n')
            fp.flush()

        # Keep track of data that will be used to track the
        # performance of the algorithm
        niters = 0
        gmres_count = 0

        # Do not use an exact Hessian at first
        self.exact_hessian = False

        # Keep track of the iterations
        itr = 0

        # Perform the Newton-Krylov iterations
        new_barrier = False

        # We try to always use the merit function line search, but
        # sometimes we don't!
        merit_line_search = True

        # Keep track of the most recent residual norm
        res_norm = 1e20

        # Keep track of the penalty search parameter
        rho_hat = self.rho_min

        # Set the value of alpha
        alpha = 1.0
        alpha_max = 1.0

        # Keep track of the difference in mass between iterations -
        # this is not a great convergence test, but I'm not sure what
        # else to do
        mass_prev = 1.0

        for inewton in xrange(self.max_newton_iters):
            # Write the solution file
            if itr % 5 == 0:
                fname = '%s/solution%03d.dat'%(prefix, itr)
                self.prob.writeSolution(self.u, self.psi, self.x,
                                        filename=fname)

            # Check whether to use the exact Hessian on the next time
            # through
            if (self.use_exact_hessian and
                    self.barrier == self.barrier_target):
                self.exact_hessian = True

            # If this is a new barrier, adjust the Lagrange multipliers
            if new_barrier:
                old_barrier = 1.0*self.barrier
                self.barrier = max(self.barrier_fraction*self.barrier,
                                   self.barrier_target)
                self.psi.scale(self.barrier/old_barrier)
                if self.lambd is not None:
                    self.lambd.scale(self.barrier/old_barrier)

            # Assemble the residuals and approximate Jacobians for the
            # KKT system
            self.assembleKKTResMat(self.x, self.lambd, self.u, self.psi,
                                   self.barrier, res)

            if inewton == self.kkt_test_iter:
                self.prob.test(self.x, self.u)
                self.testKKTJacobian()

            # Compute the residual norms of each component
            res_norm = np.sqrt(res.dot(res))

            # Compute the individual component norms for output
            x_norm = res.x.infty()
            lambd_norm = 0.0
            if self.lambd is not None:
                lambd_norm = res.lambd.infty()

            # Normalize the Lagrange multiplier equation residual by
            # the barrier parameter
            u_norm = res.u.infty()/max(1.0, self.barrier)
            psi_norm = res.psi.infty()

            # Take the max of the infinity norms
            res_infty = max(x_norm, lambd_norm, u_norm, psi_norm)

            # Use a fixed relative tolerance, except if the last line
            # search was not very successful
            nktol = 1.0*self.max_newton_tol
            if alpha < 0.1*alpha_max:
                nktol *= 0.1

            # Compute the mass at the current design point
            mass = self.prob.computeMass(self.x)

            # Print out the residual to the screen
            s = '%5d %10.4e %10.3e %10.3e %10.3e '%(
                itr, mass, alpha, alpha_max, self.barrier)
            s += '%10.3e %10.3e %10.3e %10.3e %10.3e '%(
                x_norm, lambd_norm, u_norm, psi_norm, res_infty)
            s += '%6d %4d %9.2e'%(
                gmres_count, niters, nktol)
            itr += 1

            # Print things to the screen first, then to the file
            print s
            sys.stdout.flush()
            if fp:
                fp.write(s + '\n')
                fp.flush()

            # Test convergence
            if (self.barrier == self.barrier_target and
                ((x_norm < self.opt_tol and
                  lambd_norm < self.opt_tol and
                  psi_norm < self.opt_tol) or
                 (alpha == 1.0 and
                  np.fabs((mass - mass_prev)/mass) < 1e-3))):
                break

            # Keep track of the mass at the last iteration
            mass_prev = 1.0*mass

            # Check if we need a new barrier
            if new_barrier:
                # Reset the penalty search parameter
                rho_hat = self.rho_min
                new_barrier = False
            elif res_infty < 10.0*self.barrier:
                new_barrier = True
                if self.barrier == self.barrier_target:
                    break

            # Compute the Newton step using the W/Z subspace
            t0 = MPI.Wtime()
            niters = fgmres(self, res, step, W, Z, rtol=nktol, print_flag=5,
                            min_iters=self.fgmres_min_iters)
            tgmres = MPI.Wtime() - t0
            print 'FGMRES[ts]: %10.3e s/iter'%(tgmres/niters)
            gmres_count += niters
            step.scale(-1.0)

            # Compute a step that does not violate the stress
            # constraints or the lower bound constraint
            alpha_max = self.computeMaxStep(self.x, self.u,
                                            step.x, step.u)

            # Save the initial value of the line search parameter
            alpha = 1.0*alpha_max

            # Test the step, if this is the right iteration
            if inewton == self.kkt_test_iter:
                self.testKKTStep(step, res)

            # This is a globalization-ish procedure. We accept the
            # step when it either satisfies ||R(x+p)||_{2} <
            # (1-alpha*c1)*||R(x)||_{2}, or when the merit function
            # line search is satisfied. If we don't find a descent
            # direction, we use ||R(x+alpha*p)||_{2} as a merit
            # function. If that doesn't work then we crash and burn.

            # Always try a merit function line search first
            merit_line_search = True

            # Evaluate the derivative of the merit function at alpha = 0
            m0, dm0, infeas0, rho_hat = self.evalMeritDeriv(self.x, self.u,
                                                            step.x, step.u,
                                                            rho=rho_hat)

            print '%5s %10s %10s %10s %10s %10s'%(
                'Iter', ' ', 'alpha', 'Merit', 'Deriv', 'rho')
            print '%5d %10s %10.3e %10.2e %10.2e %10.2e'%(
                0, ' ', alpha, m0, dm0, rho_hat)

            if merit_line_search:
                if dm0 >= 0.0:
                    # This is not a descent direction
                    merit_line_search = False
                else:
                    if self.line_search_check:
                        # Check the directional derivative against
                        # finite difference. Evaluate the derivative
                        # of the merit function at alpha = 0
                        dh = 1e-5
                        temp.x.copy(self.x)
                        temp.u.copy(self.u)
                        temp.x.axpy(-dh, step.x)
                        temp.u.axpy(-dh, step.u)
                        m1, infeas1 = self.evalMerit(temp.x, temp.u,
                                                     rho=rho_hat)

                        temp.x.copy(self.x)
                        temp.u.copy(self.u)
                        temp.x.axpy(dh, step.x)
                        temp.u.axpy(dh, step.u)
                        m2, infeas2 = self.evalMerit(temp.x, temp.u,
                                                     rho=rho_hat)
                        fd = 0.5*(m2 - m1)/dh

                        # Print out the results of the line search check
                        print '%5s %10s %10s %10s %10s %10s'%(
                            'Iter', ' ', 'alpha', 'Deriv', 'FD', 'Rel err')
                        print '%5d %10s %10.3e %10.2e %10.2e %10.2e'%(
                            0, ' ', alpha, dm0, fd, (dm0 - fd)/dm0)

                    # Perform the line search
                    soc = False
                    for i in xrange(self.max_line_iters):
                        # Compute the variables at the next step
                        temp.x.copy(self.x)
                        temp.u.copy(self.u)
                        temp.x.axpy(alpha, step.x)
                        temp.u.axpy(alpha, step.u)

                        # Evaluate the merit function at the first step point
                        m1, infeas1 = self.evalMerit(temp.x, temp.u,
                                                     rho=rho_hat)

                        print '%5d %10s %10.3e %10.2e'%(i+1, ' ', alpha, m1)

                        # Check if the first Wolfe condition is satisfied,
                        # otherwise quit the line search
                        if m1 != m1:
                            alpha *= 0.5
                        elif (m1 < m0 + alpha*dm0*self.c1):
                            break
                        elif i == 0 and infeas1 > infeas0:
                            # The infeasibility accounted for the line
                            # search failure, try a second-order
                            # correction step to find an acceptable
                            # step length through a second-order
                            # correction step (just in u):

                            # Assemble and factor the stiffness matrix
                            self.prob.assembleKMat(temp.x, self.kmat)
                            self.kmat.mult(temp.u, res.u)
                            res.u.axpy(-1.0, self.f)

                            # Factor the preconditioner, and
                            self.kmat.factor()
                            fgmres(self.kmat, res.u, self.utemp,
                                   Wu, Zu, rtol=1e-6, print_flag=0)
                            self.utemp.scale(-1.0)

                            # Compute the maximum step
                            self.xtemp.zero()
                            alpha_new = self.computeMaxStep(
                                temp.x, temp.u, self.xtemp, self.utemp)
                            temp.u.axpy(alpha_new, self.utemp)

                            # Evaluate the merit function
                            m2, infeas2 = self.evalMerit(temp.x, temp.u,
                                                         rho=rho_hat)

                            print '%5d %10s %10.3e %10.2e'%(
                                i+1, 'SOC', alpha_new, m2)

                            # Check if this point decreases the merit function
                            if m2 != m2:
                                alpha *= 0.5
                            elif infeas2 < infeas0:
                                # Copy over the new values
                                soc = True
                                self.x.copy(temp.x)
                                self.u.copy(temp.u)

                                # Compute the new values of the multipliers
                                if self.lambd is not None:
                                    self.lambd.axpy(alpha, step.lambd)
                                self.psi.axpy(alpha, step.psi)

                                break
                            else:
                                alpha *= 0.5
                        else:
                            alpha *= 0.5

                    if not soc:
                        self.x.axpy(alpha, step.x)
                        if self.lambd is not None:
                            self.lambd.axpy(alpha, step.lambd)
                        self.u.axpy(alpha, step.u)
                        self.psi.axpy(alpha, step.psi)

            # This is a line search based on the norm of the residual.
            # where merit = ||R(x + alpha*p)||_{2}. p is a descent
            # direction, but the derivative dm0 = -||R||_{2} is an
            # estimate.
            if not merit_line_search:
                m0 = res_norm
                dm0 = -res_norm

                print '%5s %10s %10s %10s %10s'%(
                    'Iter', ' ', 'alpha', 'Merit', 'Deriv')
                print '%5d %10s %10.3e %10.2e %10.2e'%(0, ' ', alpha, m0, dm0)

                for i in xrange(self.max_line_iters):
                    # Compute the variables at the next step
                    temp.x.copy(self.x)
                    if self.lambd is not None:
                        temp.lambd.copy(self.lambd)
                    temp.u.copy(self.u)
                    temp.psi.copy(self.psi)

                    # Take the step
                    temp.axpy(alpha, step)

                    # Evaluate the residual at the next point
                    self.assembleKKTRes(temp.x, temp.lambd, temp.u, temp.psi,
                                        self.barrier, res)

                    # Compute the residual norm
                    m1 = np.sqrt(res.dot(res))
                    print '%5d %10s %10.3e %10.2e'%(i+1, ' ', alpha, m1)

                    # Check the update
                    if m1 < m0 + alpha*dm0*self.c1:
                        break
                    else:
                        alpha *= 0.5

                self.x.axpy(alpha, step.x)
                if self.lambd is not None:
                    self.lambd.axpy(alpha, step.lambd)
                self.u.axpy(alpha, step.u)
                self.psi.axpy(alpha, step.psi)

        # Write the solution out at the final iteration
        fname = '%s/solution%03d.dat'%(prefix, itr)
        self.prob.writeSolution(self.u, self.psi, self.x,
                                filename=fname)

        return

    def evalMeritDeriv(self, x, u, px, pu, rho=0.0,
                       xi=0.0, eta=0.0, descent_fraction=0.3):
        '''
        Evaluate the derivative of the merit function
        '''

        # Compute the merit function
        f1 = self.prob.computeMass(x)

        # Add the terms from the constraint bounds
        if self.lb is not None:
            f1 -= self.barrier*np.sum(np.log(x.x - self.lb.x))
        if self.ub is not None:
            f1 -= self.barrier*np.sum(np.log(self.ub.x - x.x))

        # Compute the sum of log(stress)
        logsum = self.prob.computeLogStressSum(x, u)
        m0 = f1 - self.stress_barrier_scale*self.barrier*logsum

        # Compute the derivative of the merit function
        self.prob.computeMassDeriv(x, self.xtemp)
        df1 = self.xtemp.dot(px)

        if self.lb is not None:
            df1 -= self.barrier*np.sum(px.x/(x.x - self.lb.x))
        if self.ub is not None:
            df1 += self.barrier*np.sum(px.x/(self.ub.x - x.x))

        # Compute the contribution to the directional derivative from
        # the barrier objective
        self.xtemp.zero()
        self.utemp.zero()
        self.prob.addLogStressSumDeriv(x, u,
                                       self.stress_barrier_scale*self.barrier,
                                       self.xtemp, self.utemp)
        dm0 = df1 + self.utemp.dot(pu) + self.xtemp.dot(px)

        # Compute the residual of the stiffness matrix
        self.prob.assembleKMat(x, self.kmat)
        self.kmat.mult(u, self.rtemp)
        self.rtemp.axpy(-1.0, self.f)
        norm = np.sqrt(self.rtemp.dot(self.rtemp))

        # Compute utemp = K(x)*pu
        self.kmat.mult(pu, self.utemp)
        self.prob.multAMatAdd(x, u, px, self.utemp)

        # Compute the directional derivative of the term ||K(x)*u - f||_{2}
        # deriv = (K(x)*u - f)^{T}(K(x)*pu + A(u)*px)
        deriv = self.rtemp.dot(self.utemp)/norm

        # If the weight constraints are active, compute their
        # effect
        if self.lambd is not None:
            # Evaluate the weight constraint and constraint norm
            self.prob.computeWeightCon(x, self.wtemp1)
            wnorm = np.sqrt(self.wtemp1.dot(self.wtemp1))

            # Evaluate the derivative
            if wnorm > 0.0:
                self.wtemp2.zero()
                self.prob.multAwMatAdd(x, px, self.wtemp2)
                wderiv = self.wtemp1.dot(self.wtemp2)/wnorm
            else:
                wderiv = 0.0

            deriv = deriv + wderiv
            norm = norm + wnorm

        # Determine the value of rho to ensure that we have a descent
        # direction
        rho = max(0.5*rho, self.rho_min,
                  -dm0/(deriv + descent_fraction*norm))

        m0 = m0 + rho*norm
        dm0 = dm0 + rho*deriv

        return m0, dm0, norm, rho

    def evalMerit(self, x, u, rho=10.0, xi=0.0, eta=0.0):
        '''
        Evaluate the merit function
        '''

        # Evaluate the merit function
        f1 = self.prob.computeMass(x)

        # Add the terms from the constraint bounds
        if self.lb is not None:
            f1 -= self.barrier*np.sum(np.log(x.x - self.lb.x))
        if self.ub is not None:
            f1 -= self.barrier*np.sum(np.log(self.ub.x - x.x))

        # Compute the sum of log(stress)
        logsum = self.prob.computeLogStressSum(x, u)

        # Assemble the residual of the stiffness matrix
        self.prob.assembleKMat(x, self.kmat)
        self.kmat.mult(u, self.rtemp)
        self.rtemp.axpy(-1.0, self.f)
        norm = np.sqrt(self.rtemp.dot(self.rtemp))

        # Compute the weighting constraints
        if self.lambd is not None:
            self.prob.computeWeightCon(x, self.wtemp1)
            norm += np.sqrt(self.wtemp1.dot(self.wtemp1))

        m0 = f1 - self.stress_barrier_scale*self.barrier*logsum + rho*norm

        return m0, norm

    def computeMaxStep(self, x, u, xstep, ustep, tau=0.95):
        '''
        Compute the maximum step along the given search direction
        before violating either the lower bound or the stress
        constraints.
        '''

        alpha = self.prob.computeMaxStressStep(x, u, xstep, ustep, tau)
        if alpha < 0.0:
            print 'Negative maximum step returned from stress-constraint search'

        # Calculate the maximum step from bounds
        ne = x.x.shape[0]
        nb = x.x.shape[1]
        if self.lb is not None:
            for i in xrange(ne):
                for j in xrange(nb):
                    if xstep.x[i,j] < 0.0:
                        tmp = -tau*(x.x[i,j] - self.lb.x[i,j])/xstep.x[i,j]
                        alpha = min(alpha, tmp)

        if self.ub is not None:
            for i in xrange(ne):
                for j in xrange(nb):
                    if xstep.x[i,j] > 0.0:
                        tmp = tau*(self.ub.x[i,j] - x.x[i,j])/xstep.x[i,j]
                        alpha = min(alpha, tmp)

        if alpha < 0.0:
            print 'Error in upper-bound step computation'

        return alpha

    def checkBounds(self, x):
        '''
        Check the bounds - return True if x is inside the bounds, and
        False otherwise.
        '''

        # Get the dimensions of the array
        ne = x.x.shape[0]
        nb = x.x.shape[1]
        for i in xrange(ne):
            for j in xrange(nb):
                if x.x[i,j] < self.lb.x[i,j] or x.x[i,j] > self.ub.x[i,j]:
                    return False

        return True

    def assembleKKTRes(self, x, lambd, u, psi, barrier, res):
        '''
        Assemble the residual of the KKT system in the vector res.
        '''

        # Compute the residual from the first equation
        self.prob.computeMassDeriv(x, res.x)

        # Complete the design variable residual
        if self.lb is not None:
            res.x.x[:] -= barrier/(x.x - self.lb.x)
        if self.ub is not None:
            res.x.x[:] += barrier/(self.ub.x - x.x)

        # Add res.x += A(u)^{T}*psi
        self.prob.multAMatTransposeAdd(x, u, psi, res.x)

        # Add the contribution from the weighting constraints
        if self.lambd is not None:
            self.prob.multAwMatTransposeAdd(x, lambd, res.x)
            self.prob.computeWeightCon(x, res.lambd)

        # Assemble and factor the stiffness matrix
        self.prob.assembleKMat(x, self.kmat)

        # Compute the values of the second residual
        self.kmat.mult(psi, res.u)
        self.prob.addLogStressSumDeriv(x, u,
                                       self.stress_barrier_scale*barrier,
                                       res.x, res.u)

        # Compute the residuals of the third equations (equilibrium)
        self.kmat.mult(u, res.psi)
        res.psi.axpy(-1.0, self.f)

        return

    def assembleKKTResMat(self, x, lambd, u, psi, barrier,
                          res, diag_fact=0.0):
        '''
        Assemble the residual of the KKT system in the vector res and
        compute the elements of the linearized KKT system required for
        the solution phase of the algorithm.
        '''

        # Compute the digonal contribution to the dmat
        self.ddiag.x[:] = diag_fact
        if self.lb is not None:
            self.ddiag.x[:] += barrier/(x.x - self.lb.x)**2
        if self.ub is not None:
            self.ddiag.x[:] += barrier/(self.ub.x - x.x)**2

        # Assemble and factor the D matrix
        self.prob.assembleDMat(x, u, self.ddiag,
                               self.stress_barrier_scale*barrier, self.dmat)

        # Add the second derivative of the mass
        self.prob.addMass2ndDeriv(x, self.dmat)

        # Add any extra terms to the D-matrix
        self.prob.addExtraDMatTerms(x, lambd, u, psi, self.dmat,
                                    exact_flag=False)

        # Assemble the exact Hessian too, if the flag is activated
        if self.exact_hessian:
            self.dmat_exact.A[0][:] = self.dmat.A[0][:]
            self.prob.addExtraDMatTerms(x, lambd, u, psi, self.dmat_exact,
                                        exact_flag=True)

        # Set up the null-space method if required
        if self.lambd is not None:
            self.prob.setUpNullSpace(x, self.dmat)
        else:
            self.dmat.factor()

        # Assemble and factor the stiffness matrix
        self.prob.assembleKMat(x, self.kmat)
        self.kmat.factor()

        # Compute the residual from the first equation
        self.prob.computeMassDeriv(x, res.x)

        # Complete the design variable residual
        if self.lb is not None:
            res.x.x[:] -= barrier/(x.x - self.lb.x)
        if self.ub is not None:
            res.x.x[:] += barrier/(self.ub.x - x.x)

        # Add res.x += A(u)^{T}*psi
        self.prob.multAMatTransposeAdd(x, u, psi, res.x)

        # Add the contribution from the weighting constraints
        if self.lambd is not None:
            self.prob.multAwMatTransposeAdd(x, lambd, res.x)
            self.prob.computeWeightCon(x, res.lambd)

        # Compute the values of the second residual
        self.kmat.mult(psi, res.u)
        self.prob.addLogStressSumDeriv(x, u,
                                       self.stress_barrier_scale*barrier,
                                       res.x, res.u)

        # Compute the residuals of the third equations (equilibrium)
        self.kmat.mult(u, res.psi)
        res.psi.axpy(-1.0, self.f)

        return

    def mult(self, p, y):
        '''
        Compute the matrix-vector product with the matrix:

        [           D;  Aw^{T}; A(psi)^{T} + E^{T};  A(u)^{T} ]
        [          Aw;        ;                   ;           ]
        [ A(psi) +  E;        ;                  C;      K(x) ]
        [        A(u);        ;               K(x);           ]
        '''

        # Compute the matrix-vector product with the first system of
        # equations
        if self.exact_hessian:
            self.dmat_exact.mult(p.x, y.x)
        else:
            self.dmat.mult(p.x, y.x)
        self.prob.multAMatTransposeAdd(self.x, self.psi, p.u, y.x)
        self.prob.multAMatTransposeAdd(self.x, self.u, p.psi, y.x)

        # Multiply the off-diagonal terms
        if self.lambd is not None:
            self.prob.multAwMatTransposeAdd(self.x, p.lambd, y.x)
            y.lambd.zero()
            self.prob.multAwMatAdd(self.x, p.x, y.lambd)

        # Add the contribution from the second system of equations
        self.kmat.mult(p.psi, y.u)
        self.prob.multCMat(self.x, self.u, self.barrier,
                           p.u, self.utemp)
        y.u.axpy(1.0, self.utemp)
        self.prob.multAMatAdd(self.x, self.psi, p.x, y.u)

        # Add the off-diagonal contributions
        self.prob.multAddMatOffDiag(self.x, self.u,
                                    self.stress_barrier_scale*self.barrier,
                                    p.x, p.u, y.x, y.u)

        # Compute the product with the thrid system of equations
        self.kmat.mult(p.u, y.psi)
        self.prob.multAMatAdd(self.x, self.u, p.x, y.psi)

        return

    def applyPc(self, res, y):
        '''
        Compute the action of the preconditioner
        '''

        if self.pc_type == 'Biros':
            # Solve K(x)*psi = bu
            if self.pc_gmres_subspace_size > 0:
                fgmres(self.kmat, res.u, y.psi, self.Wd, self.Zd, rtol=1e-12)
            else:
                self.kmat.applyPc(res.u, y.psi)

            # Solve D*yx + Aw^{T}*yw = bx - A(u)^{T}*psi
            #       Aw*yx = bw
            y.x.zero()
            self.prob.multAMatTransposeAdd(self.x, self.u, y.psi, y.x)
            self.xtemp.copy(res.x)
            self.xtemp.axpy(-1.0, y.x)

            # Compute the step in (x, lambd) using the Null-space method
            if self.lambd is not None:
                self.prob.solveNullSpace(self.x, self.dmat,
                                         self.xtemp, res.lambd, y.x, y.lambd)
            else:
                self.dmat.applyPc(self.xtemp, y.x)

            # Solve K(x)*u = bpsi - A(psi)*yx
            self.utemp.copy(res.psi)
            self.utemp.scale(-1.0)
            self.prob.multAMatAdd(self.x, self.u, y.x, self.utemp)
            self.utemp.scale(-1.0)
            if self.pc_gmres_subspace_size > 0:
                fgmres(self.kmat, self.utemp, y.u, self.Wd, self.Zd, rtol=1e-12)
            else:
                self.kmat.applyPc(self.utemp, y.u)
        else:
            # Compute the step in (x, lambd) using the Null-space method
            if self.lambd is not None:
                self.prob.solveNullSpace(self.x, self.dmat,
                                         res.x, res.lambd, y.x, y.lambd)
            else:
                self.dmat.applyPc(res.x, y.x)

            # Solve: K(x)*yu = bpsi - A(u)*yx
            self.utemp.zero()
            self.prob.multAMatAdd(self.x, self.u, y.x, self.utemp)
            self.utemp.axpy(-1.0, res.psi)
            if self.pc_gmres_subspace_size > 0:
                fgmres(self.kmat, self.utemp, y.u, self.Wd, self.Zd, rtol=1e-12)
            else:
                self.kmat.applyPc(self.utemp, y.u)
            y.u.scale(-1.0)

            # Solve K(x)*ypsi = bu - A(psi)*yx - C*yu
            self.prob.multCMat(self.x, self.u, self.barrier,
                               y.u, self.utemp)
            self.prob.multAMatAdd(self.x, self.psi, y.x, self.utemp)
            self.utemp.axpy(-1.0, res.u)
            if self.pc_gmres_subspace_size > 0:
                fgmres(
                    self.kmat, self.utemp, y.psi, self.Wd, self.Zd, rtol=1e-12)
            else:
                self.kmat.applyPc(self.utemp, y.psi)
            y.psi.scale(-1.0)

        return

    def testKKTJacobian(self, dh=1e-6):
        '''
        Test the KKT Jacobian against the finite-difference method.
        '''

        # Duplicate the variables and copy them
        vec = KKTVec(self.x, self.lambd, self.u, self.psi)

        # Allocate additional temporary vectors for the computation
        fd = vec.duplicate()
        res = vec.duplicate()
        step = vec.duplicate()
        ans = vec.duplicate()

        # Test the indefinite term if required
        if self.exact_hessian:
            self.prob.testIndefTerm(self.x, self.u, self.psi, dh=1e-6)

        self.assembleKKTResMat(self.x, self.lambd, self.u, self.psi,
                               self.barrier, vec)

        for k in xrange(4):
            # Copy the values from the current iterate
            vec.x.copy(self.x)
            if self.lambd is not None:
                vec.lambd.copy(self.lambd)
            vec.u.copy(self.u)
            vec.psi.copy(self.psi)

            # Set the perturbation
            step.zero()
            perb = None
            if k == 0:
                perb = self.x.duplicate()
                perb.x = np.random.uniform(size=perb.x.shape)
                vec.x.axpy(-dh, perb)
                step.x.copy(perb)
            elif k == 1 and self.lambd is not None:
                perb = self.lambd.duplicate()
                perb.x = np.random.uniform(size=perb.x.shape)
                vec.lambd.axpy(-dh, perb)
                step.lambd.copy(perb)
            elif k == 2:
                perb = self.u.duplicate()
                perb.x = np.random.uniform(size=perb.x.shape)
                sparse.applyvecbcs(self.prob.bcnodes,
                                   self.prob.bcvars.T, perb.x.T)
                vec.u.axpy(-dh, perb)
                step.u.copy(perb)
            else:
                perb = self.psi.duplicate()
                perb.x = np.random.uniform(size=perb.x.shape)
                sparse.applyvecbcs(self.prob.bcnodes,
                                   self.prob.bcvars.T, perb.x.T)
                vec.psi.axpy(-dh, perb)
                step.psi.copy(perb)

            # Assemble the residual
            self.assembleKKTRes(vec.x, vec.lambd, vec.u, vec.psi,
                                self.barrier, res)

            # Copy the values from the current iterate
            vec.x.copy(self.x)
            if self.lambd is not None:
                vec.lambd.copy(self.lambd)
            vec.u.copy(self.u)
            vec.psi.copy(self.psi)

            # Perturb the variables in the forward direction
            if k == 0:
                vec.x.axpy(dh, perb)
            elif k == 1 and self.lambd is not None:
                vec.lambd.axpy(dh, perb)
            elif k == 2:
                vec.u.axpy(dh, perb)
            else:
                vec.psi.axpy(dh, perb)

            # Assemble the residual
            self.assembleKKTRes(vec.x, vec.lambd, vec.u, vec.psi,
                                self.barrier, fd)

            # Compute the residual
            fd.axpy(-1.0, res)
            fd.scale(0.5/dh)

            # Assemble the residuals once again so that we have the
            # Jacobian at the initial point
            self.assembleKKTRes(self.x, self.lambd, self.u, self.psi,
                                self.barrier, res)

            # Now check the result against the real product
            self.mult(step, ans)

            # Print the errors
            if k == 0:
                # print 'Error in the first equation'
                self.prob.printError(ans.x, fd.x)
            ans.axpy(-1.0, fd)

            # Print out the error in each component
            print 'Column %d'%(k)
            print 'Eq 1: ', np.sqrt(ans.x.dot(ans.x))
            if self.lambd is not None:
                print 'Eq 2: ', np.sqrt(ans.lambd.dot(ans.lambd))
            print 'Eq 3: ', np.sqrt(ans.u.dot(ans.u))
            print 'Eq 4: ', np.sqrt(ans.psi.dot(ans.psi))

        return

    def testKKTStep(self, step, res):
        '''
        Test to see that the given step satisfies the linearized
        '''

        ans = step.duplicate()
        self.mult(step, ans)
        ans.axpy(1.0, res)

        # Print out the error in each component
        print 'Residual'
        print 'Eq 1: ', np.sqrt(ans.x.dot(ans.x))
        if self.lambd is not None:
            print 'Eq 2: ', np.sqrt(ans.lambd.dot(ans.lambd))
        print 'Eq 3: ', np.sqrt(ans.u.dot(ans.u))
        print 'Eq 4: ', np.sqrt(ans.psi.dot(ans.psi))

        ans.zero()
        self.applyPc(step, ans)

        # Multiplication consistency check
        print 'Preconditioner check'
        print 'Eq 1: ', np.sqrt(ans.x.dot(ans.x))
        if self.lambd is not None:
            print 'Eq 2: ', np.sqrt(ans.lambd.dot(ans.lambd))
        print 'Eq 3: ', np.sqrt(ans.u.dot(ans.u))
        print 'Eq 4: ', np.sqrt(ans.psi.dot(ans.psi))

        self.applyPc(step, ans)

        # Multiplication consistency check
        print 'Preconditioner consistency check'
        print 'Eq 1: ', np.sqrt(ans.x.dot(ans.x))
        if self.lambd is not None:
            print 'Eq 2: ', np.sqrt(ans.lambd.dot(ans.lambd))
        print 'Eq 3: ', np.sqrt(ans.u.dot(ans.u))
        print 'Eq 4: ', np.sqrt(ans.psi.dot(ans.psi))

        return

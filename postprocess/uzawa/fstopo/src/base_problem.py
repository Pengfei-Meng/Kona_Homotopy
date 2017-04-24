import numpy as np

import fstopo.sparse as sparse

class FullSpaceProblem:
    '''
    The following is the base class for the full-space barrier
    methods.
    '''

    def createDesignVec(self):
        return None

    def createSolutionVec(self):
        return None

    def createWeightVec(self):
        return None

    def getForceVec(self):
        return None

    def getKMat(self):
        return None

    def getDMat(self):
        return None

    def computeMass(self, x):
        '''Compute the mass'''
        return 0.0

    def computeMassDeriv(self, x, rx):
        '''Compute the derivative of the mass'''
        return

    def addMass2ndDeriv(self, x, dmat):
        '''Compute the second derivative of the mass'''
        return

    def addExtraDMatTerms(self, x, lambd, u, psi, dmat, exact_flag=False):
        '''Add any extra desired terms to the Dmatrix'''
        return

    def assembleKMat(self, x, mat):
        '''
        Assemble the finite-element stiffness matrix
        '''
        return

    def multAMatAdd(self, x, u, px, yu):
        '''Compute the product: yu += A(u)*px'''
        return

    def multAMatTransposeAdd(self, x, u, pu, yx):
        '''Compute the product yx = A(u)^{T}*pu'''
        return

    def computeMaxStressStep(self, x, u, xstep, ustep, tau):
        '''
        Compute the maximum (or approximate maximum) step length along
        the given direction and back-off a fraction tau.
        '''
        return 0.0

    def computeLogStressSum(self, x, u):
        '''
        Compute the sum of the logarithm of the stress constraints
        '''
        return 0.0

    def addLogStressSumDeriv(self, x, u, barrier, rx, ru):
        '''
        Add the derivatives of the logarithm of the stress constraints
        w.r.t. x and u to the given residual term
        '''
        return

    def assembleDMat(self, x, u, diag, barrier, mat):
        '''
        Assemble the D matrix and add the given diagonal entries
        '''
        return

    def multAddMatOffDiag(self, x, u, barrier, xstep, ustep, rx, ru):
        '''
        Add the off-diagonal matrix-vector products with the
        C-matrices
        '''
        return

    def useWeightCon(self):
        '''Use/implement the weight constraints'''
        return False

    def computeWeightCon(self, x, rw):
        '''Compute the values of the weighting constraints.'''
        return

    def multAwMatAdd(self, x, px, rw):
        '''Compute the Jacobian-vector product'''
        return

    def multAwMatTransposeAdd(self, x, pw, rx):
        '''Compute the transpose Jacobian-vector product'''
        return

    def setUpNullSpace(self, x, dmat):
        '''At a design point, set up the null-space'''
        return

    def solveNullSpace(self, x, dmat, rx, rw, px, pw):
        '''Solve the null-space problem'''
        return

    def writeSolution(self, u, psi, x, filename='solution.dat'):
        '''Write out the solution'''
        return

    def printError(self, grad, fd, printtol=1e-4):
        '''
        Print the relative error between x1 and x2
        '''

        print '%3s %15s %15s %15s %15s'%(
            'Cmp', 'Grad', 'FD', 'Abs err', 'Rel err')

        if len(grad.x.shape) == 2:
            for i in xrange(grad.x.shape[0]):
                for j in xrange(grad.x.shape[1]):
                    relerr = 0.0
                    abserr = np.fabs((grad.x[i,j] - fd.x[i,j]))
                    if grad.x[i,j] != 0.0:
                        relerr = np.fabs((grad.x[i,j] - fd.x[i,j])/grad.x[i,j])

                    if relerr > printtol or abserr > printtol:
                        print '%3d %15.5e %15.5e %15.5e %15.5e'%(
                            i, grad.x[i,j], fd.x[i,j], abserr, relerr)
        else:
            for i in xrange(grad.x.shape[0]):
                relerr = 0.0
                abserr = np.fabs((grad.x[i] - fd.x[i]))
                if grad.x[i] != 0.0:
                    relerr = np.fabs((grad.x[i] - fd.x[i])/grad.x[i])

                if relerr > printtol or abserr > printtol:
                    print '%3d %15.5e %15.5e %15.5e %15.5e'%(
                        i, grad.x[i], fd.x[i], abserr, relerr)

        return

    def testAMatProducts(self, x, u, dh=1e-6, printtol=1e-6):
        '''
        Test the implementation of the A-products
        '''

        kmat = self.getKMat()

        # Create a temporary array
        utemp1 = u.duplicate()
        utemp2 = u.duplicate()

        # Assemble the stiffness matrix
        self.assembleKMat(x, kmat)
        kmat.mult(u, utemp1)

        # Copy the design variables
        xtemp1 = x.duplicate()
        xtemp1.copy(x)

        # Create a random perturbation
        xperb = x.duplicate()
        xperb.x[:] = np.random.uniform(size=xperb.x.shape)

        uperb = u.duplicate()
        uperb.x[:] = np.random.uniform(size=uperb.x.shape)
        sparse.applyvecbcs(self.bcnodes, self.bcvars.T, uperb.x.T)

        # Perturb the design variables and re-assemble the stiffness
        # matrix
        xtemp1.axpy(dh, xperb)
        self.assembleKMat(xtemp1, kmat)
        kmat.mult(u, utemp2)

        # Compute the finite-difference approximation
        utemp2.axpy(-1.0, utemp1)
        utemp2.scale(1.0/dh)

        utemp1.zero()
        self.multAMatAdd(x, u, xperb, utemp1)
        dotAx = utemp1.dot(uperb)

        print 'AMat-product error'
        self.printError(utemp1, utemp2, printtol=printtol)

        # Now, test the transpose product. Instead of checking all
        # components, this test ensures that the product is consistent
        # with the transpose product.
        xtemp2 = x.duplicate()
        xtemp2.zero()
        self.multAMatTransposeAdd(x, u, uperb, xtemp2)
        dotATx = xtemp2.dot(xperb)

        print 'AMat transpose-consistency check'
        print 'dotAx = ', dotAx
        print 'dotATx = ', dotATx
        print '%15s %15s %15s %15s'%(
            'Forward', 'Transpose', 'Abs err', 'Rel err')
        print '%15.5e %15.5e %15.5e %15.5e'%(
            dotAx, dotATx, np.fabs(dotAx - dotATx),
            np.fabs((dotAx - dotATx)/dotAx))

        return

    def testLogStressSum(self, x, u, dh=1e-5, printtol=1e-5):
        '''
        Test that the log sum of the stress constraints is implemented
        correctly
        '''

        # Create the temporary vectors required for this study
        xtemp = x.duplicate()
        xtemp.copy(x)
        xgrad = x.duplicate()
        xfd = x.duplicate()

        utemp = u.duplicate()
        utemp.copy(u)
        ugrad = u.duplicate()
        ufd = u.duplicate()

        # Evaluate the sum of the logaraithms
        logsum1 = self.computeLogStressSum(x, u)

        # Evaluate the gradient
        xgrad.zero()
        ugrad.zero()
        self.addLogStressSumDeriv(x, u, -1.0, xgrad, ugrad)

        # Compute the derivative w.r.t. the design variables
        for i in xrange(x.x.shape[0]):
            for j in xrange(x.x.shape[1]):
                xtemp.x[i,j] = x.x[i,j] + dh
                logsum2 = self.computeLogStressSum(xtemp, u)
                xfd.x[i,j] = (logsum2 - logsum1)/dh
                xtemp.x[i,j] = x.x[i,j]

        print 'Log sum stress error in design components'
        print 'dh = %15.5e'%(dh)
        self.printError(xgrad, xfd, printtol=printtol)

        # Adjust the step size so that it reflects the difference
        # in scaling between the displacements/design variables
        # uav = np.sum(np.fabs(u.x))/(u.x.shape[0]*u.x.shape[1])
        # dh = uav*dh

        # Compute the derivative w.r.t. the displacements
        for i in xrange(u.x.shape[0]):
            for j in xrange(u.x.shape[1]):
                utemp.x[i,j] = u.x[i,j] + dh
                logsum2 = self.computeLogStressSum(x, utemp)
                utemp.x[i,j] = u.x[i,j] - dh
                logsum3 = self.computeLogStressSum(x, utemp)
                ufd.x[i,j] = 0.5*(logsum2 - logsum3)/dh
                utemp.x[i,j] = u.x[i,j]

        print 'Log sum stress error in displacement components'
        print 'dh = %15.5e'%(dh)
        self.printError(ugrad, ufd, printtol=printtol)

        return

    def testMass(self, x, dh=1e-5, printtol=1e-5):
        '''
        Test that the log sum of the stress constraints is implemented
        correctly
        '''

        # Create the temporary vectors required for this study
        xtemp = x.duplicate()
        xtemp.copy(x)
        xgrad = x.duplicate()
        xfd = x.duplicate()

        # Evaluate the mass and its derivative
        mass = self.computeMass(x)
        self.computeMassDeriv(x, xgrad)

        # Compute the derivative w.r.t. the design variables
        for i in xrange(x.x.shape[0]):
            for j in xrange(x.x.shape[1]):
                xtemp.x[i,j] = x.x[i,j] + dh
                mass2 = self.computeMass(xtemp)
                xfd.x[i,j] = (mass2 - mass)/dh
                xtemp.x[i,j] = x.x[i,j]

        print 'Mass derivative componet error'
        print 'dh = %15.5e'%(dh)
        self.printError(xgrad, xfd, printtol=printtol)

        # Now test the second derivatives of the mass
        xperb = x.duplicate()
        xperb.x = np.random.uniform(size=xperb.x.shape)

        # Compute the product of the C-matrix with the perturbation
        dmat = self.getDMat()
        dmat.A[0][:] = 0.0
        self.addMass2ndDeriv(x, dmat)
        xgrad = x.duplicate()
        dmat.mult(xperb, xgrad)

        # Duplicate the vectors required and compute an approximate
        # product using centeral differencing
        xtemp2 = x.duplicate()

        # Evaluate at the point (u - dh*uperb)
        xtemp.copy(x)
        xtemp.axpy(-dh, xperb)
        self.computeMassDeriv(xtemp, xtemp2)

        # Evaluate at the point (u + dh*uperb)
        xtemp.copy(x)
        xtemp.axpy(dh, xperb)
        self.computeMassDeriv(xtemp, xfd)

        # Compute the approximation
        xfd.axpy(-1.0, xtemp2)
        xfd.scale(0.5/dh)

        print 'Mass 2nd derivative test'
        self.printError(xgrad, xfd, printtol=printtol)

        return

    def testDiagonalMat(self, x, u, barrier=1.25, dh=1e-6, printtol=1e-6):
        '''
        Test the product of a vector with the C-matrix
        '''

        # Create a perturbation vector
        uperb = u.duplicate()
        uperb.x[:] = np.random.uniform(size=uperb.x.shape)
        sparse.applyvecbcs(self.bcnodes, self.bcvars.T, uperb.x.T)

        # Compute the product of the C-matrix with the perturbation
        utemp = u.duplicate()
        ugrad = u.duplicate()
        self.multCMat(x, u, barrier, uperb, ugrad)

        # Duplicate the vectors required and compute an approximate
        # product using centeral differencing
        ufd = u.duplicate()
        utemp2 = u.duplicate()
        xtemp = x.duplicate()

        # Evaluate at the point (u - dh*uperb)
        utemp.copy(u)
        utemp.axpy(-dh, uperb)
        self.addLogStressSumDeriv(x, utemp, barrier, xtemp, utemp2)

        # Evaluate at the point (u + dh*uperb)
        utemp.copy(u)
        utemp.axpy(dh, uperb)
        self.addLogStressSumDeriv(x, utemp, barrier, xtemp, ufd)

        # Compute the approximation
        ufd.axpy(-1.0, utemp2)
        ufd.scale(0.5/dh)

        print 'CMat-product test'
        self.printError(ugrad, ufd, printtol=printtol)

        # Create a perturbation vector in the design variables
        xperb = x.duplicate()
        xperb.x = np.random.uniform(size=xperb.x.shape)

        # Compute the product of the C-matrix with the perturbation
        dmat = self.getDMat()
        xtemp.zero()
        self.assembleDMat(x, u, xtemp, barrier, dmat)
        xgrad = x.duplicate()
        dmat.mult(xperb, xgrad)

        # Duplicate the vectors required and compute an approximate
        # product using centeral differencing
        xfd = x.duplicate()
        xtemp2 = x.duplicate()

        # Evaluate at the point (u - dh*uperb)
        xtemp.copy(x)
        xtemp.axpy(-dh, xperb)
        self.addLogStressSumDeriv(xtemp, u, barrier, xtemp2, utemp)

        # Evaluate at the point (u + dh*uperb)
        xtemp.copy(x)
        xtemp.axpy(dh, xperb)
        self.addLogStressSumDeriv(xtemp, u, barrier, xfd, utemp)

        # Compute the approximation
        xfd.axpy(-1.0, xtemp2)
        xfd.scale(0.5/dh)

        print 'DMat-product test'
        self.printError(xgrad, xfd, printtol=printtol)

        return

    def testOffDiagMat(self, x, u, dh=1e-6, printtol=1e-6):
        '''
        Test the product of a vector with the C-matrix
        '''

        # Create perturbation vectors
        uperb = u.duplicate()
        uperb.x[:] = np.random.uniform(size=uperb.x.shape)
        sparse.applyvecbcs(self.bcnodes, self.bcvars.T, uperb.x.T)

        xperb = x.duplicate()
        xperb.x = np.random.uniform(size=xperb.x.shape)

        # Compute the off-diagonal result
        ugrad = u.duplicate()
        xgrad = x.duplicate()
        self.multAddMatOffDiag(x, u, 1.0, xperb, uperb, xgrad, ugrad)

        # Compute the product using centeral differencing
        ufd = u.duplicate()
        utemp = u.duplicate()
        xtemp = x.duplicate()
        xout = x.duplicate()

        # Evaluate at the point (x - dh*xperb)
        xtemp.copy(x)
        xtemp.axpy(-dh, xperb)
        self.addLogStressSumDeriv(xtemp, u, 1.0, xout, utemp)

        # Evaluate at the point (x + dh*xperb)
        xtemp.copy(x)
        xtemp.axpy(dh, xperb)
        self.addLogStressSumDeriv(xtemp, u, 1.0, xout, ufd)

        # Compute the approximation
        ufd.axpy(-1.0, utemp)
        ufd.scale(0.5/dh)

        print 'Off-diagonal product w.r.t. x'
        self.printError(ugrad, ufd, printtol=printtol)

        # Now, compute the product with uperb
        xfd = x.duplicate()
        xtemp = x.duplicate()
        uout = u.duplicate()

        # Evaluate at the point (u - dh*uperb)
        utemp.copy(u)
        utemp.axpy(-dh, uperb)
        self.addLogStressSumDeriv(x, utemp, 1.0, xtemp, uout)

        # Evaluate at the point (u + dh*uperb)
        utemp.copy(u)
        utemp.axpy(dh, uperb)
        self.addLogStressSumDeriv(x, utemp, 1.0, xfd, uout)

        # Compute the approximation
        xfd.axpy(-1.0, xtemp)
        xfd.scale(0.5/dh)

        print 'Off-diagonal-product w.r.t. u'
        self.printError(xgrad, xfd, printtol=printtol)

        return

    def testWeightCon(self, x, dh=1e-6, printtol=1e-6):
        '''
        Test the implementation of the weighting constraints
        '''

        # Create a perturbation vector
        xperb = x.duplicate()
        xperb.x = np.random.uniform(size=xperb.x.shape)

        # Create the temporary constraint vectors
        c1 = self.createWeightVec()
        c2 = c1.duplicate()
        rw = c1.duplicate()

        # Create a perturbation to w
        wperb = c1.duplicate()
        wperb.x = np.random.uniform(size=wperb.x.shape)

        # Compute the exact value
        self.computeWeightCon(x, c1)
        self.multAwMatAdd(x, xperb, rw)
        dotAx = rw.dot(wperb)

        # Perturb the design variables and evaluate the constraints again
        xtemp = x.duplicate()
        xtemp.copy(x)
        xtemp.axpy(dh, xperb)
        self.computeWeightCon(xtemp, c2)

        # Finish computing the forward difference approximation
        c2.axpy(-1.0, c1)
        c2.scale(1.0/dh)

        print 'Weighting constraint Jacobian test'
        self.printError(rw, c2, printtol=printtol)

        xtemp.zero()
        self.multAwMatTransposeAdd(x, wperb, xtemp)
        dotATx = xtemp.dot(xperb)

        print 'AwMat transpose-consistency check'
        print 'dotAx = ', dotAx
        print 'dotATx = ', dotATx
        print '%15s %15s %15s %15s'%(
            'Forward', 'Transpose', 'Abs err', 'Rel err')
        print '%15.5e %15.5e %15.5e %15.5e'%(
            dotAx, dotATx, np.fabs(dotAx - dotATx),
            np.fabs((dotAx - dotATx)/dotAx))

        return

    def testNullSpace(self, x, u, barrier=1.25, printtol=1e-6):
        '''
        Test that the null space method is implemented correctly
        '''

        # Create the vector that will be added to the diagonal of the
        # D-matrix
        xtemp = x.duplicate()
        xtemp.x[:] = 100.0

        # Assemble the D-matrix
        dmat = self.getDMat()
        self.assembleDMat(x, u, xtemp, barrier, dmat)

        # Set up the null space method
        self.setUpNullSpace(x, dmat)

        # Set a random vector for the right-hand-side
        rx = x.duplicate()
        rx.x = np.random.uniform(size=x.x.shape)
        px = x.duplicate()

        # First, make is so that the right-hand-side is zero
        rw = self.createWeightVec()
        rw.x = np.random.uniform(size=rw.x.shape)
        pw = rw.duplicate()

        # Solve the system of equations
        self.solveNullSpace(x, dmat, rx, rw, px, pw)

        # Now, check that the equations are satisfied
        rx1 = rx.duplicate()
        rw1 = rw.duplicate()

        # Form the right-hand-side for x from the solution
        dmat.mult(px, rx1)
        self.multAwMatTransposeAdd(x, pw, rx1)

        print 'Error in D*px + Aw^{T}*pw (Note this may be inexact)'
        self.printError(rx, rx1, printtol=printtol)

        # Form the right-hand-side for lambda from the solution
        rw1.zero()
        self.multAwMatAdd(x, px, rw1)

        print 'Error in the equation Aw*px'
        self.printError(rw, rw1, printtol=printtol)

        return

    def test(self, x, u, dh=5e-7, printtol=1e-5):
        '''
        Run all of the tests
        '''

        self.testMass(x, dh=dh, printtol=printtol)
        self.testAMatProducts(x, u, dh=dh, printtol=printtol)
        self.testLogStressSum(x, u, dh=dh, printtol=printtol)
        self.testDiagonalMat(x, u, dh=dh, printtol=printtol)
        self.testOffDiagMat(x, u, dh=dh, printtol=printtol)

        if self.useWeightCon():
            self.testWeightCon(x, dh=dh, printtol=printtol)
            self.testNullSpace(x, u)

        return

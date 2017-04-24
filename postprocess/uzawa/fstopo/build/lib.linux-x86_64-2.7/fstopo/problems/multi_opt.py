import numpy as np
import os

import fstopo.multi as multi
import fstopo.sparse as sparse
from fstopo.linalg import MGVec, fgmres
from fstopo.base_problem import FullSpaceProblem

class WeightConVec(object):
    '''
    A vector class for the weighting constraints
    '''
    def __init__(self, n):
        self.x = np.zeros(n)
        return

    def duplicate(self):
        '''
        Duplicate the vector, creating a new one
        '''
        return WeightConVec(len(self.x))

    def copy(self, vec):
        '''
        Copy the values from the given vector
        '''
        self.x[:] = vec.x[:]
        return

    def dot(self, vec):
        '''
        Compute the dot-product of two vectors
        '''
        return self.x.dot(vec.x)

    def zero(self):
        '''
        Zero the values in the array
        '''
        self.x[:] = 0.0
        return

    def scale(self, alpha):
        '''
        Scale the values
        '''
        self.x[:] *= alpha
        return

    def axpy(self, alpha, vec):
        '''
        Compute self = self + alpha*vec
        '''
        self.x[:] += alpha*vec.x
        return

    def infty(self):
        '''
        Compute the infinity norm
        '''
        return max(self.x.max(), -self.x.min())

class MultiProblem(FullSpaceProblem):
    '''
    The following is the base class for the full-space barrier
    methods.
    '''

    def __init__(self, conn, X, tconn, tweights, elems,
                 kmat, dmat, rho, Cmats, qval, qxval, h, G, epsilon,
                 bcnodes, bcvars, zdz_gmres_size=0,
                 force=None, ply_problem=False):
        '''
        Record all the data that's required for optimization
        '''

        self.mscale = 1.0
        self.mpower = -1.0

        self.xi = 0.0
        self.eta = 0.0

        # Store the failure and parametrization types
        self.ptype = 1
        self.ftype = 1

        # Copy the connectivity of the mesh and the nodal locations
        self.conn = conn
        self.X = X

        # Copy over the elements in which to apply the constraints
        self.elems = elems

        # Copy over the filter connectivity
        self.tconn = tconn
        self.tweights = tweights

        # Copy over the material data
        self.rho = rho
        self.Cmats = Cmats
        self.qval = qval
        self.qxval = qxval

        # Record whether to use ply-angle outputs
        self.ply_problem = ply_problem
        if self.ply_problem:
            # Determine the angles
            nmats = self.Cmats.shape[0]
            self.angles = np.pi*np.linspace(-90, 90, nmats+1)[1:]/180.0
        else:
            # Else assign some array
            self.angles = []
            nmats = self.Cmats.shape[0]
            for i in xrange(nmats):
                self.angles.append(None)

        # Copy over the allocated matrices
        nmats = self.Cmats.shape[0]
        self.kmat = kmat
        self.cmat = kmat.duplicate()
        self.dmat = dmat
        self.zdzmat = dmat.duplicate(nb=nmats)

        # Copy over the failure criteria data
        self.h = h
        self.G = G
        self.epsilon = epsilon

        # Record the boundary conditions
        self.bcnodes = bcnodes
        self.bcvars = bcvars

        # Set the force vector
        self.force = force

        # Allocate the initial Lagrange multipliers
        ne = self.tconn.shape[0]
        self.lambd = WeightConVec(ne)

        # Allocate the null-space for the constraints
        self.Z = np.zeros((ne, nmats, 1+nmats))
        self.Aw = np.zeros((ne, 1+nmats))
        self.Adiag = np.zeros(ne)

        # Temporary vectors for the null-space method
        self.zrhs = MGVec(n=ne, nb=nmats)
        self.zupdate = MGVec(n=ne, nb=nmats)
        self.xtemp = self.createDesignVec()

        self.pc_gmres_subspace_size = 0
        self.Wd = []
        self.Zd = []

        return

    def setUpGMRESSubspace(self, size):
        '''
        Set up a GMRES subspace to use with the null-space method
        '''

        # Allocate vectors for the dmat solution
        self.pc_gmres_subspace_size = size
        if self.pc_gmres_subspace_size > 0:
            self.Wd = [self.zrhs.duplicate()]
            self.Zd = []
            for i in xrange(self.pc_gmres_subspace_size):
                self.Wd.append(self.zrhs.duplicate())
                self.Zd.append(self.zrhs.duplicate())

        return

    def testIndefTerm(self, x, u, psi, dh=1e-6, printtol=1e-12):
        '''
        Test the term d2/dx^2(u^{T}K(x)*psi)
        '''

        # Zero the entries in the D-matrix
        dmat = self.getDMat()
        dmat.A[0][:] = 0.0
        exact = True
        multi.adddmatindef(exact, self.ptype, self.conn.T, self.X.T,
                           u.x.T, psi.x.T,
                           self.tconn.T, self.tweights.T, x.x.T,
                           self.qval, self.qxval, self.Cmats.T,
                           dmat.rowp[0], dmat.cols[0], dmat.A[0].T)

        xperb = x.duplicate()
        xperb.x = np.random.uniform(size=xperb.x.shape)

        # Compute the off-diagonal result
        xgrad = x.duplicate()
        dmat.mult(xperb, xgrad)

        # Compute the product using centeral differencing
        xfd = x.duplicate()
        xtemp = x.duplicate()
        xtemp2 = x.duplicate()

        # Evaluate at the point (x - dh*xperb)
        xtemp.copy(x)
        xtemp.axpy(-dh, xperb)
        xtemp2.zero()
        self.multAMatTransposeAdd(xtemp, u, psi, xtemp2)

        # Evaluate at the point (x + dh*xperb)
        xtemp.copy(x)
        xtemp.axpy(dh, xperb)
        xfd.zero()
        self.multAMatTransposeAdd(xtemp, u, psi, xfd)

        # Compute the approximation
        xfd.axpy(-1.0, xtemp2)
        xfd.scale(0.5/dh)

        print 'Indefinite contributions to the matrix w.r.t. x'
        self.printError(xgrad, xfd, printtol=printtol)

        return

    def createDesignVec(self):
        return MGVec(n=self.conn.shape[0],
                     nb=(1+self.Cmats.shape[0]))

    def createSolutionVec(self):
        return MGVec(n=self.X.shape[0], nb=2)

    def createWeightVec(self):
        return WeightConVec(n=self.conn.shape[0])

    def getForceVec(self):
        return self.force

    def getKMat(self):
        return self.kmat

    def getCMat(self):
        return self.cmat

    def getDMat(self):
        return self.dmat

    def computeMass(self, x):
        '''Compute the mass'''

        mass = multi.computemass(self.conn.T, self.X.T,
                                 self.tconn.T, self.tweights.T, x.x.T,
                                 self.rho)
        return mass

    def computeMassDeriv(self, x, rx):
        '''Compute the derivative of the mass'''
        rx.zero()
        multi.addmassderiv(self.conn.T, self.X.T,
                           self.tconn.T, self.tweights.T, x.x.T,
                           self.rho, rx.x.T)
        return

    def addMass2ndDeriv(self, x, dmat):
        '''Add the second derivative of the mass'''

        # Add the second derivative of the mass
        multi.addmass2ndderiv(
            self.conn.T, self.X.T, self.tconn.T, self.tweights.T, x.x.T,
            self.rho, dmat.rowp[0], dmat.cols[0], dmat.A[0].T)
        return

    def addExtraDMatTerms(self, x, lambd, u, psi, dmat, exact_flag=False):
        '''
        Add any extra terms that may or may not be important
        '''

        self.xtemp.zero()
        ne = self.tconn.shape[0]
        if exact_flag:
            # Add the terms from the Hessian of the term (lambd^{T}*cw(x))
            for i in xrange(ne):
                self.xtemp.x[i,1:] = 2.0*lambd.x[i]/x.x[i,1:]**3
        else:
            for i in xrange(ne):
                if lambd.x[i] > 0.0:
                    self.xtemp.x[i,1:] = 2.0*lambd.x[i]/x.x[i,1:]**3
        sparse.adddiagonal(
            self.xtemp.x.T, dmat.rowp[0], dmat.cols[0], dmat.A[0].T)

        # Add the potential indefinite term from the second
        # derivatives of the product of the Lagrange multiplier vector
        # with the governing equation: psi^{T}*(K(x)*u - f)
        multi.adddmatindef(exact_flag, self.ptype, self.conn.T, self.X.T,
                           psi.x.T, u.x.T,
                           self.tconn.T, self.tweights.T, x.x.T,
                           self.qval, self.qxval, self.Cmats.T,
                           dmat.rowp[0], dmat.cols[0], dmat.A[0].T)

        return

    def computeMaxStressStep(self, x, u, xstep, ustep, tau):
        '''
        Compute the maximum step length before violating the
        stress-constraint boundary
        '''

        # Compute the maximum step length from the stress
        alpha = multi.computemaxstep(
            self.ftype, self.elems, self.xi, self.eta, self.conn.T,
            self.X.T, u.x.T, ustep.x.T, self.tconn.T, self.tweights.T,
            x.x.T, xstep.x.T, self.epsilon, self.h.T, self.G.T, tau)

        return alpha

    def computeLogStressSum(self, x, u):
        '''
        Compute the sum of the logarithms of the stress constrains.
        '''
        logsum = multi.computelogstresssum(
            self.ftype, self.elems, self.xi, self.eta, self.conn.T,
            self.X.T, u.x.T, self.tconn.T, self.tweights.T, x.x.T,
            self.epsilon, self.h.T, self.G.T)

        return logsum

    def addLogStressSumDeriv(self, x, u, barrier, rx, ru):
        '''
        Add the derivative of the following term to the residual:
        .        - barrier*log(det(W))
        '''
        multi.addlogstresssumderiv(self.ftype, self.elems, self.xi, self.eta,
                                   self.conn.T, self.X.T, u.x.T,
                                   self.tconn.T, self.tweights.T, x.x.T,
                                   self.epsilon, self.h.T, self.G.T,
                                   barrier, rx.x.T, ru.x.T)

        # Apply the boundary conditions
        sparse.applyvecbcs(self.bcnodes, self.bcvars.T, ru.x.T)

        return

    def assembleKMat(self, x, mat):
        '''
        Assemble the finite-element stiffness matrix
        '''

        # Compute the stiffness matrix on the finest level
        multi.computekmat(self.ptype, self.conn.T, self.X.T,
                          self.tconn.T, self.tweights.T, x.x.T,
                          self.qval, self.qxval, self.Cmats.T,
                          mat.rowp[0], mat.cols[0], mat.A[0].T)

        # Apply the boundary conditions
        ident = True
        sparse.applymatbcs(mat.bcnodes[0], mat.bcvars[0].T,
                           mat.rowp[0], mat.cols[0], mat.A[0].T, ident)

        return

    def assembleDMat(self, x, u, diag, barrier, mat):
        '''
        Assemble the D matrixFalse
        '''
        # Compute the D-matrix
        multi.computedmat(
            self.ftype, self.elems, self.xi, self.eta, self.conn.T,
            self.X.T, u.x.T, self.tconn.T, self.tweights.T, x.x.T,
            self.epsilon, self.h.T, self.G.T, mat.rowp[0],
            mat.cols[0], mat.A[0].T)

        # Scale the result by the barrier parameter
        mat.A[0] *= barrier
        sparse.adddiagonal(diag.x.T, mat.rowp[0], mat.cols[0], mat.A[0].T)

        return

    def multCMat(self, x, u, barrier, pu, ru):
        '''
        Compute the product of an input vector and the second
        derivative matrix
        '''

        multi.computecmatproduct(self.ftype, self.elems, self.xi, self.eta,
                                 self.conn.T, self.X.T, u.x.T,
                                 self.tconn.T, self.tweights.T, x.x.T,
                                 self.epsilon, self.h.T, self.G.T,
                                 pu.x.T, ru.x.T)
        ru.x *= barrier
        sparse.applyvecbcs(self.bcnodes, self.bcvars.T, ru.x.T)

        return

    def assembleCMat(self, x, u, diag, barrier, mat):
        '''
        Assemble the matrix of second derivatives
        '''

        # Compute the stiffness matrix on the finest level
        multi.computecmat(self.ftype, self.elems, self.xi, self.eta,
                          self.conn.T, self.X.T, u.x.T,
                          self.tconn.T, self.tweights.T, x.x.T,
                          self.epsilon, self.h.T, self.G.T,
                          mat.rowp[0], mat.cols[0], mat.A[0].T)

        # Scale the matrix by the barrier parameter
        mat.A[0] *= barrier

        # Add the diagonal compnents
        sparse.adddiagonal(diag.x.T, mat.rowp[0], mat.cols[0], mat.A[0].T)

        # Apply the boundary conditions
        ident = True
        sparse.applymatbcs(mat.bcnodes[0], mat.bcvars[0].T,
                           mat.rowp[0], mat.cols[0], mat.A[0].T, ident)

        return

    def multAMatAdd(self, x, u, px, yu):
        '''
        Compute the product: yu += A(u)*px
        '''
        multi.addamatproduct(self.ptype, self.conn.T, self.X.T, u.x.T,
                             self.tconn.T, self.tweights.T, x.x.T, px.x.T,
                             self.qval, self.qxval, self.Cmats.T, yu.x.T)
        sparse.applyvecbcs(self.bcnodes, self.bcvars.T, yu.x.T)
        return

    def multAMatTransposeAdd(self, x, u, pu, yx):
        '''
        Compute the product yx = A(u)^{T}*pu
        '''
        multi.addamattransposeproduct(
            self.ptype, self.conn.T, self.X.T, u.x.T, pu.x.T,
            self.tconn.T, self.tweights.T, x.x.T, self.qval, self.qxval,
            self.Cmats.T, yx.x.T)

        return

    def multAddMatOffDiag(self, x, u, barrier, xstep, ustep, rx, ru):
        '''
        Add the off-diagonal matrix-vector products with the
        C-matrices
        '''
        # Compute the off-diagonal products
        multi.addcmatoffdiagproduct(
            self.ftype, self.elems, self.xi, self.eta, self.conn.T,
            self.X.T, u.x.T, ustep.x.T, self.tconn.T, self.tweights.T,
            x.x.T, xstep.x.T, self.epsilon, self.h.T, self.G.T,
            barrier, rx.x.T, ru.x.T)

        # Apply the boundary conditions
        sparse.applyvecbcs(self.bcnodes, self.bcvars.T, ru.x.T)

        return

    def useWeightCon(self):
        '''Use the weighting constraints'''
        return True

    def computeWeightCon(self, x, rw):
        '''
        Compute the values of the weighting constraints. Note that
        these constraints are equalities and do not depend on the
        design variables.
        '''
        multi.computeweightcon(x.x.T, rw.x)
        return

    def multAwMatAdd(self, x, px, rw):
        '''
        Compute the product of the Jacobian of the weighting
        constraints and a design vector.
        '''
        multi.multawmatadd(x.x.T, px.x.T, rw.x)
        return

    def multAwMatTransposeAdd(self, x, pw, rx):
        '''
        Compute the transpose product of the Jacobian of the weighting
        constraints and a Lagrange multiplier vector.
        '''
        multi.multawmattransposeadd(x.x.T, pw.x, rx.x.T)
        return

    def setUpNullSpace(self, x, dmat):
        '''
        Given the second derivative matrix dmat, and the design
        variables, set up the null space method at the current point.
        '''

        # Compute the null space matrix Z
        multi.computenullspace(x.x.T, self.Adiag, self.Z.T, self.Aw.T)

        # Compute the product Z^{T}*D*Z
        sparse.computeinnernull(self.Z.T,
                                dmat.rowp[0], dmat.cols[0], dmat.A[0].T,
                                self.zdzmat.A[0].T)

        # Factor the matrix
        self.zdzmat.factor()

        return

    def solveNullSpace(self, x, dmat, rx, rw, px, pw):
        '''
        Use the null-space method to compute the step px and pw.

        [  D;  Aw^T ][ px ] = [ rx ]
        [ Aw;       ][ pw ] = [ rw ]

        Adiag = Aw*Aw^{T}
        px = Z*u + Aw^{T}*Adiag^{-1}*rw
        pw = Adiag^{-1}*Aw*(rx - D*p)
        '''

        # Compute zrhs = Z^{T}*(rx - D*Aw^{T}*Adiag^{-1}*rw)
        multi.computenullrhs(self.Adiag, self.Z.T, self.Aw.T,
                             dmat.rowp[0], dmat.cols[0], dmat.A[0].T,
                             rx.x.T, rw.x, self.zrhs.x.T)

        # Compute u = [Z^{T}*D*Z]^{-1}*zrhs
        if self.pc_gmres_subspace_size > 0:
            # Solve for the update using preconditioned GMRES
            fgmres(self.zdzmat, self.zrhs, self.zupdate,
                   self.Wd, self.Zd, rtol=1e-12, print_flag=0)
        else:
            self.zdzmat.applyPc(self.zrhs, self.zupdate)

        # Compute the solution (px, pw) given the inexact solution
        multi.computenullsolution(self.Adiag, self.Z.T, self.Aw.T,
                                  self.zupdate.x.T, rw.x, px.x.T)

        # Compute rw = Aw*(D*p - rx)
        dmat.mult(px, self.xtemp)
        self.xtemp.axpy(-1.0, rx)
        pw.zero()
        self.multAwMatAdd(x, self.xtemp, pw)

        # Finish the computation and compute: pw = Adiag^{-1}*Aw*(rx - D*p)
        pw.x[:] = -pw.x/self.Adiag

        return

    def writeTikzStressFile(self, stress, filename='solution.tex'):
        '''
        Write out a .tikz file for the stresses
        '''

        # The blue-yellow-red color map from tecplot
        rgbbreak = np.linspace(0, 1, 7)
        rgbvals = [
            [69, 145, 224, 255, 254, 252, 215],
            [117, 191, 243, 255, 224, 141, 48],
            [180, 219, 248, 191, 144, 89, 39]]

        # Create the initial part of the tikz string
        tikz = '\\documentclass{article}\n'
        tikz += '\\usepackage[usenames,dvipsnames]{xcolor}\n'
        tikz += '\\usepackage{tikz}\n'
        tikz += '\\usepackage[active,tightpage]{preview}\n'
        tikz += '\\PreviewEnvironment{tikzpicture}\n'
        tikz += '\\setlength\PreviewBorder{5pt}%\n\\begin{document}\n'
        tikz += '\\begin{figure}\n\\begin{tikzpicture}[x=0.25cm, y=0.25cm]\n'
        tikz += '\\sffamily\n'

        # Extract the stress constraints to plot - only the minimum
        maxs = np.min(stress, axis=1)
        smax = min(2.0, np.max(maxs))
        smin = 0.0 # np.min(maxs)

        for i in xrange(self.conn.shape[0]):
            # Determine the color to use
            tval = (maxs[i] - smin)/(smax - smin)

            if tval <= 0.0:
                r = rgbvals[0][0]
                g = rgbvals[1][0]
                b = rgbvals[2][0]
            elif tval >= 1.0:
                r = rgbvals[0][-1]
                g = rgbvals[1][-1]
                b = rgbvals[2][-1]
            else:
                for j in xrange(6):
                    if tval >= rgbbreak[j] and tval <= rgbbreak[j+1]:
                        u = (tval - rgbbreak[j])/(rgbbreak[j+1] - rgbbreak[j])
                        r = (1.0 - u)*rgbvals[0][j] + u*rgbvals[0][j+1]
                        g = (1.0 - u)*rgbvals[1][j] + u*rgbvals[1][j+1]
                        b = (1.0 - u)*rgbvals[2][j] + u*rgbvals[2][j+1]
                        break

            tikz += '\\definecolor{mycolor}{RGB}{%d,%d,%d}'%(
                int(r), int(g), int(b))
            tikz += '\\fill[mycolor] (%f, %f) -- (%f, %f) -- '%(
                self.X[self.conn[i,0]-1,0], self.X[self.conn[i,0]-1,1],
                self.X[self.conn[i,1]-1,0], self.X[self.conn[i,1]-1,1])
            tikz += '(%f, %f) -- (%f, %f) -- cycle;\n'%(
                self.X[self.conn[i,3]-1,0], self.X[self.conn[i,3]-1,1],
                self.X[self.conn[i,2]-1,0], self.X[self.conn[i,2]-1,1])

        tikz += '\\end{tikzpicture}\\end{figure}\\end{document}\n'

        # Write the solution file
        fp = open(filename, 'w')
        if fp:
            fp.write(tikz)
            fp.close()

        return

    def writeTikzFile(self, x, filename='solution.tex'):
        '''
        Use the design variables to write out a .tikz file
        '''

        # Determine the design variable values
        xval = x.x
        xdv = np.zeros(xval.shape)

        for i in xrange(self.tconn.shape[0]):
            for j in xrange(self.tconn.shape[1]):
                if self.tconn[i,j] > 0:
                    xdv[i,:] += self.tweights[i,j]/xval[self.tconn[i,j]-1,:]

        # Create the initial part of the tikz string
        tikz = '\\documentclass{article}\n'
        tikz += '\\usepackage[usenames,dvipsnames]{xcolor}\n'
        tikz += '\\usepackage{tikz}\n'
        tikz += '\\usepackage[active,tightpage]{preview}\n'
        tikz += '\\PreviewEnvironment{tikzpicture}\n'
        tikz += '\\setlength\PreviewBorder{5pt}%\n\\begin{document}\n'
        tikz += '\\begin{figure}\n\\begin{tikzpicture}[x=0.25cm, y=0.25cm]\n'
        tikz += '\\sffamily\n'

        # Set the range of thickness values
        tmin = min(xdv[:,0])-1e-2
        tmax = max(xdv[:,0])+1e-2

        nmat = self.Cmats.shape[0]

        if self.ply_problem:
            # The blue-yellow-red color map from tecplot
            rgbbreak = [0.0, 0.05, 0.1, 0.325, 0.55, 0.755, 1.0]
            rgbvals = [
                [69, 145, 224, 255, 254, 252, 215],
                [117, 191, 243, 255, 224, 141, 48],
                [180, 219, 248, 191, 144, 89, 39]]

            for i in xrange(self.conn.shape[0]):
                # Determine the color to use
                tval = (xdv[i,0] - tmin)/(tmax - tmin)
                for j in xrange(6):
                    if tval >= rgbbreak[j] and tval <= rgbbreak[j+1]:
                        u = (tval - rgbbreak[j])/(rgbbreak[j+1] - rgbbreak[j])
                        r = (1.0 - u)*rgbvals[0][j] + u*rgbvals[0][j+1]
                        g = (1.0 - u)*rgbvals[1][j] + u*rgbvals[1][j+1]
                        b = (1.0 - u)*rgbvals[2][j] + u*rgbvals[2][j+1]
                        break

                tikz += '\\definecolor{mycolor}{RGB}{%d,%d,%d}'%(
                    int(r), int(g), int(b))
                tikz += '\\fill[mycolor] (%f, %f) -- (%f, %f) -- '%(
                    self.X[self.conn[i,0]-1,0], self.X[self.conn[i,0]-1,1],
                    self.X[self.conn[i,1]-1,0], self.X[self.conn[i,1]-1,1])
                tikz += '(%f, %f) -- (%f, %f) -- cycle;\n'%(
                    self.X[self.conn[i,3]-1,0], self.X[self.conn[i,3]-1,1],
                    self.X[self.conn[i,2]-1,0], self.X[self.conn[i,2]-1,1])

                l = 0.5*((self.X[self.conn[i,1]-1,0]
                         - self.X[self.conn[i,0]-1,0])
                         + (self.X[self.conn[i,3]-1,0]
                            - self.X[self.conn[i,2]-1,0]))
                width = 0.2*l
                l *= 0.85

                xav = 0.0
                yav = 0.0
                for j in xrange(4):
                    xav += 0.25*self.X[self.conn[i,j]-1,0]
                    yav += 0.25*self.X[self.conn[i,j]-1,1]

                # Now, draw the angle to use
                for j in xrange(nmat):
                    if xdv[i,1+j] > 0.75:
                        c = np.cos(self.angles[j])
                        s = np.sin(self.angles[j])
                        tikz += '\\draw[line width=' + \
                            '%.2fmm] (%f, %f) -- (%f, %f);\n'%(
                                width,
                                xav - 0.5*l*c, yav - 0.5*l*s,
                                xav + 0.5*l*c, yav + 0.5*l*s)
        else:
            # Write out the file so that it looks like a multi-material problem
            grey = [225, 225, 225]
            rgbvals = [
                [0, 15, 255, 178, 0],
                [100, 15, 0, 34, 255],
                [0, 150, 255, 34, 255]]

            for i in xrange(self.conn.shape[0]):
                # Determine the color to use
                jmax = np.argmax(xdv[i,1:])

                u = (xdv[i,0] - tmin)/(tmax - tmin)
                r = (1.0 - u)*grey[0] + u*rgbvals[0][jmax]
                g = (1.0 - u)*grey[1] + u*rgbvals[1][jmax]
                b = (1.0 - u)*grey[2] + u*rgbvals[2][jmax]

                tikz += '\\definecolor{mycolor}{RGB}{%d,%d,%d}'%(
                    int(r), int(g), int(b))
                tikz += '\\fill[mycolor] (%f, %f) -- (%f, %f) -- '%(
                    self.X[self.conn[i,0]-1,0], self.X[self.conn[i,0]-1,1],
                    self.X[self.conn[i,1]-1,0], self.X[self.conn[i,1]-1,1])
                tikz += '(%f, %f) -- (%f, %f) -- cycle;\n'%(
                    self.X[self.conn[i,3]-1,0], self.X[self.conn[i,3]-1,1],
                    self.X[self.conn[i,2]-1,0], self.X[self.conn[i,2]-1,1])

                if self.angles[jmax] is not None:
                    l = 0.5*((self.X[self.conn[i,1]-1,0] -
                              self.X[self.conn[i,0]-1,0]) +
                             (self.X[self.conn[i,3]-1,0] -
                              self.X[self.conn[i,2]-1,0]))
                    width = 0.2*l
                    l *= 0.85

                    xav = 0.0
                    yav = 0.0
                    for j in xrange(4):
                        xav += 0.25*self.X[self.conn[i,j]-1,0]
                        yav += 0.25*self.X[self.conn[i,j]-1,1]

                    # Now, draw the angle to use
                    if xdv[i,1+jmax] > 0.75:
                        c = np.cos(self.angles[jmax])
                        s = np.sin(self.angles[jmax])
                        tikz += '\\draw[line width=' + \
                            '%.2fmm] (%f, %f) -- (%f, %f);\n'%(
                                width,
                                xav - 0.5*l*c, yav - 0.5*l*s,
                                xav + 0.5*l*c, yav + 0.5*l*s)

        tikz += '\\end{tikzpicture}\\end{figure}\\end{document}\n'
        # Write the solution file
        fp = open(filename, 'w')
        if fp:
            fp.write(tikz)
            fp.close()

        return

    def writeSolution(self, u, psi, x, filename='solution.dat'):
        '''
        Write out the solution
        '''

        # Get the dimensions of the different arrays
        n = self.X.shape[0]
        ne = self.tconn.shape[0]
        nmat = self.Cmats.shape[0]

        # Compute the values of the stresses
        stress = np.zeros((ne, nmat))

        # Compute all the stresses
        multi.computeallstress(self.ftype, self.xi, self.eta,
                               self.conn.T, self.X.T, u.x.T,
                               self.tconn.T, self.tweights.T, x.x.T,
                               self.epsilon, self.h.T, self.G.T, stress.T)

        # Write the equivalent tex file
        tikzfile = os.path.splitext(filename)[0] + '.tex'
        self.writeTikzFile(x, filename=tikzfile)

        tikzfile = os.path.splitext(filename)[0] + 'stress.tex'
        self.writeTikzStressFile(stress, tikzfile)

        # Write out the solution file for tecplotting
        fp = open(filename, 'w')

        if fp:
            # Determine the design variable values
            xval = x.x
            xdv = np.zeros(xval.shape)

            for i in xrange(self.tconn.shape[0]):
                for j in xrange(self.tconn.shape[1]):
                    if self.tconn[i,j] > 0:
                        xdv[i,:] += \
                            self.tweights[i,j]/xval[self.tconn[i,j]-1,:]

            fp.write('Title = \"Solution\"\n')
            s = 'Variables = x, y, u, v, psiu, psiv, t'
            for i in xrange(nmat):
                s += ', x%d'%(i+1)
            for i in xrange(nmat):
                s += ', stress%d'%(i+1)
            s += '\n'
            fp.write(s)
            fp.write('Zone T=structure n=%d e=%d '%(n, ne))
            fp.write('datapacking=block ')
            fp.write('zonetype=fequadrilateral ')
            s = 'varlocation=([7'
            for i in xrange(2*nmat):
                s += ',%d'%(i+8)
            s += ']=cellcentered)\n'
            fp.write(s)

            # Write out the nodal locations
            for k in xrange(2):
                for i in xrange(n):
                    fp.write('%e\n'%(self.X[i,k]))

            # Write out the displacements
            for k in xrange(2):
                for i in xrange(n):
                    fp.write('%e\n'%(u.x[i,k]))

            # Write out the adjoint variables
            fp.write('\n\n')
            for k in xrange(2):
                for i in xrange(n):
                    fp.write('%e\n'%(psi.x[i,k]))

            # Write out the design variables
            fp.write('\n\n')
            for j in xrange(nmat+1):
                for i in xrange(ne):
                    fp.write('%e\n'%(xdv[i,j]))

            # Write out the stresses
            fp.write('\n\n')
            for j in xrange(nmat):
                for i in xrange(ne):
                    fp.write('%e\n'%(stress[i,j]))

            # Write the connectivity
            for i in xrange(ne):
                fp.write('%d %d %d %d\n'%(
                    self.conn[i,0], self.conn[i,1],
                    self.conn[i,3], self.conn[i,2]))

            fp.close()

        return

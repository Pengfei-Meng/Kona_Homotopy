import numpy as np
import os

import fstopo.thickness as thickness
import fstopo.sparse as sparse
from fstopo.linalg import MGVec
from fstopo.base_problem import FullSpaceProblem

class ThicknessProblem(FullSpaceProblem):
    '''
    The following is the base class for the full-space barrier
    methods.
    '''

    def __init__(self, conn, X, tconn, tweights,
                 kmat, dmat, Cmat, qval, h, G, epsilon,
                 bcnodes, bcvars, force=None, elem_area=None, rho=None):
        '''
        Record all the data that's required for optimization
        '''
        if elem_area is not None:
            self.elem_area = elem_area
        else:
            self.elem_area = 1.0
        if rho is not None:
            self.rho = rho
        else:
            self.rho = 1.0
        self.mscale = 1e-1 * self.elem_area * self.rho
        self.mpower = -1.0

        self.xi = 0.0
        self.eta = 0.0

        # Store the failure and parametrization types
        self.ptype = 3
        self.ftype = 1

        # Copy the connectivity of the mesh and the nodal locations
        self.conn = conn
        self.X = X

        # Copy over the filter connectivity
        self.tconn = tconn
        self.tweights = tweights

        # Copy over the material data
        self.Cmat = Cmat
        self.qval = qval

        # Copy over the allocated matrices
        self.kmat = kmat
        self.cmat = kmat.duplicate()
        self.dmat = dmat

        # Copy over the failure criteria data
        self.h = h
        self.G = G
        self.epsilon = epsilon

        # Record the boundary conditions
        self.bcnodes = bcnodes
        self.bcvars = bcvars

        # Set the force vector
        self.force = force

        # Allocate a temporary vector
        self.xtemp = self.createDesignVec()

        return

    def createDesignVec(self):
        return MGVec(n=self.conn.shape[0], nb=1)

    def createSolutionVec(self):
        return MGVec(n=self.X.shape[0], nb=2)

    def createWeightVec(self):
        return None

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
        # compute the filtered thicknesses
        thick = np.zeros(x.x.shape)
        for i in xrange(self.tconn.shape[0]):
            for j in xrange(self.tconn.shape[1]):
                if self.tconn[i,j] > 0:
                    if self.ptype == 3 or self.ptype == 4:
                        thick[i,:] += \
                            self.tweights[i,j]*x.x[self.tconn[i,j]-1,:]
                    elif self.ptype == 1 or self.ptype == 2:
                        thick[i,:] += \
                            self.tweights[i,j]/x.x[self.tconn[i,j]-1,:]
        # compute the mass
        return self.mscale*np.sum(thick)

    def computeMassDeriv(self, x, rx):
        '''Compute the derivative of the mass'''
        # compute the filtered thickness derivative
        thick = np.zeros(x.x.shape)
        for i in xrange(self.tconn.shape[0]):
            for j in xrange(self.tconn.shape[1]):
                if self.tconn[i,j] > 0:
                    if self.ptype == 3 or self.ptype == 4:
                        thick[i,:] += self.tweights[i,j]
                    elif self.ptype == 1 or self.ptype == 2:
                        thick[i,:] += \
                            -self.tweights[i,j]/(x.x[self.tconn[i,j]-1,:]**2)
        rx.x = self.mscale*thick

    def addMass2ndDeriv(self, x, dmat):
        '''Compute the second derivative of the mass'''
        # compute the filtered thickness second derivative
        thick = np.zeros(x.x.shape)
        for i in xrange(self.tconn.shape[0]):
            for j in xrange(self.tconn.shape[1]):
                if self.tconn[i,j] > 0:
                    if self.ptype == 1 or self.ptype == 2:
                        thick[i,:] += \
                            2*self.tweights[i,j]/(x.x[self.tconn[i,j]-1,:]**3)
        # Add the terms from the second derivatives of the mass
        self.xtemp.zero()
        if self.ptype == 3 or self.ptype == 4:
            dmat.A = np.zeros(dmat.A.shape)
        elif self.ptype == 1 or self.ptype == 2:
            self.xtemp.x = thick
            sparse.adddiagonal(
                self.xtemp.x.T, dmat.rowp[0], dmat.cols[0], dmat.A[0].T)
        return

    def computeMaxStressStep(self, x, u, xstep, ustep, tau):
        '''
        Compute the maximum step length before violating the
        stress-constraint boundary
        '''

        # Compute the maximum step length from the stress
        alpha = thickness.computemaxstep(
            self.ftype, self.xi, self.eta, self.conn.T,
            self.X.T, u.x.T, ustep.x.T, self.tconn.T, self.tweights.T,
            x.x.T, xstep.x.T, self.epsilon, self.h.T, self.G.T, tau)

        return alpha

    def computeLogStressSum(self, x, u):
        '''
        Compute the sum of the logarithms of the stress constrains.
        '''
        logsum = thickness.computelogstresssum(
            self.ftype, self.xi, self.eta, self.conn.T, self.X.T, u.x.T,
            self.tconn.T, self.tweights.T, x.x.T, self.epsilon,
            self.h.T, self.G.T)

        return logsum

    def addLogStressSumDeriv(self, x, u, barrier, rx, ru):
        '''
        Add the derivative of the following term to the residual:
        .        - barrier*log(det(W))
        '''
        thickness.addlogstresssumderiv(self.ftype, self.xi, self.eta,
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
        thickness.computekmat(self.ptype, self.conn.T, self.X.T,
                              self.tconn.T, self.tweights.T, x.x.T,
                              self.qval, self.Cmat.T,
                              mat.rowp[0], mat.cols[0], mat.A[0].T)

        # Apply the boundary conditions
        ident = True
        sparse.applymatbcs(mat.bcnodes[0], mat.bcvars[0].T,
                           mat.rowp[0], mat.cols[0], mat.A[0].T, ident)

        return

    def assembleDMat(self, x, u, diag, barrier, mat):
        '''
        Assemble the D matrix
        '''
        # Compute the D-matrix
        thickness.computedmat(self.ftype, self.xi, self.eta,
                              self.conn.T, self.X.T, u.x.T,
                              self.tconn.T, self.tweights.T, x.x.T,
                              self.epsilon, self.h.T, self.G.T,
                              mat.rowp[0], mat.cols[0], mat.A[0].T)

        # Scale the result by the barrier parameter
        mat.A[0] *= barrier

        sparse.adddiagonal(diag.x.T, mat.rowp[0], mat.cols[0], mat.A[0].T)

        return

    def assembleCMat(self, x, u, diag, barrier, mat):
        '''
        Assemble the matrix of second derivatives
        '''

        # Compute the stiffness matrix on the finest level
        thickness.computecmat(self.ftype, self.xi, self.eta,
                              self.conn.T, self.X.T, u.x.T,
                              self.tconn.T, self.tweights.T, x.x.T,
                              self.epsilon, self.h.T, self.G.T,
                              mat.rowp[0], mat.cols[0], mat.A[0].T)

        # Scale the matrix by the barrier parameter
        mat.A[0] *= barrier

        # Apply the boundary conditions
        ident = True
        sparse.applymatbcs(mat.bcnodes[0], mat.bcvars[0].T,
                           mat.rowp[0], mat.cols[0], mat.A[0].T, ident)

        return

    def multAMatAdd(self, x, u, px, yu):
        '''
        Compute the product: yu += A(u)*px
        '''
        thickness.addamatproduct(self.ptype, self.conn.T, self.X.T, u.x.T,
                                 self.tconn.T, self.tweights.T, x.x.T, px.x.T,
                                 self.qval, self.Cmat.T, yu.x.T)
        sparse.applyvecbcs(self.bcnodes, self.bcvars.T, yu.x.T)
        return

    def multAMatTransposeAdd(self, x, u, pu, yx):
        '''
        Compute the product yx = A(u)^{T}*pu
        '''
        sparse.applyvecbcs(self.bcnodes, self.bcvars.T, pu.x.T)
        thickness.addamattransposeproduct(self.ptype, self.conn.T,
                                          self.X.T, u.x.T, pu.x.T,
                                          self.tconn.T, self.tweights.T, x.x.T,
                                          self.qval, self.Cmat.T, yx.x.T)
        return

    def multAddMatOffDiag(self, x, u, barrier, xstep, ustep, rx, ru):
        '''
        Add the off-diagonal matrix-vector products with the
        C-matrices
        '''
        # Compute the off-diagonal products
        thickness.addcmatoffdiagproduct(
            self.ftype, self.xi, self.eta, self.conn.T,
            self.X.T, u.x.T, ustep.x.T, self.tconn.T, self.tweights.T,
            x.x.T, xstep.x.T, self.epsilon, self.h.T, self.G.T,
            barrier, rx.x.T, ru.x.T)

        # Apply the boundary conditions
        sparse.applyvecbcs(self.bcnodes, self.bcvars.T, ru.x.T)

        return

    def computeWeightCon(self, x, rw):
        '''
        Compute the values of the weighting constraints. Note that
        these constraints are equalities and do not depend on the
        design variables.
        '''
        thickness.computeweightcon(x.x.T, rw.x)
        return

    def writeTikzStressFile(self, stress, filename='solution.tex'):
        '''
        Write out a .tikz file for the stresses
        '''

        # The blue-yellow-red color map from tecplot
        rgbbreak = [0.0, 0.05, 0.1, 0.325, 0.55, 0.755, 1.0]
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

        smax = np.max(stress)
        smin = np.min(stress)

        for i in xrange(self.conn.shape[0]):
            # Determine the color to use
            r = 0
            g = 0
            b = 0
            tval = (stress[i] - smin)/(smax - smin)
            for j in xrange(6):
                if tval >= rgbbreak[j] and tval <= rgbbreak[j+1]:
                    u = (tval - rgbbreak[j])/(rgbbreak[j+1] - rgbbreak[j])
                    r = (1.0 - u)*rgbvals[0][j] + u*rgbvals[0][j+1]
                    g = (1.0 - u)*rgbvals[1][j] + u*rgbvals[1][j+1]
                    b = (1.0 - u)*rgbvals[2][j] + u*rgbvals[2][j+1]
                    break

            tikz += '\\definecolor{mycolor}{RGB}{%d,%d,%d}'%(
                int(r), int(g), int(b))
            tikz += '\\fill[mycolor] ' + \
                '(%f, %f) -- (%f, %f) -- (%f, %f) -- (%f, %f) -- cycle;\n'%(
                    self.X[self.conn[i,0]-1,0], self.X[self.conn[i,0]-1,1],
                    self.X[self.conn[i,1]-1,0], self.X[self.conn[i,1]-1,1],
                    self.X[self.conn[i,3]-1,0], self.X[self.conn[i,3]-1,1],
                    self.X[self.conn[i,2]-1,0], self.X[self.conn[i,2]-1,1])

        tikz += '\\end{tikzpicture}\\end{figure}\\end{document}\n'

        # Write the solution file
        fp = open(filename, 'w')
        if fp:
            fp.write(tikz)
            fp.close()

        return

    def writeTikzFile(self, x, filename='solution.tikz'):
        '''
        Use the design variables to write out a .tikz file
        '''

        # Determine the design variable values
        xval = x.x
        xdv = np.zeros(xval.shape)

        for i in xrange(self.tconn.shape[0]):
            for j in xrange(self.tconn.shape[1]):
                if self.tconn[i,j] > 0:
                    if self.ptype == 3 or self.ptype == 4:
                        xdv[i,:] += self.tweights[i,j]*xval[self.tconn[i,j]-1,:]
                    elif self.ptype == 1 or self.ptype == 2:
                        xdv[i,:] += self.tweights[i,j]/xval[self.tconn[i,j]-1,:]

        # Set the range of thickness values
        tmin = min(xdv[:,0])-1e-2
        tmax = max(xdv[:,0])+1e-2

        # The blue-yellow-red color map from tecplot
        rgbbreak = [0.0, 0.05, 0.1, 0.325, 0.55, 0.755, 1.0]
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
            tikz += '\\fill[mycolor] ' + \
                '(%f, %f) -- (%f, %f) -- (%f, %f) -- (%f, %f) -- cycle;\n'%(
                    self.X[self.conn[i,0]-1,0], self.X[self.conn[i,0]-1,1],
                    self.X[self.conn[i,1]-1,0], self.X[self.conn[i,1]-1,1],
                    self.X[self.conn[i,3]-1,0], self.X[self.conn[i,3]-1,1],
                    self.X[self.conn[i,2]-1,0], self.X[self.conn[i,2]-1,1])

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

        # Get the problem dimensions
        n = self.X.shape[0]
        ne = self.tconn.shape[0]

        # Compute the values of the stresses
        stress = np.zeros(ne)

        # Compute all the stresses
        thickness.computeallstress(self.ftype, self.xi, self.eta,
                                   self.conn.T, self.X.T, u.x.T,
                                   self.tconn.T, self.tweights.T, x.x.T,
                                   self.epsilon, self.h.T, self.G.T, stress)

        # Write the equivalent tex file
        tikzfile = os.path.splitext(filename)[0] + '.tex'
        self.writeTikzFile(x, filename=tikzfile)

        # Write the equivalent tex file
        tikzfile = os.path.splitext(filename)[0] + '_stress.tex'
        self.writeTikzStressFile(stress, filename=tikzfile)

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

            fp.write('TITLE = \"Solution\"\n')
            fp.write('VARIABLES = x, y, u, v, psiu, psiv, t, stress\n')
            fp.write('ZONE T=\"structure\" N=%d E=%d '%(n, ne))
            fp.write('F=FEBLOCK ')
            fp.write('ET=QUADRILATERAL ')
            fp.write('VARLOCATION=([7,8]=cellcentered)\n')

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
            for i in xrange(ne):
                fp.write('%e\n'%(xdv[i,0]))

            # Write out the stresses
            fp.write('\n\n')
            for i in xrange(ne):
                fp.write('%e\n'%(stress[i]))

            # Write the connectivity
            for i in xrange(ne):
                fp.write('%d %d %d %d\n'%(
                    self.conn[i,0], self.conn[i,1],
                    self.conn[i,3], self.conn[i,2]))

            fp.close()

        return

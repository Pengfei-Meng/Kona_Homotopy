import numpy as np

import fstopo.sparse as sparse

def fgmres_solve(mat, rhs, x, m=30, rtol=1e-6, atol=1e-30, print_flag=0):
    '''
    Solve a linear system using FGMRES using the specified
    tolerances.
    '''

    W = [x.duplicate()]
    Z = []
    for i in xrange(m):
        W.append(x.duplicate())
        Z.append(x.duplicate())

    return fgmres(mat, rhs, x, W, Z,
                  rtol=rtol, atol=atol, print_flag=print_flag)


def fgmres(mat, rhs, x, W, Z, min_iters=-1,
           c1=0.5, rtol=1e-6, atol=1e-30, print_flag=0):
    '''
    Solve a linear system using flexible GMRES

    The use of FGMRES is required since we use a flexible
    preconditioner. The use of the flexible preconditioner destroys
    the short-term Lanczos recurrence formula that is the basis of
    Krylov subspace methods for symmetric systems. Therefore, we use
    GMRES will full orthogonalizaiton.  In addition, FGMRES is
    required when approximate Hessian-vector products are
    employed. Here we compute Hessian-vector products that include
    finite-difference matrix-vector products which introduce numerical
    errors.

    input:
    rhs:  the right-hand-side (np array)
    W:    list of temp vectors, len(W) = m+1
    Z:    list of temp vectors, len(Z) = m
    rtol: the relative residual tolerance
    atol: the absolute residual tolerance
    '''

    m = min(len(W)-1, len(Z))

    # Allocate the Hessenberg - this allocates a full matrix
    H = np.zeros((m+1, m))

    # Allocate small arrays of size m
    res = np.zeros(m+1)

    # Store the normal rotations
    Qsin = np.zeros(m)
    Qcos = np.zeros(m)

    # Perform the initialization: copy over rhs to W[0] and
    # normalize the result - store the entry in res[0]
    W[0].copy(rhs)
    res[0] = np.sqrt(W[0].dot(W[0]))
    W[0].scale(1.0/res[0])

    # Store the initial residual norm
    rhs_norm = res[0]
    if print_flag != 0:
        print 'FGMRES[%2d]: %10.3e' % (0, res[0])

    # Keep track of how many iterations are actually required
    niters = 0

    # Perform the matrix-vector products
    for i in xrange(m):
        # Apply the preconditioner - multigrid
        mat.applyPc(W[i], Z[i])

        # Compute the matrix-vector product
        mat.mult(Z[i], W[i+1])

        # Perform modified Gram-Schmidt orthogonalization
        for j in xrange(i+1):
            H[j, i] = W[i+1].dot(W[j])
            W[i+1].axpy(-H[j, i], W[j])

        # Compute the norm of the orthogonalized vector and
        # normalize it
        H[i+1, i] = np.sqrt(W[i+1].dot(W[i+1]))
        W[i+1].scale(1.0/H[i+1, i])

        # Apply the Givens rotations
        for j in xrange(i):
            h1 = H[j, i]
            h2 = H[j+1, i]
            H[j, i] = h1*Qcos[j] + h2*Qsin[j]
            H[j+1, i] = -h1*Qsin[j] + h2*Qcos[j]

        # Compute the contribution to the Givens rotation
        # for the current entry
        h1 = H[i, i]
        h2 = H[i+1, i]
        sq = np.sqrt(h1*h1 + h2*h2)
        Qcos[i] = h1/sq
        Qsin[i] = h2/sq

        # Apply the newest Givens rotation to the last entry
        H[i, i] = h1*Qcos[i] + h2*Qsin[i]
        H[i+1, i] = -h1*Qsin[i] + h2*Qcos[i]

        # Update the residual
        h1 = res[i]
        res[i] = h1*Qcos[i]
        res[i+1] = -h1*Qsin[i]

        # Update the iteration count
        niters += 1
        if print_flag != 0 and niters % print_flag == 0:
            print 'FGMRES[%2d]: %10.3e' % (niters, abs(res[i+1]))

        # Perform the convergence check
        if (i >= min_iters and
            (np.fabs(res[i+1]) < atol or
             np.fabs(res[i+1]) < rtol*rhs_norm)):
            break

    # Compute the linear combination
    for i in xrange(niters-1, -1, -1):
        for j in xrange(i+1, niters):
            res[i] -= H[i, j]*res[j]
        res[i] /= H[i, i]

    # Form the linear combination
    x.zero()
    for i in xrange(niters):
        x.axpy(res[i], Z[i])

    return niters

class MGMat(object):
    def __init__(self):
        '''
        Store the components of the multgrid matrix
        '''

        # Set the preconditioning options
        self.pc_nsmooth = 3
        self.pc_omega = 1.25
        self.pc_niters = 1
        self.pc_cycle = 'W'

        # The non-zero pattern on all levels
        self.nlevels = 0
        self.nb = 0
        self.A = []
        self.rowp = []
        self.cols = []

        # The constrained nodes/variables
        self.bcnodes = None
        self.bcvars = None

        # The inter-grid interpolation for multigrid
        self.irowp = []
        self.icols = []
        self.interp = []

        # Temporary variables for the multigrid method - generated
        # during the first call to multigrid
        self.b = []
        self.res = []
        self.u = []

        # The inverse of the diagonal on all levels - generated
        # during a factorization
        self.D = []

        return

    def duplicate(self, nb=None):
        '''
        Create a duplicate of this matrix. Note that this does not
        copy the entries, but just duplicates the memory so that you
        have two independent matrices that can be used.
        '''

        # Create a new matrix
        mat = MGMat()

        # copy over the options
        mat.pc_nsmooth = self.pc_nsmooth
        mat.pc_omega = self.pc_omega
        mat.pc_niters = self.pc_niters
        mat.pc_cycle = self.pc_cycle

        # Copy over the level information
        mat.nlevels = self.nlevels
        if nb is None:
            mat.nb = self.nb
        else:
            mat.nb = nb
        mat.rowp = self.rowp
        mat.cols = self.cols

        # Allocate new space for the matrix
        for k in xrange(self.nlevels):
            mat.A.append(np.zeros((self.A[k].shape[0], mat.nb, mat.nb)))

        # Set the pointer to the boundary conditions
        mat.bcnodes = self.bcnodes
        mat.bcvars = self.bcvars

        # Set pointers to the inter-grid interpolation
        mat.irowp = self.irowp
        mat.icols = self.icols
        mat.interp = self.interp

        return mat

    def mginit(self, nb, nlevels, rowp, cols,
               irowp, icols, interp, bcnodes=None, bcvars=None):
        '''
        Initialize the multigrid levels.

        Note that the matrix retains a copy to the rowp/cols data and
        the irowp/icols/interp (which are passed in as a list of
        interpolation matrices).  Therefore, if you change this data
        externally, bad stuff will happen.

        Input:
        nb:       the block size for the problem
        nlevels:  the number of levels in the problem
        rowp:     CSR row-pointer for the finest level
        cols:     column indices for the finest level
        irowp:    LIST of CSR pointers for the interpolation
        icols:    LIST of column indices for the interpolation
        interp:   the interpolation weights
        bcnodes:  the nodes for the boundary conditions
        bcvars:   the variables associated with the bcs
        '''

        self.nb = nb
        self.nlevels = nlevels

        # Set the first
        self.rowp.append(rowp)
        self.cols.append(cols)

        # Allocate the space for the new matrix
        ncols = rowp[-1]-1
        A = np.zeros((ncols, self.nb, self.nb))
        self.A.append(A)

        # Record the interpolation data
        self.irowp = irowp
        self.icols = icols
        self.interp = interp
        self.bcnodes = bcnodes
        self.bcvars = bcvars

        # Set the levels
        for k in xrange(self.nlevels-1):
            # Compute the number of rows/cols in the next matrix
            n = len(irowp[k])-1
            new_rowp = np.zeros(n+1, dtype=np.intc)

            # Estimate the size of the new matrix
            new_cols = np.zeros(50*n, dtype=np.intc)

            # Allocate space for the transpose of the interpolation
            m = max(icols[k])
            icolp = np.zeros(m+1, dtype=np.intc)
            irows = np.zeros(len(icols[k]), dtype=np.intc)

            # Compute the reduced matrix
            info = sparse.computenzmatmultinner(
                irowp[k], icols[k], self.rowp[k], self.cols[k],
                icolp, irows, new_rowp, new_cols)

            # If it didn't work, figure out how many entries we should
            # have allocated
            if info != 0:
                new_cols = np.zeros(info, dtype=np.intc)
                info = sparse.computenzmatmultinner(
                    irowp[k], icols[k], self.rowp[k], self.cols[k],
                    icolp, irows, new_rowp, new_cols)

            # Allocate the matrix
            ncols = new_rowp[-1]-1
            new_cols = np.array(new_cols[:ncols], dtype=np.intc)

            # Append the next matrix
            self.rowp.append(new_rowp)
            self.cols.append(new_cols)

            # Allocate the space for the new level
            A = np.zeros((ncols, self.nb, self.nb))
            self.A.append(A)

        return

    def factor(self):
        '''
        Compute the factor on all levels
        '''

        # Check if we need to allocate the diagonal matrix
        if len(self.D) == 0:
            for k in xrange(self.nlevels):
                n = self.rowp[k].shape[0]-1
                self.D.append(np.zeros((n, self.nb, self.nb)))

        # Interpolate down the multigrid levels
        for k in xrange(1, self.nlevels):
            sparse.computematmultinner(
                self.irowp[k-1], self.icols[k-1], self.interp[k-1],
                self.rowp[k-1], self.cols[k-1], self.A[k-1].T,
                self.rowp[k], self.cols[k], self.A[k].T)

        # Apply the boundary conditions
        if self.bcnodes is not None:
            for k in xrange(self.nlevels):
                ident = True
                sparse.applymatbcs(self.bcnodes[k], self.bcvars[k].T,
                                   self.rowp[k], self.cols[k],
                                   self.A[k].T, ident)

        for k in xrange(self.nlevels):
            sparse.computediagfactor(self.rowp[k], self.cols[k],
                                     self.A[k].T, self.D[k].T)

        return

    def mult(self, x, y):
        '''
        Compute the matrix-vector product
        '''
        # Compute the matrix-vector product
        sparse.matmult(self.rowp[0], self.cols[0],
                       self.A[0].T, x.x.T, y.x.T)
        # Apply the boundary conditions
        if self.bcnodes:
            sparse.applyvecbcs(self.bcnodes[0], self.bcvars[0].T, y.x.T)

        return

    def applyPc(self, b, ans):
        '''
        Apply a single cycle of multigrid as a preconditioner
        '''

        if len(self.b) < self.nlevels:
            for k in xrange(self.nlevels):
                n = self.rowp[k].shape[0] - 1
                self.b.append(np.zeros((n, self.nb)))
                self.res.append(np.zeros((n, self.nb)))
                self.u.append(np.zeros((n, self.nb)))

        # Copy the right-hand-side into the multigrid data
        self.b[0][:] = b.x[:]
        self.u[0][:] = 0.0

        # Keep track of whether this is a V or W cycle
        niters = 1
        if self.pc_cycle == 'W':
            niters = 2

        for k in xrange(self.pc_niters):
            # Apply multigrid using a recursive algorithm
            self.applyMGRecursive(lev=0, niters=niters,
                                  nsmooth=self.pc_nsmooth,
                                  omega=self.pc_omega)

        # Copy the answer from the result
        ans.x[:] = self.u[0][:]

        if self.bcnodes:
            sparse.applyvecbcs(self.bcnodes[0], self.bcvars[0].T, ans.x.T)

        return

    def applyMGRecursive(self, lev=0, niters=1, nsmooth=1, omega=1.0):
        '''
        Apply the multigrid recursively at each level. Only apply a
        smoother at the lowest level of multigrid
        '''

        if lev == self.nlevels-1:
            sparse.applysor(
                self.rowp[lev], self.cols[lev], self.A[lev].T,
                self.D[lev].T, nsmooth, omega,
                self.b[lev].T, self.u[lev].T)
        else:
            # Smooth the residual at this level
            sparse.applysor(self.rowp[lev], self.cols[lev],
                            self.A[lev].T, self.D[lev].T,
                            nsmooth, omega, self.b[lev].T, self.u[lev].T)

            # Compute the residual at this level
            sparse.matmult(self.rowp[lev], self.cols[lev], self.A[lev].T,
                           self.u[lev].T, self.res[lev].T)
            self.res[lev][:] = self.b[lev][:] - self.res[lev][:]

            # Restrict the residual to the next-lowest level
            # Note that the fortran-level restriction does not zero entries
            self.b[lev+1][:] = 0.0
            sparse.computerestrict(
                self.irowp[lev], self.icols[lev], self.interp[lev],
                self.res[lev].T, self.b[lev+1].T)
            if self.bcnodes is not None:
                sparse.applyvecbcs(self.bcnodes[lev+1], self.bcvars[lev+1].T,
                                   self.b[lev+1].T)

            # Apply multigrid at the next mesh level
            self.u[lev+1][:] = 0.0
            for j in xrange(niters):
                self.applyMGRecursive(lev=lev+1, niters=niters,
                                      nsmooth=nsmooth, omega=omega)

            # Interpolate back from the next lowest level
            sparse.computeinterp(
                self.irowp[lev], self.icols[lev], self.interp[lev],
                self.u[lev+1].T, self.u[lev].T)
            if self.bcnodes is not None:
                sparse.applyvecbcs(self.bcnodes[lev], self.bcvars[lev].T,
                                   self.u[lev].T)

            # Smooth the residual at this level
            sparse.applysor(self.rowp[lev], self.cols[lev],
                            self.A[lev].T, self.D[lev].T,
                            nsmooth, omega, self.b[lev].T, self.u[lev].T)

        return


class MGVec(object):
    '''
    A vector for multigrid
    '''
    def __init__(self, x=None, n=0, nb=2):
        if x is not None:
            self.x = x
        else:
            self.n = n
            self.x = np.zeros((n, nb))
        return

    def duplicate(self):
        '''
        Duplicate the vector, creating a new one
        '''
        return MGVec(n=self.x.shape[0], nb=self.x.shape[1])

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
        return sparse.vecdot(self.x.T, vec.x.T)

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
        Compute the infinity norm of the vector
        '''
        return max(self.x.max(), -self.x.min())


class KKTVec(object):
    def __init__(self, x, lambd, u, psi):
        '''
        Create the KKT vector
        '''
        self.x = x.duplicate()
        if lambd is not None:
            self.lambd = lambd.duplicate()
        else:
            self.lambd = None
        self.u = u.duplicate()
        self.psi = psi.duplicate()
        return

    def duplicate(self):
        return KKTVec(self.x, self.lambd, self.u, self.psi)

    def copy(self, vec):
        '''
        Copy the values from the given vector
        '''
        self.x.copy(vec.x)
        if self.lambd is not None:
            self.lambd.copy(vec.lambd)
        self.u.copy(vec.u)
        self.psi.copy(vec.psi)
        return

    def dot(self, vec):
        '''
        Compute the dot-product of two vectors
        '''
        if self.lambd is not None:
            return (self.x.dot(vec.x) + self.lambd.dot(vec.lambd) +
                    self.u.dot(vec.u) + self.psi.dot(vec.psi))

        return (self.x.dot(vec.x) + self.u.dot(vec.u) +
                self.psi.dot(vec.psi))

    def zero(self):
        '''
        Zero the values in the array
        '''
        self.x.zero()
        if self.lambd is not None:
            self.lambd.zero()
        self.u.zero()
        self.psi.zero()
        return

    def scale(self, alpha):
        '''
        Scale the values
        '''
        self.x.scale(alpha)
        if self.lambd is not None:
            self.lambd.scale(alpha)
        self.u.scale(alpha)
        self.psi.scale(alpha)
        return

    def axpy(self, alpha, vec):
        '''
        Compute self = self + alpha*vec
        '''
        self.x.axpy(alpha, vec.x)
        if self.lambd is not None:
            self.lambd.axpy(alpha, vec.lambd)
        self.u.axpy(alpha, vec.u)
        self.psi.axpy(alpha, vec.psi)
        return

    def infty(self):
        '''
        Compute the infinity norm of the vector
        '''
        if self.lambd is not None:
            return max(self.x.infty(), self.u.infty(),
                       self.psi.infty(), self.lambd.infty())
        return max(self.x.infty(), self.u.infty(), self.psi.infty())

import sys
import os
import numpy as np
from mpi4py import MPI

# Import the material routines
from fstopo.material import *

# Import the fortran-level routines
import fstopo.sparse as sparse

# Import the python-level objects
import fstopo.linalg as linalg
import fstopo.problems.multi_opt as multi_opt
import fstopo.problems.thickness_opt as thickness_opt
from fstopo.optimizer import FullSpaceOpt

def create_blanked_mesh(nb, nc):
    '''
    Create the element connectivity and nodes associated with the
    bracket mesh.
    '''

    nx = nb + nc

    # Allocate all of the nodes (including blanked nodes)
    nodes = np.ones((nx+1, nx+1), dtype=np.int)
    nodes[nb+1:, nb+1:] = -1

    # Allocate all of the elements (including blanked elements)
    elems = np.ones((nx, nx), dtype=np.int)
    elems[nb:, nb:] = -1

    index = 1
    for j in xrange(nx,-1,-1):
        for i in xrange(nx,-1,-1): # nx+1):
            if nodes[i,j] >= 0:
                nodes[i,j] = index
                index += 1

    index = 1
    for i in xrange(nx):
        for j in xrange(nx-1,-1,-1):
            if elems[i,j] >= 0:
                elems[i,j] = index
                index += 1

    return nodes, elems

def create_mesh_conn(nb, nc, nodes, elems, L):
    '''
    Set up the mesh on the most-refined level
    '''

    n = (nb+1)**2 + 2*(nb+1)*nc
    ne = nb**2 + 2*nb*nc

    # Allocate the arrays
    conn = np.zeros((ne, 4), dtype=np.intc)
    X = np.zeros((n, 2))

    nx = nb+nc
    for j in xrange(nx+1):
        for i in xrange(nx+1):
            if nodes[i,j] >= 0:
                X[nodes[i,j]-1, 0] = L*i/nx
                X[nodes[i,j]-1, 1] = L*j/nx

    # Set the nodes in the a-lever
    for j in xrange(nx):
        for i in xrange(nx):
            if elems[i,j] >= 0:
                conn[elems[i,j]-1, 0] = nodes[i,j]
                conn[elems[i,j]-1, 1] = nodes[i+1,j]
                conn[elems[i,j]-1, 2] = nodes[i,j+1]
                conn[elems[i,j]-1, 3] = nodes[i+1,j+1]

    # Set up the filter
    r0 = 2.2
    ri = int(np.floor(r0))

    # Compute the weights on the design variables using a conic filter
    tconn = -np.ones((ne, (2*ri+1)**2), dtype=np.intc)
    tweights = np.zeros((ne, (2*ri+1)**2))

    # Set up the filter for the elements
    for j in xrange(nb+nc):
        for i in xrange(nb+nc):
            if elems[i,j] >= 0:
                index = 0
                for jj in xrange(max(0, j-ri-1), min(nb+nc, j+ri+1)):
                    for ii in xrange(max(0, i-ri-1), min(nb+nc, i+ri+1)):
                        r = np.sqrt((i - ii)**2 + (j - jj)**2)
                        if r < r0 and elems[ii,jj] >= 0:
                            tweights[elems[i,j]-1, index] = (r0 - r)/r0
                            tconn[elems[i,j]-1, index] = elems[ii,jj]
                            index += 1

                # Normalize the thickness weight variables
                tweights[elems[i,j]-1, :] /= np.sum(tweights[elems[i,j]-1, :])

    return conn, X, tconn, tweights

def create_multi_problem(rho, Cmats, qval, qxval, h, G, epsilon,
                         L, nb, nc, nlevels,
                         thickness_flag=False):
    '''
    Create the multi-material problem
    '''

    # Create the mesh data for the finest mesh
    ref_nodes, ref_elems = create_blanked_mesh(nb, nc)
    conn, X, tconn, tweights = create_mesh_conn(nb, nc, ref_nodes, ref_elems, L)

    # Set the nodes where we will apply the force
    fnodes = np.array(ref_nodes[-1, (3*nb/4):nb+2])

    # Now, create the data for the remaining levels - all that is
    # required is the stiffnes info and the connectivity

    # The inter-level interpolation
    mg_interp = []
    mg_irowp = []
    mg_icols = []

    # The inter-level design interpolation
    mg_qinterp = []
    mg_qrowp = []
    mg_qcols = []

    # The stiffness matrix data at each level
    mg_bcnodes = []
    mg_bcvars = []

    # Set up the boundary conditions
    bcnodes = np.array(ref_nodes[:nb+1, -1], dtype=np.intc)
    bcvars = np.zeros((nb+1, 2), dtype=np.intc)
    bcvars[:,0] = 1
    bcvars[:,1] = 2

    mg_bcnodes.append(bcnodes)
    mg_bcvars.append(bcvars)

    for k in xrange(1, nlevels):
        nbb = nb/2**k
        ncc = nc/2**k
        nxx = nbb + ncc

        # Create the mesh
        nodes, elems = create_blanked_mesh(nbb, ncc)

        # Set up the boundary conditions
        bcnodes = np.array(nodes[:nbb+1, -1], dtype=np.intc)
        bcvars = np.zeros((nbb+1, 2), dtype=np.intc)
        bcvars[:,0] = 1
        bcvars[:,1] = 2

        mg_bcnodes.append(bcnodes)
        mg_bcvars.append(bcvars)

        # The interpolation stencil
        interpw = [0.5, 1.0, 0.5]

        # All the interpolation weights
        nnodes = (nbb+1)**2 + 2*ncc*(nbb+1)
        interp = []
        icols = []
        for i in xrange(nnodes):
            interp.append([])
            icols.append([])

        # Set up the interpolation between grids
        for j in xrange(nxx+1):
            for i in xrange(nxx+1):
                if nodes[i,j] >= 0:
                    # Compute the interpolation between grid levels
                    var = []
                    weights = []

                    # Take a weighted stencil from the
                    # adjacent nodes
                    for jj in xrange(-1,2):
                        if 2*j+jj >= 0 and 2*j+jj < 2*nxx+1:
                            for ii in xrange(-1,2):
                                if (2*i+ii >= 0 and 2*i+ii < 2*nxx+1):
                                    if ref_nodes[2*i+ii, 2*j+jj] >= 0:
                                        var.append(
                                            ref_nodes[2*i+ii, 2*j+jj])
                                        weights.append(
                                            interpw[ii+1]*interpw[jj+1])

                    # Normalize the weights
                    ws = sum(weights)
                    for k in xrange(len(weights)):
                        weights[k] /= ws

                    # Arg-sort the arrays
                    isort = np.argsort(var)

                    # Add the weights to the icols array and update irowp
                    for k in isort:
                        interp[nodes[i,j]-1].append(weights[k])
                        icols[nodes[i,j]-1].append(var[k])

        # Flatten things into a single array
        irowp = np.zeros(nnodes+1, dtype=np.intc)
        irowp[0] = 1
        a = []
        cols = []
        for i in xrange(nnodes):
            irowp[i+1] = irowp[i] + len(interp[i])
            a.extend(interp[i])
            cols.extend(icols[i])

        # Add the inter-grid interpolation
        mg_interp.append(np.array(a))
        mg_icols.append(np.array(cols, dtype=np.intc))
        mg_irowp.append(irowp)

        # Set the design-dependent interpolation
        nelems = nbb**2 + 2*nb*nc
        qinterp = []
        qcols = []
        for i in xrange(nelems):
            qinterp.append([])
            qcols.append([])

        # Set the interpolation between designs
        for j in xrange(nxx):
            for i in xrange(nxx):
                if elems[i,j] >= 0:
                    # Compute the interpolation between grid levels
                    var = []
                    weights = []

                    # Take a weighted stencil from the adjacent
                    # elements
                    for jj in xrange(-1,2):
                        if (2*j+jj >= 0 and 2*j+jj < 2*nxx):
                            for ii in xrange(-1,2):
                                if (2*i+ii >= 0 and 2*i+ii < 2*nxx):
                                    var.append(ref_elems[2*i+ii, 2*j+jj])
                                    weights.append(interpw[ii+1]*interpw[jj+1])

                    # Normalize the weights
                    ws = sum(weights)
                    for k in xrange(len(weights)):
                        weights[k] /= ws

                    # Arg-sort the arrays
                    isort = np.argsort(var)

                    # Add the weights to the icols array and update irowp
                    for k in isort:
                        qinterp[elems[i,j]-1].append(weights[k])
                        qcols[elems[i,j]-1].append(var[k])

        # Flatten things into a single array
        qrowp = np.zeros(nelems+1, dtype=np.intc)
        qrowp[0] = 1
        a = []
        cols = []
        for i in xrange(nelems):
            qrowp[i+1] = qrowp[i] + len(qinterp[i])
            a.extend(qinterp[i])
            cols.extend(qcols[i])

        # Add the inter-grid interpolation
        mg_qinterp.append(np.array(a))
        mg_qcols.append(np.array(cols, dtype=np.intc))
        mg_qrowp.append(qrowp)

        ref_nodes = nodes
        ref_elems = elems

    # Create the stiffness matrix
    # ---------------------------
    n = X.shape[0]
    ne = conn.shape[0]
    rowp = np.zeros(n+1, dtype=np.intc)
    cols = np.zeros(9*n, dtype=np.intc)

    # Compute the non-zero pattern
    info = sparse.computekmatnzpattern(conn.T, rowp, cols)
    if info != 0:
        cols = np.zeros(info, dtype=np.intc)
        sparse.computekmatnzpattern(conn.T, rowp, cols)

    # Trim the size of the cols array
    cols = np.array(cols[:rowp[-1]-1], dtype=np.intc)

    # Create the multigrid matrix
    kmat = linalg.MGMat()
    kmat.mginit(2, nlevels, rowp, cols,
                mg_irowp, mg_icols, mg_interp,
                bcnodes=mg_bcnodes, bcvars=mg_bcvars)

    kmat.pc_niters = 1
    kmat.pc_omega = 1.25

    # Create the second-derivative matrix
    # -----------------------------------
    rowp = np.zeros(ne+1, dtype=np.intc)
    cols = np.zeros(9*ne, dtype=np.intc)

    # Compute the non-zero pattern
    info = sparse.computedmatnzpattern(tconn.T, rowp, cols)
    if info != 0:
        cols = np.zeros(info, dtype=np.intc)
        sparse.computedmatnzpattern(tconn.T, rowp, cols)

    # Trim the size of the cols array
    cols = np.array(cols[:rowp[-1]-1], dtype=np.intc)

    # Create the multigrid matrix
    dmat = linalg.MGMat()

    # Set up the problem data
    if thickness_flag:
        dmat.mginit(1, nlevels, rowp, cols,
                    mg_qrowp, mg_qcols, mg_qinterp)
        prob = thickness_opt.ThicknessProblem(
            conn, X, tconn, tweights, kmat, dmat, Cmats,
            qval, h, G, epsilon, mg_bcnodes[0], mg_bcvars[0])
    else:
        nmats = Cmats.shape[0]
        dmat.mginit(1+nmats, nlevels, rowp, cols,
                    mg_qrowp, mg_qcols, mg_qinterp)

        stress_elems = np.arange(1, conn.shape[0]+1, dtype=np.intc)

        prob = multi_opt.MultiProblem(
            conn, X, tconn, tweights, stress_elems, kmat, dmat, rho,
            Cmats, qval, qxval, h, G, epsilon, mg_bcnodes[0], mg_bcvars[0])

    force = prob.createSolutionVec()

    # Distribute the force over a portion of the right-hand-side
    F = -500.0
    for i in xrange(len(fnodes)-1):
        force.x[fnodes[i]-1, 1] = 0.5*F/(len(fnodes)-1)
        force.x[fnodes[i+1]-1, 1] = 0.5*F/(len(fnodes)-1)

    return prob, force

# The dimensions of the problem
L = 100.0
nb = 32
nc = 48
nlevels = 5

thickness_flag = False
ortho_flag = False
angles = []

# Set up the material stiffness/failure properties
if 'thickness' in sys.argv:
    angles = [None]
    thickness_flag = True

    # Set the material properties
    rho = 1.265
    E = 70e3
    nu = 0.3
    smax = 300.0

    # Set the material properties
    Cmats = np.zeros((3,3))

    # Set the constitutive matrix
    Cmats[0,0] = E/(1.0 - nu**2)
    Cmats[0,1] = nu*E/(1.0 - nu**2)
    Cmats[1,0] = Cmats[0,1]
    Cmats[1,1] = Cmats[0,0]

    # Compute the shear stiffness
    Cmats[2,2] = 0.5*E/(1.0 + nu)

    h = np.zeros(3)
    G = np.zeros((3,3))

    scale = (E/(smax*(1.0 - nu**2)))**2
    G[0,0] = scale*(1.0 - nu + nu**2)
    G[0,1] = -scale*(0.5 - 2*nu + 0.5*nu**2)
    G[1,0] = G[0,1]
    G[1,1] = G[0,0]
    G[2,2] = scale*(3.0/4.0)*(1.0 - nu)**2

elif 'multi' in sys.argv:
    # Set the number of materials to use
    rho = [2.810]

    # Create the Cmat matrices
    thetas = [0.0]
    angles = [None, 0.0]
    rho.extend(len(thetas)*[1.265])

    # Set up the matrix
    nmats = len(rho)
    rho = np.array(rho)
    E = [70e3]
    nu = [0.3]
    smax = [300.0]

    Cmats1 = get_isotropic(E, nu)
    h1, G1 = get_von_mises(E, nu, smax)

    # These properties are taken from Jones, pg. 101 for a
    # graphite-epoxy material. Note that units are in MPa.
    E1 = 207e3
    E2 = 5e3
    nu12 = 0.25
    G12 = 2.6e3

    # The strength properties
    Xt = 1035.0
    Yt = 41.0
    S12 = 69.0
    Xc = 689.0
    Yc = 117.0

    Cmats2 = get_global_stiffness(E1, E2, nu12, G12, thetas)

    # Create the failure coefficients
    F1, F2, F11, F22, F12, F66 = get_tsai_wu(Xt, Xc, Yt, Yc, S12)
    h2, G2 = get_failure_coeffs(E1, E2, nu12, G12,
                                F1, F2, F11, F22, F12, F66, thetas)

    # Copy over the values
    h = np.zeros((nmats, 3))
    h[0,:] = h1[0,:]
    h[1:,:] = h2[:,:]

    G = np.zeros((nmats, 3, 3))
    G[0,:,:] = G1[0,:,:]
    G[1:,:,:] = G2[:,:,:]

    # Copy over the stiffness matrices
    Cmats = np.zeros((nmats, 3, 3))
    Cmats[0,:,:] = Cmats1[0,:,:]
    Cmats[1:,:,:] = Cmats2[:,:,:]

else:
    ortho_flag = True

    # These properties are taken from Jones, pg. 101 for a
    # graphite-epoxy material. Note that units are in MPa.
    E1 = 207e3
    E2 = 5e3
    nu12 = 0.25
    G12 = 2.6e3

    # The strength properties
    Xt = 1035.0
    Yt = 41.0
    S12 = 69.0
    Xc = 689.0
    Yc = 117.0

    rho_mat = 1.265

    # Set the material options
    if 'twelve' in sys.argv:
        nmats = 12
    elif 'six' in sys.argv:
        nmats = 6
    else:
        nmats = 4
    thetas = (np.pi/180.0)*np.linspace(-90, 90, nmats+1)[1:]
    angles = thetas

    # Set the material density
    rho = rho_mat*np.ones(nmats)

    # Create the Cmat matrices
    Cmats = get_global_stiffness(E1, E2, nu12, G12, thetas)

    # Create the failure coefficients
    F1, F2, F11, F22, F12, F66 = get_tsai_wu(Xt, Xc, Yt, Yc, S12)
    h, G = get_failure_coeffs(E1, E2, nu12, G12,
                              F1, F2, F11, F22, F12, F66, thetas)

# Use RAMP penalization
qval = 5.0
qxval = 2.0
epsilon = 0.1

# Print out the solution parameters
print 'rho = ', rho
print 'Cmats = ', Cmats
print 'h = ', h
print 'G = ', G
print 'qval = ', qval
print 'qxval = ', qxval
print 'epsilon = ', epsilon

# Set the problem scaling
rho_scale = 1.0
mat_scale = 1.0
stress_scale = 0.1

rho *= rho_scale
h *= stress_scale
G *= stress_scale**2
Cmats *= mat_scale

prob, force = create_multi_problem(rho, Cmats, qval, qxval, h, G, epsilon,
                                   L, nb, nc, nlevels,
                                   thickness_flag=thickness_flag)

force.x *= (mat_scale/stress_scale)

# Record the angles we'll use
prob.angles = angles

# Visualize using ply-orientations
if ortho_flag:
    prob.ply_problem = True

# Create the load vector
u = prob.createSolutionVec()

# Create the design variable vector
x = prob.createDesignVec()
lb = x.duplicate()
ub = x.duplicate()

# Set the thickness lower and upper bounds
t_lb = 0.01
t_ub = 1.0

# Set the lower/upper bounds for the material
mat_lb = 0.001
mat_ub = 1.0

# Set the design variable values and lower/upper bounds
lb.x[:,0] = 1.0/t_ub
ub.x[:,0] = 1.0/t_lb
x.x[:,0] = 1.0/(0.02*t_lb + 0.98*t_ub)

if not thickness_flag:
    lb.x[:,1:] = 1.0/mat_ub
    ub.x[:,1:] = 1.0/mat_lb

    # Initialize the material variables with x = nmats
    x.x[:,1:] = nmats

# Set the file prefix
prefix = 'BracketOrtho'
if thickness_flag:
    prefix = 'BracketThickness'
elif 'multi' in sys.argv:
    prefix = 'BracketMulti'

# Set up a continuous thickness problem
prob.ptype = 1
prob.ftype = 1

# Otherwise, set up a topology problem
if 'topo' in sys.argv:
    prob.ptype = 2
    prob.ftype = 2
    prefix += 'Topo'

# Compute the initial state variable values
kmat = prob.getKMat()
prob.assembleKMat(x, kmat)
kmat.factor()

# Solve the system of equations
linalg.fgmres_solve(kmat, force, u, print_flag=1)

# Optionally, test all the functions
if 'test' in sys.argv:
    prob.test(x, u)

# Create the directory if it does not exist
if not os.path.isdir(prefix):
    os.mkdir(prefix)

prefix += '%s%dx%d'%(os.path.sep, nb, nc)
if ortho_flag:
    prefix += 'x%d'%(nmats)
if not os.path.isdir(prefix):
    os.mkdir(prefix)

# Create the topology optimization class
topo = FullSpaceOpt(prob)

# Set up the gmres subspace for the null-space method?
pc_gmres_size = 0
if not thickness_flag:
    prob.setUpGMRESSubspace(pc_gmres_size)

# Set the information about the size of the subspaces to use
if 'test' in sys.argv:
    topo.kkt_test_iter = 3

# Settings for optimization
topo.pc_type = 'Biros'
topo.fgmres_subspace_size = 40
topo.pc_gmres_subspace_size = 0
topo.max_newton_iters = 1000
topo.line_search_check = False
topo.max_line_iters = 8
topo.c1 = 1e-4
topo.use_exact_hessian = False

# Set the target barrier parameter
topo.barrier_target = 0.25

# Set the barrier parameters
topo.barrier = 1000.0

# Set the optimality tolerance
topo.opt_tol = 1.0

# Set the maximum newton tolerance
topo.max_newton_tol = 0.001

# Set lambda - the Lagrange multipliers for the equality constraints
lambd = prob.createWeightVec()
if lambd is not None:
    lambd.x[:] = 1.0

# Print out the initial solution
fname = '%s/init.dat'%(prefix)
prob.writeSolution(u, u, x, filename=fname)

# Find the maximum step to the full solution
xzero = x.duplicate()
uzero = u.duplicate()
tau = 0.95
alpha = prob.computeMaxStressStep(x, uzero, xzero, u, tau)
print 'Max step: ', alpha
u.scale(alpha)

# Optimize the problem
t0 = MPI.Wtime()
topo.optimize(x, u, lb, ub, force, lambd=lambd,
              prefix=prefix, filename='%s/opt_hist.dat'%(prefix))
t1 = MPI.Wtime() - t0
print 'Total optimization time: ', t1

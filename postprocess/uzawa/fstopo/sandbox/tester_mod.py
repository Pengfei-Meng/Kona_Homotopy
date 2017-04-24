import sys
sys.path.remove('/users/mengp2/uzawa/fstopo')
sys.path.append('/users/mengp2/uzawa/local/lib/python')

# print sys.path

import os
import numpy as np
import pickle
import pdb

# Import the material routines
from fstopo.material import *

# Import the fortran-level routines
import fstopo.sparse as sparse

# Import the python-level objects
import fstopo.linalg as linalg
import fstopo.problems.multi_opt as multi_opt
import fstopo.problems.thickness_opt as thickness_opt
import fstopo.problems.kona_opt as kona_opt
from kona.linalg.vectors.composite import ReducedKKTVector
from kona.linalg.vectors.composite import CompositePrimalVector

# Import Kona Optimization Library
import kona



def create_multi_problem(rho, Cmats, qval, qxval, h, G, epsilon,
                         Lx, Ly, nxg, nyg, nlevels,
                         thickness_flag=False,
                         problem_type='truss'):
    '''
    Create the multi-material problem
    '''

    # Set up the local mesh
    n = (nxg+1)*(nyg+1)
    ne = nxg*nyg
    conn = np.zeros((ne, 4), dtype=np.intc)
    X = np.zeros((n, 2))

    # calculate elem area
    ex = 1e-1*Lx/nxg
    ey = 1e-1*Ly/nyg
    elem_area = ex*ey

    # Set up the local connectivity
    for j in xrange(nyg):
        for i in xrange(nxg):
            index = i + nxg*j
            conn[index, 0] = i + (nxg+1)*j + 1
            conn[index, 1] = i+1 + (nxg+1)*j + 1
            conn[index, 2] = i + (nxg+1)*(j+1) + 1
            conn[index, 3] = i+1 + (nxg+1)*(j+1) + 1

    # Set the nodal locations
    for j in xrange(nyg+1):
        for i in xrange(nxg+1):
            index = i + (nxg+1)*j
            X[index, 0] = (Lx*i)/nxg
            X[index, 1] = (Ly*j)/nyg

    # Set up the filter variables
    r0 = 2.2
    ri = int(np.floor(r0))

    # Compute the weights on the design variables using a conic filter
    tconn = -np.ones((ne, (2*ri+1)**2), dtype=np.intc)
    tweights = np.zeros((ne, (2*ri+1)**2))

    for j in xrange(nyg):
        for i in xrange(nxg):
            index = 0
            for jj in xrange(max(0, j-ri-1), min(nyg, j+ri+1)):
                for ii in xrange(max(0, i-ri-1), min(nxg, i+ri+1)):
                    r = np.sqrt((i - ii)**2 + (j - jj)**2)
                    if r < r0:
                        tweights[i + j*nxg, index] = (r0 - r)/r0
                        tconn[i + j*nxg, index] = ii + jj*nxg + 1
                        index += 1

            # Normalize the thickness weight variables
            tweights[i + j*nxg, :] /= np.sum(tweights[i + j*nxg, :])

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

    # Set the boundary conditions
    if problem_type == 'mbb':
        naw = 1
        nbc = nyg+1 + naw
        bcnodes = np.zeros(nbc, dtype=np.intc)
        bcvars = np.zeros((nbc,2), dtype=np.intc)

        index = 0
        for j in xrange(nyg+1):
            bcnodes[index] = j*(nxg+1) + 1
            bcvars[index,0] = 1
            bcvars[index,1] = -1
            index += 1

        for i in xrange(nxg+1-naw, nxg+1):
            bcnodes[index] = i+1
            bcvars[index,0] = -1
            bcvars[index,1] = 2
            index += 1
    else:
        bcnodes = np.zeros(nyg+1, dtype=np.intc)
        bcvars = np.zeros((nyg+1,2), dtype=np.intc)
        for j in xrange(nyg+1):
            bcnodes[j] = j*(nxg+1) + 1
            bcvars[j,0] = 1
            bcvars[j,1] = 2

    # Add the boundary conditions to the lists
    mg_bcnodes.append(bcnodes)
    mg_bcvars.append(bcvars)

    for k in xrange(1, nlevels):
        # Compute the number of elements along each coordinate
        # direction
        nx = nxg/2**k
        ny = nyg/2**k

        # Compute the number of nodes/elements
        n = (nx+1)*(ny+1)
        ne = nx*ny

        # Set the boundary conditions
        if problem_type == 'mbb':
            naw = 1
            nbc = ny+1 + naw
            bcnodes = np.zeros(nbc, dtype=np.intc)
            bcvars = np.zeros((nbc,2), dtype=np.intc)

            index = 0
            for j in xrange(ny+1):
                bcnodes[index] = j*(nx+1) + 1
                bcvars[index,0] = 1
                bcvars[index,1] = -1
                index += 1

            for i in xrange(nx+1-naw, nx+1):
                bcnodes[index] = i+1
                bcvars[index,0] = -1
                bcvars[index,1] = 2
                index += 1
        else:
            bcnodes = np.zeros(ny+1, dtype=np.intc)
            bcvars = np.zeros((ny+1,2), dtype=np.intc)
            for j in xrange(ny+1):
                bcnodes[j] = j*(nx+1) + 1
                bcvars[j,0] = 1
                bcvars[j,1] = 2

        mg_bcnodes.append(bcnodes)
        mg_bcvars.append(bcvars)

        # The interpolation stencil
        interpw = [0.5, 1.0, 0.5]

        # All the interpolation weights
        interp = []
        icols = []
        irowp = [1]

        # Set up the interpolation between grids
        for j in xrange(ny+1):
            for i in xrange(nx+1):
                # Compute the interpolation between grid levels
                var = []
                weights = []

                # Take a weighted stencil from the
                # adjacent nodes
                for jj in xrange(-1,2):
                    if (2*j+jj >= 0 and 2*j+jj < 2*ny+1):
                        for ii in xrange(-1,2):
                            if (2*i+ii >= 0 and 2*i+ii < 2*nx+1):
                                var.append(2*i+ii + (2*j+jj)*(2*nx+1) + 1)
                                weights.append(interpw[ii+1]*interpw[jj+1])

                # Normalize the weights
                ws = sum(weights)
                for i in xrange(len(weights)):
                    weights[i] /= ws

                # Add the weights to the icols array and update irowp
                interp.extend(weights)
                icols.extend(var)
                irowp.append(len(interp)+1)

        # Add the intra-grid interpolation
        mg_interp.append(np.array(interp))
        mg_irowp.append(np.array(irowp, dtype=np.intc))
        mg_icols.append(np.array(icols, dtype=np.intc))

        # Set the design-dependent interpolation
        qinterp = []
        qcols = []
        qrowp = [1]

        # Set the interpolation between designs
        for j in xrange(ny):
            for i in xrange(nx):
                # Compute the interpolation between grid levels
                var = []
                weights = []

                # Take a weighted stencil from the adjacent
                # elements
                for jj in xrange(-1,2):
                    if (2*j+jj >= 0 and 2*j+jj < 2*ny):
                        for ii in xrange(-1,2):
                            if (2*i+ii >= 0 and 2*i+ii < 2*nx):
                                var.append(2*i+ii + (2*j+jj)*(2*nx) + 1)
                                weights.append(interpw[ii+1]*interpw[jj+1])

                # Normalize the weights
                ws = sum(weights)
                for i in xrange(len(weights)):
                    weights[i] /= ws

                # Add the weights to the icols array and update irowp
                qinterp.extend(weights)
                qcols.extend(var)
                qrowp.append(len(qinterp)+1)

        # Add the inter-grid interpolation
        mg_qinterp.append(np.array(qinterp))
        mg_qrowp.append(np.array(qrowp, dtype=np.intc))
        mg_qcols.append(np.array(qcols, dtype=np.intc))

    # Create the stiffness matrix
    # ---------------------------
    n = (nxg+1)*(nyg+1)
    ne = nxg*nyg
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
        prob = thickness_opt.ThicknessProblem(conn, X, tconn, tweights,
                                              kmat, dmat, Cmats, qval,
                                              h, G, epsilon,
                                              mg_bcnodes[0], mg_bcvars[0],
                                              elem_area=elem_area, rho=rho)
    else:
        nmats = Cmats.shape[0]
        dmat.mginit(1+nmats, nlevels, rowp, cols,
                    mg_qrowp, mg_qcols, mg_qinterp)

        # Find the element numbers
        elems = np.arange(nxg*nyg, dtype=np.intc) + 1

        # Set up the problem
        prob = multi_opt.MultiProblem(conn, X, tconn, tweights, elems,
                                      kmat, dmat, rho, Cmats,
                                      qval, qxval, h, G, epsilon,
                                      mg_bcnodes[0], mg_bcvars[0])
    return prob

# The dimensions of the problem
Lx = 200.0 # centimeters
Ly = 100.0 # centimeters

nx = 64
ny = 32

if 'tiny' in sys.argv:
    nx = 16
    ny = 8
elif 'small' in sys.argv:
    nx = 32
    ny = 16
elif'medium' in sys.argv:
    nx = 64
    ny = 32
elif 'large' in sys.argv:
    nx = 96
    ny = 48
elif 'xlarge' in sys.argv:
    nx = 128
    ny = 64
elif 'xxlarge' in sys.argv:
    nx = 256
    ny = 128

# Set the problem label
problem_type = 'truss'

# The number of multigrid levels to use
nlevels = int(np.floor(np.log(min(nx, ny))/np.log(2)))
print 'nlevels = ', nlevels

# Set the type of problem flag
thickness_flag = False
ortho_flag = False

angles = []

# Set up the material stiffness/failure properties
if 'thickness' in sys.argv:
    thickness_flag = True

    # Set the material properties (aluminum)
    rho = 2.7 # kg/m3
    E = 70e3 # MPa
    nu = 0.3
    smax = 300.0 # MPa

    C = get_isotropic([E], [nu])
    hm, Gm = get_von_mises([E], [nu], [smax])

    Cmats = C[0,:,:]
    h = hm[0,:]
    G = Gm[0,:,:]

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

# Use RAMP penalization
qval = 5.0
qxval = 2.0
epsilon = 0.1

# Print out the parameters
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

# Create the optimization problem
prob = create_multi_problem(rho, Cmats, qval, qxval, h, G, epsilon,
                            Lx, Ly, nx, ny, nlevels,
                            thickness_flag=thickness_flag,
                            problem_type=problem_type)

# Assign the angles for visualization purposes
prob.angles = angles

# Create the load vector
u = prob.createSolutionVec()
force = prob.createSolutionVec()

# Distribute the force over a portion of the right-hand-side
t_lower_list = []
F = -1500.0*(mat_scale/stress_scale)

# Set the forces
nj = ny/8
for j in xrange(nj):
    force.x[(nx+1)*(j+1)-1, 1] += 0.5*F*(1.0/nj)
    force.x[(nx+1)*(j+2)-1, 1] += 0.5*F*(1.0/nj)

# Set the element domain
nj = ny/8 + 2
index = 1
elems = []
for j in xrange(ny):
    iend = nx
    if j < nj:
        iend = nx-2

    for i in xrange(iend):
        elems.append(i + j*nx + 1)
    for i in xrange(iend, nx):
        t_lower_list.append(i + j*nx)

prob.elems = np.array(elems, dtype=np.intc)

# Create the design variable vector
x = prob.createDesignVec()
lb = x.duplicate()
ub = x.duplicate()

# Set the file prefix
if thickness_flag:
    prefix = 'kona_thickness'
elif 'multi' in sys.argv:
    prefix = 'kona_multi'

prob.ftype = 1
if 'topo' in sys.argv:
    # Set up a topology proble,
    prob.ptype = 1
    prefix += '_topo'
    # Set thickness lower and upper bounds (in centimeters)
    t_lb = 0.1
    t_ub = 10.0
    # Set the design variable values and lower/upper bounds
    lb.x[:,0] = 1.0/t_ub
    ub.x[:,0] = 1.0/t_lb
    x.x[:,0] = 2.0/(0.02*t_lb + 0.98*t_ub)
else:
    # Set up a continuous thickness proble
    prob.ptype = 3
    # Set thickness lower and upper bounds (in centimeters)
    t_lb = 0.1
    t_ub = 10.0
    # Set the design variable values and lower/upper bounds
    lb.x[:,0] = t_lb
    ub.x[:,0] = t_ub
    x.x[:,0] = (0.02*t_lb + 0.98*t_ub)/2

# Compute the initial state variable values
kmat = prob.getKMat()
prob.assembleKMat(x, kmat)
kmat.factor()

# Solve the system of equations
linalg.fgmres_solve(kmat, force, u, print_flag=1, rtol=1e-16, atol=1e-8)

# Create the directory if it does not exist
if not os.path.isdir(prefix):
    os.mkdir(prefix)

prefix += '%s%dx%d'%(os.path.sep, nx, ny)
if not os.path.isdir(prefix):
    os.mkdir(prefix)

# initialize the FSTOPO solver wrapper
solver = kona_opt.FSTopoSolver(
    prob, force, x, lower=lb.x, upper=ub.x,
    num_aggr=0, ks_rho=35., cnstr_scale=False, prefix=prefix)


# optns = {
#     'matrix_explicit'    : True,
#     'verify' : {
#         'primal_vec'     : True,
#         'state_vec'      : True,
#         'dual_vec_eq'    : True,
#         'dual_vec_in'    : True,
#         'gradients'      : True,
#         'pde_jac'        : True,
#         'cnstr_jac_eq'   : True,
#         'cnstr_jac_in'   : True,
#         'red_grad'       : True,
#         'lin_solve'      : True,
#         'out_file'       : 'kona_verify.dat',
#     },
# }

optns = {
    'info_file' : outdir + 'kona_info.dat',
    'hist_file' : outdir + 'kona_hist.dat',
    'max_iter' : 200,
    'opt_tol' : 1e-4,
    'feas_tol' : 1e-4,
    'globalization' : 'trust',

    'trust' : {
        'init_radius' : 1,
        'max_radius' : 5,
        'min_radius' : 1e-4,
    },

    'penalty' : {
        'mu_init' : 0.4,
        'mu_pow' : 0.2,
        'mu_max' : 1e5,
    },

    'rsnk' : {
        'precond'       : 'uzawa',
        # rsnk algorithm settings
        'dynamic_tol'   : False,
        'nu'            : 0.95,
        # reduced KKT matrix settings
        'product_fac'   : 0.001,
        'lambda'        : 0.0,
        'scale'         : 1.0,
        'grad_scale'    : 1.0,
        'feas_scale'    : 1.0,
        # FLECS solver settings
        'krylov_file'   : outdir + 'kona_krylov.dat',
        'subspace_size' : 5,
        'check_res'     : True,
        'rel_tol'       : 1e-4,
        'abs_tol'       : 1e-6,
    },

    'verify' : {
        'primal_vec' : True,
        'state_vec'  : False, 
        'dual_vec'   : True, 
        'gradients'  : True, 
        'pde_jac'    : False,  
        'cnstr_jac'  : True, 
        'red_grad'   : True, 
        'lin_solver' : True, 
        'out_file'   : outdir + 'kona_verify.dat', 
    },    
}


algorithm = kona.algorithms.FLECS_RSNK                #Verifier
optimizer = kona.Optimizer(solver, algorithm, optns)
optimizer.solve()

"""
# initialize Kona memory manager
km = kona.linalg.memory.KonaMemory(solver)
pf = km.primal_factory
sf = km.state_factory
df = km.dual_factory

# request some work vectors
pf.request_num_vectors(3)
sf.request_num_vectors(3)
df.request_num_vectors(3)

# initialize the total derivative matrices
W = kona.linalg.matrices.hessian.LagrangianHessian([pf, sf, df])
A = kona.linalg.matrices.hessian.TotalConstraintJacobian([pf, sf, df])

# trigger memory allocations
km.allocate_memory()

# request vectors for the linearization point
at_design = pf.generate()
at_state = sf.generate()
at_adjoint = sf.generate()
adjoint_rhs = sf.generate()
at_dual = df.generate()
at_slack = df.generate()
# X = kona.linalg.vectors.composite.ReducedKKTVector(
#     kona.linalg.vectors.composite.CompositePrimalVector(
#         at_design, at_slack),
#     at_dual)

# do some matrix aliasing
dRdU = kona.linalg.matrices.common.dRdU
dCdU = kona.linalg.matrices.common.dCdU


def _generate_KKT_vector():
    design = pf.generate()
    slack = df.generate()
    primal = CompositePrimalVector(design, slack)
    dual = df.generate()
    return ReducedKKTVector(primal, dual)

def _generate_primal_vector():
    design = pf.generate()
    slack = df.generate()
    return CompositePrimalVector(design, slack)


# request some input/output vectors for the products
in_design = pf.generate()
out_design = pf.generate()
in_slack = df.generate()
out_dual = df.generate()
out_compPrime = kona.linalg.vectors.composite.CompositePrimalVector(
                out_design, out_dual)
in_compPrime = kona.linalg.vectors.composite.CompositePrimalVector(
                in_design, in_slack)


# set the point at which products will be evaluated
design_file = open('design', 'r')
at_design._data.data.x = pickle.load(design_file)
design_file.close()
dual_file = open('dual', 'r')
dual_vec = pickle.load(dual_file)
dual_file.close()
at_dual._data.x_lower.x = dual_vec[0]
at_dual._data.x_upper.x = dual_vec[1]
at_dual._data.stress.x = dual_vec[2]
slack_file = open('slack', 'r')
slack_vec = pickle.load(slack_file)
slack_file.close()
at_slack._data.x_lower.x = slack_vec[0]
at_slack._data.x_upper.x = slack_vec[1]
at_slack._data.stress.x = slack_vec[2]


X = kona.linalg.vectors.composite.ReducedKKTVector(
    kona.linalg.vectors.composite.CompositePrimalVector(
        at_design, at_slack),
    at_dual)



# compute states
at_state.equals_primal_solution(at_design)

# perform an adjoint solution for the Lagrangian
adjoint_rhs.equals_objective_partial(at_design, at_state)
dCdU(at_design, at_state).T.product(at_dual, at_adjoint)
adjoint_rhs.plus(at_adjoint)
adjoint_rhs.times(-1.)
dRdU(at_design, at_state).T.solve(adjoint_rhs, at_adjoint)

# linearize the Kona matrix objects
W.linearize(X, at_state, at_adjoint)
# A.linearize(at_design, at_state)
A.linearize(X, at_state)

# initialize containers for the explicit matrices
num_design = at_design._data.data.x.shape[0]
num_stress = at_dual._data.stress.x.shape[0]
num_dual = num_design + num_design + num_stress
# W_full = np.zeros((num_design, num_design))
# A_full = np.zeros((num_dual, num_design))
W_full = np.zeros((num_design+num_dual, num_design+num_dual))
A_full = np.zeros((num_dual, num_design+num_dual))

# loop over design variables and start assembling the matrices
for i in xrange(num_design+num_dual):
    print 'Evaluating design var:', i+1
    # set the input vector so that we only pluck out one column of the matrix
    in_design.equals(0.0)
    in_slack.equals(0.0)
    
    if i < num_design:
        in_design._data.data.x[i, 0] = 1.

    elif i < 2*num_design:
        in_slack._data.x_lower.x[i-num_design, 0] = 1.
    elif i < 3*num_design:
        in_slack._data.x_upper.x[i-2*num_design, 0] = 1.
    else: 
        in_slack._data.stress.x[i-3*num_design, 0] = 1.


    #~~~~~~~~~~~~~~~~ Assembling A ~~~~~~~~~~~~~~~~~~~~
    # perform the Constraint Jacobian product and store
    A.slack.product(in_compPrime, out_dual)  

    A_full[:num_design, i] = out_dual._data.x_lower.x[:, 0]
    A_full[num_design:2*num_design, i] = out_dual._data.x_upper.x[:, 0]
    A_full[2*num_design:, i] = out_dual._data.stress.x[:, 0]
    #~~~~~~~~~~~~~~~~ Assembling A ~~~~~~~~~~~~~~~~~~~~


    #~~~~~~~~~~~~~~~~ Assembling W ~~~~~~~~~~~~~~~~~~~~
    W.product(in_compPrime, out_compPrime)

    if i < num_design:
        W_full[:num_design, i] = out_compPrime._design._data.data.x[:, 0]

    elif i < 2*num_design:
        W_full[num_design:2*num_design, i] = out_compPrime._slack._data.x_lower.x[:, 0]
    elif i < 3*num_design:
        W_full[2*num_design:3*num_design, i] = out_compPrime._slack._data.x_upper.x[:, 0]
    else: 
        W_full[3*num_design:, i] = out_compPrime._slack._data.stress.x[:, 0]
    #~~~~~~~~~~~~~~~~ Assembling A ~~~~~~~~~~~~~~~~~~~~


# store the matrices into a file
W_file = open('hessian_wtS', 'w')
pickle.dump(W_full, W_file)
W_file.close()
A_file = open('cnstr_jac_test', 'w')
pickle.dump(A_full, A_file)
A_file.close()

 
Note: This file assembles the matrices W - lagrangian Hessign and A - total constraint Jacobian 
at the imported point "design, slack, dual", both with slack terms included




# # -------------- Exporting dLdX RHS ---------------------- # 
# # -------------------------------------------------------- # 


primal_work = pf.generate()
dual_work = df.generate()

dLdX = _generate_KKT_vector()
dLdX.equals_KKT_conditions(X, at_state, at_adjoint, primal_work, dual_work)

dLdX.times(-1.)

RHS = np.zeros((num_design+num_dual+num_dual, 1))

RHS[:num_design] = dLdX._primal._design._data.data.x[:]

RHS[num_design:2*num_design]   = dLdX._primal._slack._data.x_lower.x[:]
RHS[2*num_design:3*num_design] = dLdX._primal._slack._data.x_upper.x[:]
RHS[3*num_design:4*num_design] = dLdX._primal._slack._data.stress.x[:]

RHS[4*num_design:5*num_design] = dLdX._dual._data.x_lower.x[:]
RHS[5*num_design:6*num_design] = dLdX._dual._data.x_upper.x[:]
RHS[6*num_design:]             = dLdX._dual._data.stress.x[:]

# store the RHS to a file
rhs_file = open('kkt_rhs_wtS','w')
pickle.dump(RHS, rhs_file)
rhs_file.close()
"""

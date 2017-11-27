from pyamg.krylov import fgmres
from scipy.sparse.linalg import LinearOperator
import pickle
import pdb 
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt


# --------------- Preparing Solver ---------------
## --------  Verifying KKT conditions, whether Feasibility and Optimality has been reduced ----------- ##
# ----------------------------------------------------------------------------------------------
# initialize Kona memory manager
import sys
import os
import numpy as np
import pickle
import pdb, time
import matplotlib.pyplot as plt
import matplotlib.pylab as pylt
import scipy.sparse as sps
# Import the material routines
from fstopo.material import *

# Import the fortran-level routines
import fstopo.sparse as sparse

# Import the python-level objects
import fstopo.linalg as linalg
import fstopo.problems.multi_opt as multi_opt
import fstopo.problems.thickness_opt as thickness_opt
import fstopo.problems.kona_opt as kona_opt

# Import Kona Optimization Library
import kona
from kona.linalg.vectors.composite import ReducedKKTVector
from kona.linalg.vectors.composite import CompositePrimalVector


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
    prefix = 'test'
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
    # x.x[:,0] = (0.02*t_lb + 0.98*t_ub)
    # x.x[:,0] = (t_ub-t_lb)*np.random.random_sample((len(x.x[:,0]), )) + t_lb

# Compute the initial state variable values
kmat = prob.getKMat()
prob.assembleKMat(x, kmat)
kmat.factor()

# Solve the system of equations
linalg.fgmres_solve(kmat, force, u, print_flag=1, rtol=1e-16, atol=1e-8)

#pdb.set_trace()
# Create the directory if it does not exist
if not os.path.isdir(prefix):
    os.mkdir(prefix)

# prefix += '%s%dx%d'%(os.path.sep, nx, ny)
prefix += '%stemp'%(os.path.sep)

if not os.path.isdir(prefix):
    os.mkdir(prefix)

# initialize the FSTOPO solver wrapper
solver = kona_opt.FSTopoSolver(
    prob, force, x, lower=lb.x, upper=ub.x,
    num_aggr=0, ks_rho=35., cnstr_scale=False, prefix=prefix)

# ------------------------------------------------

class KKT_COND:
    def __init__(self, design, slack, dual):
        km = kona.linalg.memory.KonaMemory(solver)
        self.primal_factory = km.primal_factory
        self.state_factory = km.state_factory
        self.ineq_factory = km.ineq_factory

        # request some work vectors
        self.primal_factory.request_num_vectors(15)
        self.state_factory.request_num_vectors(15)
        self.ineq_factory.request_num_vectors(15)

        self.design = design
        self.slack = slack
        self.dual = dual

        # trigger memory allocations
        km.allocate_memory()

    def _generate_kkt(self):
        prim = self.primal_factory.generate()
        slak = self.ineq_factory.generate()        
        dual = self.ineq_factory.generate()
        return ReducedKKTVector(CompositePrimalVector(prim, slak), dual)


    def kkt_condition(self): 
        # request vectors for the linearization point
        at_state = self.state_factory.generate()
        state_work = self.state_factory.generate()
        lag_adj = self.state_factory.generate()

        X = self._generate_kkt()
        dJdX = self._generate_kkt()
         
        X.primal.design.base.data = self.design
        X.primal.slack.base.data = self.slack
        X.dual.base.data = self.dual

        # compute states
        at_state.equals_primal_solution(X.primal)

        # compute the lagrangian adjoint
        lag_adj.equals_lagrangian_adjoint(
            X, at_state, state_work, obj_scale=1.0, cnstr_scale=1.0)
            
        # compute initial KKT conditions
        dJdX.equals_KKT_conditions(
            X, at_state, lag_adj, obj_scale=1.0, cnstr_scale=1.0)

        print  dJdX.norm2, dJdX.primal.design.norm2, dJdX.primal.slack.norm2, dJdX.dual.norm2
        # return dJdX.primal.norm2, dJdX.primal.slack.norm2, dJdX.dual.norm2
        dJdX_data = np.concatenate( (dJdX.primal.design.base.data, dJdX.primal.slack.base.data, dJdX.dual.base.data), axis=0)

        return dJdX_data


# -------------------------------------------------

case = 'tiny'                    # tiny, small, medium
dir_data = './tiny_pc4/'
j = 0
scaled_slack = False


if case is 'tiny':
    num_design = 16*8 
elif case is 'small': 
    num_design = 32*16
elif case is 'medium':  
    num_design = 64*32
                              
num_ineq = 3*num_design

num_kkt = num_design + 2*num_ineq


f_jacobian = open(dir_data+'%i_A_exact'%j,'rb')   
Ag_exact = pickle.load(f_jacobian)
f_jacobian.close()


f_jacobian = open(dir_data+'%i_A_approx'%j,'rb')   
# f_jacobian = open(dir_data+'A_approx','rb')   
Ag_approx = pickle.load(f_jacobian)
f_jacobian.close()

f_hessian   = open(dir_data+'%i_W_exact'%j,'rb')
W_exact = pickle.load(f_hessian)
f_hessian.close()

f_hessian   = open(dir_data+'%i_W_approx'%j,'rb')
# f_hessian   = open(dir_data+'W_approx','rb')
W_approx = pickle.load(f_hessian)
f_hessian.close()

f_dldx = open(dir_data+'%i_dldx'%j,'rb')
dLdX = pickle.load(f_dldx)
f_dldx.close()

As_exact = Ag_exact[-num_design:,:]
As_approx = Ag_approx[-num_design:,:]


f_design   = open(dir_data+'design_%i'%j,'rb')
at_design = pickle.load(f_design)
f_design.close()

f_slack   = open(dir_data+'slack_%i'%j,'rb')
at_slack = pickle.load(f_slack)
f_slack.close()

f_dual   = open(dir_data+'dual_%i'%j,'rb')
at_dual_data = pickle.load(f_dual)
f_dual.close()

at_dual = at_dual_data
at_dual[abs(at_dual_data) < 1e-4] = 1e-4*np.sign( at_dual_data[abs(at_dual_data) < 1e-4] )

# -----------------  LinearOperator and Solve -------------
def mat_vec_kkt(in_vec):
    in_design = in_vec[ : num_design] 
    in_slack = in_vec[num_design :  num_design+num_ineq ] 
    in_dual  = in_vec[num_design + num_ineq : ] 

    out_design = np.dot(W_exact, in_design) + np.dot(Ag_exact.transpose(), in_dual)

    # Scaled Slack
    if scaled_slack is True:
        out_slack = -at_dual*at_slack*in_slack - at_slack*in_dual
        out_dual = np.dot(Ag_exact, in_design) - at_slack*in_slack
    else: 
        out_slack = -at_dual*in_slack - at_slack*in_dual
        out_dual = np.dot(Ag_exact, in_design) - in_slack

    out_vec = np.concatenate( (out_design, out_slack, out_dual), axis=0)
    return out_vec


# ----------------- Defining Preconditioner as LinearOperator and Solve ----------
# With Slack Version!! 
K_As = 0

def mat_vec_SVD_2nd(in_vec):
    u_x = in_vec[ : num_design] 
    u_s = in_vec[num_design :  num_design+num_ineq ] 
    u_g  = in_vec[num_design + num_ineq : ] 

    out_design = np.zeros_like(u_x)
    out_slack  = np.zeros_like(u_s)
    out_dual   = np.zeros_like(u_g)

    # 1)  W
    Ag_Winv_AgT = np.dot(Ag_exact, np.dot(np.linalg.inv(W_approx), Ag_exact.transpose() ))
    # Ag_Winv_AgT = np.dot(Ag,  Ag.transpose() )

    M_full, gam_full, N_full = np.linalg.svd(Ag_Winv_AgT, full_matrices=False)

    K_A = 80
    M = M_full[:, :K_A]    # 76
    gam = gam_full[:K_A]
    N = N_full[:K_A, :]
    # pdb.set_trace()
    sigma = - 1.0/at_slack * at_dual

    gam_N = np.dot(np.diag(gam), N)
    

    # Step 1: solve  v_g -- out_dual 
    # 2)  W 
    rhs_vg = - u_g + 1.0/at_dual * u_s + np.dot(Ag_exact, np.dot( np.linalg.inv(W_approx), u_x ))
    # rhs_vg = - u_g + 1.0/at_dual * u_s + np.dot(Ag,  u_x )

    core_mat = np.eye( len(gam) ) + np.dot( gam_N, np.dot( np.diag(sigma), M ) )

    core_inv = np.linalg.inv(core_mat)

    v_g1 = np.dot( gam_N , sigma*rhs_vg ) 
    v_g2 = np.dot( core_inv, v_g1)
    v_g3 = -sigma * np.dot(M, v_g2)
    v_g = sigma * rhs_vg + v_g3

    # Step 2: solve  v_s -- out_slack
    # Scaled Slack block
    if scaled_slack is True:
        v_s = -1.0/at_dual * v_g - 1.0/(at_dual*at_slack)*u_s
    else:
        v_s = 1.0/sigma * v_g - 1.0/at_dual*u_s

    # Step 3: solve  v_x -- out_design
    # 3) W 
    v_x1 = -np.dot( Ag_exact.transpose(), v_g) + u_x 
    v_x = sp.linalg.lu_solve(sp.linalg.lu_factor(W_approx), v_x1) 
    # v_x = v_x1

    out_vec = np.concatenate( (v_x, v_s, v_g), axis=0)
    return out_vec


def mat_vec_SVD_1st(in_vec):
    global K_As
    # # svd_pc4 as described in the paper

    u_x = in_vec[ : num_design] 
    u_s = in_vec[num_design :  num_design+num_ineq ] 
    u_g  = in_vec[num_design + num_ineq : ] 

    out_design = np.zeros_like(u_x)
    out_slack  = np.zeros_like(u_s)
    out_dual   = np.zeros_like(u_g)

    # 1)  rhs_vx 
    sigma = - 1.0/at_slack * at_dual
    sigma_l = sigma[:num_design]
    sigma_u = sigma[num_design:2*num_design]
    sigma_s = sigma[-num_design:]

    work1 = sigma * (-u_g + 1.0/at_dual * u_s )
    rhs_vx = u_x - np.dot(Ag_approx.transpose(), work1)                           # approx or exact ? 

    # # 2) decompose LHS matrix

    # 2) decompose LHS, svd on whole AsT_SigS_As
    AsT_SigS_As = np.dot(As_approx.transpose(), np.dot(np.diag(sigma_s), As_approx))      # As_approx   As_exact
    
    M_full, gam_full, N_full = np.linalg.svd(AsT_SigS_As, full_matrices=False)
    M = M_full[:, :K_As]     
    gam = gam_full[:K_As]
    N = N_full[:K_As, :]
    AsT_SigS_As_svd = np.dot(M, np.dot(np.diag(gam), N))

    # 3) LHS matrix
    W_diag = 0.1*np.ones(num_design)       # np.diag(W_approx)   
    LHS = np.diag( W_diag + sigma_l + sigma_u )  + AsT_SigS_As_svd 
    v_x = sp.linalg.lu_solve(sp.linalg.lu_factor(LHS), rhs_vx) 


    # 4) solve v_g
    rhs_g = - u_g + 1.0/at_dual * u_s + np.dot(Ag_approx, v_x)
    v_g = np.dot(np.diag(sigma), rhs_g)

    # 5) solve v_s 
    if scaled_slack is True:
        v_s = -1.0/at_dual * v_g - 1.0/(at_dual*at_slack)*u_s
    else:
        v_s = 1.0/sigma * v_g - 1.0/at_dual*u_s    

    out_vec = np.concatenate( (v_x, v_s, v_g), axis=0)

    return out_vec


def mat_vec_approx_adjoint(in_vec):

    if scaled_slack is True:
        KKT_full = np.vstack([np.hstack([W_approx,  np.zeros((num_design, num_ineq)),  Ag_approx.transpose()]), 
                              np.hstack([np.zeros((num_ineq, num_design)),  -np.diag(at_dual*at_slack), -np.diag(at_slack)]),
                              np.hstack([Ag_approx, -np.diag(at_slack),  np.zeros((num_ineq, num_ineq))]) ])
    else:
        KKT_full = np.vstack([np.hstack([W_approx,  np.zeros((num_design, num_ineq)),  Ag_approx.transpose()]), 
                              np.hstack([np.zeros((num_ineq, num_design)),  -np.diag(at_dual), -np.diag(at_slack)]),
                              np.hstack([Ag_approx, -np.eye(num_ineq),  np.zeros((num_ineq, num_ineq))]) ])    

    out_vec = sp.linalg.lu_solve(sp.linalg.lu_factor(KKT_full), in_vec)

    return out_vec


K = LinearOperator((num_kkt, num_kkt), matvec=mat_vec_kkt  )


# M_pc = LinearOperator((num_kkt, num_kkt), matvec=mat_vec_SVD_2nd  )
M_pc = LinearOperator((num_kkt, num_kkt), matvec=mat_vec_SVD_1st  )
# M_pc = LinearOperator((num_kkt, num_kkt), matvec=mat_vec_approx_adjoint  )


#------------------------ Actually solving using the preconditioner --------------
x = np.zeros(dLdX.shape)
x_pc = np.zeros(dLdX.shape)

fac = 1.0

res_hist = []
(x,flag) = fgmres(K, -dLdX,  maxiter=20, tol=1e-4, residuals=res_hist)      


K_As = 10
res_hist_pc10 = []
(x_pc10,flag) = fgmres(K, -fac*dLdX, M=M_pc, maxiter=20, tol=1e-4, residuals=res_hist_pc10)

K_As = 20
res_hist_pc20 = []
(x_pc20,flag) = fgmres(K, -fac*dLdX, M=M_pc, maxiter=20, tol=1e-4, residuals=res_hist_pc20)

K_As = 30
res_hist_pc30 = []
(x_pc30,flag) = fgmres(K, -fac*dLdX, M=M_pc, maxiter=20, tol=1e-4, residuals=res_hist_pc30)




# -------------------- Plotting --------------------------
# -------------------- plotting ---------------------
# plot the data
# ms = markersize
# mfc = markerfacecolor     mec = 'k'
# mew = markeredgewidth
axis_fs = 12 # axis title font size
axis_lw = 1.0 # line width used for axis box, legend, and major ticks
label_fs = 10 # axis labels' font size


fig = plt.figure(figsize=(7,4), facecolor=None)
ax = fig.add_subplot(111)

line1, = ax.semilogy(res_hist/res_hist[0], '-k^', linewidth=1.0, ms=6.0, mfc='w', mew=1.0) 
line2, = ax.semilogy(res_hist_pc10/res_hist_pc10[0], '-kv', linewidth=1.0, ms=6.0, mfc='w', mew=1.0)  
line3, = ax.semilogy(res_hist_pc20/res_hist_pc20[0], '-ko', linewidth=1.0, ms=6.0, mfc='w', mew=1.0) 
line4, = ax.semilogy(res_hist_pc30/res_hist_pc30[0], '-ks', linewidth=1.0, ms=6.0, mfc='w', mew=1.0)  

ax.set_position([0.15, 0.13, 0.80, 0.83])                                # position relative to figure edges    mfc=(0.35, 0.35, 0.35)
ax.set_xlabel('Krylov Iteration', fontsize=axis_fs, weight='bold')

# xmax = max(len(res_hist), len(res_hist_pc10), len(res_hist_pc20), len(res_hist_pc30))

ax.set_ylabel('Relative Residual', fontsize=axis_fs, weight='bold')
ax.grid(which='major', axis='y', linestyle='--')
ax.set_axisbelow(True) # grid lines are plotted below
plt.tick_params(labelsize=axis_fs)
rect = ax.patch # a Rectangle instance
#rect.set_facecolor('white')
#rect.set_ls('dashed')
rect.set_linewidth(axis_lw)
rect.set_edgecolor('k')


# ticks on bottom and left only
ax.xaxis.tick_bottom() # use ticks on bottom only
ax.yaxis.tick_left()
for line in ax.xaxis.get_ticklines():
    line.set_markersize(6) # length of the tick
    line.set_markeredgewidth(axis_lw) # thickness of the tick
for line in ax.yaxis.get_ticklines():
    line.set_markersize(6) # length of the tick
    line.set_markeredgewidth(axis_lw) # thickness of the tick
for label in ax.xaxis.get_ticklabels():
    label.set_fontsize(label_fs)
for label in ax.yaxis.get_ticklabels():
    label.set_fontsize(label_fs)

# define and format the minor ticks
ax.xaxis.set_ticks(np.arange(0, 21, 2.0),minor=False)
ax.xaxis.set_tick_params(which='minor', length=3, width=2.0*axis_lw/3.0)

# # ----------- text box --------------
# ax.yaxis.set_ticks(np.logspace(-8, 2, num=11))
# ax.yaxis.set_tick_params(which='minor', length=3, width=2.0*axis_lw/3.0)
# textstr = 'Number of Design : %i'%num_design 
# ax.text(xmax*0.6, 15, textstr, fontsize=label_fs, weight='bold')

# if case is 'tiny':
leg_size = 6

leg = ax.legend([line1, line2, line3, line4], ['noPC', 'PC SVD rank 10', 'PC SVD rank 20', 'PC SVD rank 30'], \
                loc=(0.01, 0.01), numpoints=1, prop={'size':leg_size},  borderpad=0.75, handlelength=4)
# leg = ax.legend([line1,  line3], ['noPC', 'PC SVD rank 20'], \
#                 loc=(0.65, 0.03), numpoints=1, prop={'size':leg_size},  borderpad=0.75, handlelength=4)

rect = leg.get_frame()
rect.set_linewidth(axis_lw)
for t in leg.get_texts():
    t.set_fontsize(10)    # the legend text fontsize

plt.show() 

fig_name = dir_data  + 'svd_ranks.eps' 
fig.savefig(fig_name, format='eps', dpi=1200)    
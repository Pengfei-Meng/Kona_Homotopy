import pickle
import matplotlib.pyplot as plt
import numpy as np
import kona

# -------- Read Singular Values from Lanczos SVD Decomposition ------------
# outdir = 'pc_svd'
# outer = 22
# inner_iters = 50
# max_iter = 20

# for j in xrange(1,max_iter):    # inner_iters
#     file_name = './test/' + outdir + '/sinvals_%d_%d.pkl'%(outer, j)
#     sins = pickle.load( open(file_name, 'rb'))
#     plt.plot(sins, 's--', label='corr %i'%(j))

# plt.title('Singular Value of A at mu = 0.0, Lanczos SVD Decomposition ')
# plt.legend(bbox_to_anchor=(1.05, 1), loc=1, borderaxespad=0.)
# plt.xticks(np.arange(0, max_iter, 1))
# plt.show()



# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# ------------------------------------------------------
# Extracting explicit W-hessian, A-constraintJacobian from the problem
# initialize Kona memory manager

km = kona.linalg.memory.KonaMemory(solver)
pf = km.primal_factory
sf = km.state_factory
df = km.ineq_factory

# request some work vectors
pf.request_num_vectors(15)
sf.request_num_vectors(15)
df.request_num_vectors(15)

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
X = kona.linalg.vectors.composite.ReducedKKTVector(
    kona.linalg.vectors.composite.CompositePrimalVector(
        at_design, at_slack),
    at_dual)

# do some matrix aliasing
dRdU = kona.linalg.matrices.common.dRdU
dCdU = kona.linalg.matrices.common.dCdU

# request some input/output vectors for the products
in_design = pf.generate()
out_design = pf.generate()
out_dual = df.generate()



outdir = 'pc_svd'    # pc_eye
inner_iters = 50
max_iter = 20

for j in xrange(0,max_iter):    # inner_iters
    # set the point at which products will be evaluated
    file_design = './test/' + outdir + '/design_%i'%j
    file_dual = './test/' + outdir + '/dual_%i'%j
    file_slack = './test/' + outdir + '/slack_%i'%j
    file_hessian = './test/' + outdir + '/hessian_%i'%j
    file_A = './test/' + outdir + '/cnstrA_%i'%j

    design_file = open(file_design, 'r')
    at_design.base.data = pickle.load(design_file)
    design_file.close()
    dual_file = open(file_dual, 'r')
    dual_vec = pickle.load(dual_file)
    dual_file.close()
    at_dual.base.data = dual_vec

    slack_file = open(file_slack, 'r')
    slack_vec = pickle.load(slack_file)
    slack_file.close()
    at_slack.base.data = slack_vec

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
    A.linearize(at_design, at_state)

    # initialize containers for the explicit matrices
    num_design = len(at_design.base.data)
    num_stress = len(at_dual.base.data)/3
    num_dual = num_design + num_design + num_stress
    W_full = np.zeros((num_design, num_design))
    A_full = np.zeros((num_dual, num_design))

    # loop over design variables and start assembling the matrices
    for i in xrange(num_design):
        print 'Evaluating design var:', i+1
        # set the input vector so that we only pluck out one column of the matrix
        in_design.equals(0.0)
        in_design.base.data[i] = 1.
        # perform the Lagrangian Hessian product and store
        # W.multiply_W(in_design, out_design)
        # W_full[:, i] = out_design.base.data
        # perform the Constraint Jacobian product and store
        A.approx.product(in_design, out_dual)
        A_full[:, i] = out_dual.base.data
        # A_full[num_design:2.*num_design, i] = out_dual._data.x_upper.x[:, 0]
        # A_full[2.*num_design:, i] = out_dual._data.stress.x[:, 0]


    # # store the matrices into a file
    # W_file = open(file_hessian, 'w')
    # pickle.dump(W_full, W_file)
    # W_file.close()
    # A_file = open(file_A, 'w')
    # pickle.dump(A_full, A_file)
    # A_file.close()

    #---- the following few lines are copied from the block below
    u,sins_A,v = np.linalg.svd(A_full)
    plt.plot(sins_A[:20], 'o--', label='corr %i'%(j))

plt.title('Singular Value of A at mu = 0.0, Using explicit matrices ')
plt.legend(bbox_to_anchor=(1.05, 1), loc=1, borderaxespad=0.)
plt.xticks(np.arange(0, max_iter, 1))
plt.show()

# # -------- Read Singular Values from Full A ------------
# outdir = 'pc_svd'    # pc_eye
# max_iter = 20

# for j in xrange(1,max_iter):    # inner_iters
#     file_name = './test/' + outdir + '/cnstrA_%i'%j
#     A = pickle.load(open(file_name, 'r'))
#     u,sins_A,v = np.linalg.svd(A)
#     plt.plot(sins_A[:20], 'o--', label='corr %i'%(j))

# plt.title('Singular Value of A at mu = 0.0, Using explicit matrices ')
# plt.legend(bbox_to_anchor=(1.05, 1), loc=1, borderaxespad=0.)
# plt.xticks(np.arange(0, max_iter, 1))
# plt.show()





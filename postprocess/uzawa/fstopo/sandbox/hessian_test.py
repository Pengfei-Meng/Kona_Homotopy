import numpy as np
import scipy.sparse as sps
import pickle
# import matplotlib
# matplotlib.use('GTKAgg')
# import matplotlib.pylab as plt

def null(A, rtol=1e-5):
    u, s, vH = np.linalg.svd(A)
    rank = (s > rtol*s[0]).sum()
    return rank, vH[rank:].T.copy()

# print 'Loading Constraint Jacobian...'
# cnstr_file = open('cnstr_jac', 'r')
# A = pickle.load(cnstr_file)
# print A.shape
# cnstr_file.close()

print 'Loading Lagrangian Hessian...'
hess_file = open('hessian', 'r')
W = pickle.load(hess_file)
print W.shape
hess_file.close()

print 'Loading slacks and duals...'
dual_file = open('dual', 'r')
lamb = pickle.load(dual_file)
lamb = np.concatenate((lamb[0][:,0], lamb[1][:,0], lamb[2][:,0]), axis=0)
count = 0
for i in xrange(len(lamb)):
    if lamb[i] > 1e-3:
        count += 1
        print 'positive found!', lamb[i]
print 'total positives:', count
print lamb.shape
dual_file.close()
slack_file = open('slack', 'r')
slack = pickle.load(slack_file)
slack = np.concatenate((slack[0][:,0], slack[1][:,0], slack[2][:,0]), axis=0)
print slack.shape
slack_file.close()

# print 'Assembling the constraint matrix...'
# sigma = np.diag(np.exp(slack))
# A_full = np.concatenate((A, -sigma), axis=1)
# print A_full.shape
# plt.spy(sps.csr_matrix(A_full), precision=1e-6, marker='.', markersize=4)
# plt.savefig('A_sparse.png')

print 'Assembling the hessian matrix...'
sigma_lamb = np.diag(np.exp(slack)*lamb)
off_diag = np.zeros((sigma_lamb.shape[0], W.shape[1]))
W_left = np.concatenate((W, off_diag), axis=0)
print W_left.shape
# print np.linalg.norm(W_left[5000, :])
# plt.spy(sps.csc_matrix(W_left), precision=1e-6, marker='.', markersize=1)
# plt.savefig('W_left.png')
W_right = np.concatenate((off_diag.T, -sigma_lamb), axis=0)
print W_right.shape
# plt.spy(sps.csc_matrix(W_right), precision=1e-6, marker='.', markersize=1)
# plt.savefig('W_right.png')
W_full = np.concatenate((W_left, W_right), axis=1)
print W_full.shape

# import pdb; pdb.set_trace()

W_file = open('hessian_withSlack', 'w')
pickle.dump(W_full, W_file)
W_file.close()

# plt.spy(sps.csc_matrix(W_full), precision=1e-6, marker='.', markersize=1)
# plt.savefig('W_sparse.png')

# print 'Calculating null space...'
# rank, Z = null(A_full)
# print rank
# print Z.shape

# print 'Projecting the hessian...'
# ZWZ = np.dot(Z.T, np.dot(W_full, Z))
# print ZWZ.shape

# print 'Calculating W eigenvalues...'
# W_eigs, _ = np.linalg.eig(ZWZ)
# idx = W_eigs.real.argsort()
# W_eigs = W_eigs[idx]
# print 'Abs-min eigval:', min(abs(W_eigs))

# print 'Plotting...'
# f, axarr = plt.subplots(2)
# axarr[0].plot(np.real(W_eigs))
# axarr[0].set_title('Z^T WZ eigs -- Real Component')
# axarr[1].plot(np.imag(W_eigs))
# axarr[1].set_title('Z^T WZ eigs -- Im Component')
# plt.savefig('W_eigs.png')

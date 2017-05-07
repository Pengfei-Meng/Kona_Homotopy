import pdb
import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pylab as pylt
import scipy.sparse as sps

from pyamg.krylov import fgmres
from scipy.sparse.linalg import LinearOperator

np.set_printoptions(threshold=np.nan)

# to load the Hessian, Stress A explicitly, test manual methods for 
# reducing its condition numbers

case = 'tiny'                                 # tiny, small, medium
dir_data = '../test/eye_' + case + '/'

j = 0

if case is 'tiny':
    num_design = 16*8 
elif case is 'small': 
    num_design = 32*16
elif case is 'medium':  
    num_design = 64*32
                              
num_ineq = 3*num_design


f_jacobian = open(dir_data+'%i_A_exact'%j,'rb')  
Ag_exact = pickle.load(f_jacobian)
f_jacobian.close()

f_jacobian = open(dir_data+'%i_A_approx'%j,'rb')   
Ag_approx = pickle.load(f_jacobian)
f_jacobian.close()

f_dldx = open(dir_data+'%i_dldx'%j,'rb')
dLdX = pickle.load(f_dldx)
f_dldx.close()


A_stress_exact = Ag_exact[2*num_design:,:]
A_stress_approx = Ag_approx[2*num_design:,:]
b = dLdX[-num_design:]


# ---- study A_stress_approx ----

print 'np.linalg.cond(A): ', np.linalg.cond(A_stress_approx)

u,sins_A,v = np.linalg.svd(Ag_exact[2*num_design:, :],  full_matrices=0)   
k = 40
U = u[:,:k]
SV = sins_A[:k]
V = v[:k, :]

print 'sv[0]/sv[-1]: ', sins_A[0]/sins_A[-1]
A_stress_SVD = np.dot(U, np.dot(np.diag(SV), V))

A_stress_SVDfull = A_stress_SVD
A_stress_SVDfull[k:,k:] = np.eye(num_design - k) * SV[-1]
pdb.set_trace()

# # The following plots show that SVD approximation is valid approximation for A 
# fig1 = pylt.figure()
# M = sps.csr_matrix(A_stress_SVD)
# pylt.spy(A_stress_SVD, precision=1e-1, marker='.', markersize=1)
# pylt.title(case + ' SVD A exact 1e-1')

# fig2 = pylt.figure()
# M = sps.csr_matrix(A_stress_SVD)
# pylt.spy(A_stress_SVD, precision=1e-2, marker='.', markersize=1)
# pylt.title(case + ' SVD A exact 1e-2')

# fig3 = pylt.figure()
# M = sps.csr_matrix(A_stress_SVD)
# pylt.spy(A_stress_SVD, precision=1e-3, marker='.', markersize=1)
# pylt.title(case + ' SVD A exact 1e-3')

# fig4 = pylt.figure()
# M = sps.csr_matrix(A_stress_SVD)
# pylt.spy(A_stress_SVD, precision=1e-4, marker='.', markersize=1)
# pylt.title(case + ' SVD A exact 1e-4')


# fig5 = pylt.figure()
# M = sps.csr_matrix(A_stress_exact)
# pylt.spy(A_stress_exact, precision=1e-1, marker='.', markersize=1)
# pylt.title(case + ' Sparsity Stress A exact 1e-1')

# fig6 = pylt.figure()
# M = sps.csr_matrix(A_stress_exact)
# pylt.spy(A_stress_exact, precision=1e-2, marker='.', markersize=1)
# pylt.title(case + ' Sparsity Stress A exact 1e-2')

# fig7 = pylt.figure()
# M = sps.csr_matrix(A_stress_exact)
# pylt.spy(A_stress_exact, precision=1e-3, marker='.', markersize=1)
# pylt.title(case + ' Sparsity Stress A exact 1e-3')

# fig8 = pylt.figure()
# M = sps.csr_matrix(A_stress_exact)
# pylt.spy(A_stress_exact, precision=1e-4, marker='.', markersize=1)
# pylt.title(case + ' Sparsity Stress A exact 1e-4')


# pylt.show()


def mat_vec_A(in_vec):
    return np.dot(A_stress_approx, in_vec)

# --------- testing on SVD approximation --------------
def mat_vec_A_svd(in_vec):
    return np.dot(A_stress_SVD, in_vec)


K = LinearOperator((num_design, num_design), matvec=mat_vec_A  )
M_pc = LinearOperator((num_design, num_design), matvec=mat_vec_A_svd )


#------------------------ Actually solving using the preconditioner --------------

res_hist = []
(x,flag) = fgmres(K, b,  maxiter=20, tol=1e-6, residuals=res_hist)      

# res_hist_svd = []
# (x_svd,flag) = fgmres(M_pc, b, maxiter=20, tol=1e-6, residuals=res_hist_svd)

res_hist_svd = []
(x_svd,flag) = fgmres(K, b, M=M_pc, maxiter=20, tol=1e-6, residuals=res_hist_svd)

res_hist = res_hist/res_hist[0]
res_hist_svd = res_hist_svd/res_hist_svd[0]

fig1 = plt.figure()

plt.semilogy(range(len(res_hist)), res_hist, 'ro', range(len(res_hist_svd)), res_hist_svd, 'bv')
plt.ylabel('Residual History')
plt.axis([0, len(res_hist)+1 , 0, max(res_hist) +0.1])
plt.show()
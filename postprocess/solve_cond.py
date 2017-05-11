import pdb
import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pylab as pylt
import scipy.sparse as sps

# to load the Hessian, Stress A explicitly, test manual methods for 
# reducing its condition numbers

case = 'tiny'                                 # tiny, small, medium
# dir_data = '../test/eye_' + case + '/'
dir_data = '../test/adj_0reg/'

j = 5

if case is 'tiny':
    num_design = 16*8 
elif case is 'small': 
    num_design = 32*16
elif case is 'medium':  
    num_design = 64*32
                              
num_ineq = 3*num_design


f_dual = open(dir_data+'dual_%i'%j,'rb')
at_dual = pickle.load(f_dual)
f_dual.close()

f_jacobian = open(dir_data+'%i_A_exact'%j,'rb')  
Ag_exact = pickle.load(f_jacobian)
f_jacobian.close()

f_jacobian = open(dir_data+'%i_A_approx'%j,'rb')   
Ag_approx = pickle.load(f_jacobian)
f_jacobian.close()

f_hessian   = open(dir_data+'%i_W_approx'%j,'rb')
W_approx = pickle.load(f_hessian)
f_hessian.close()

f_hessian   = open(dir_data+'%i_W_exact'%j,'rb')
W_exact = pickle.load(f_hessian)
f_hessian.close()

f_dldx = open(dir_data+'%i_dldx'%j,'rb')
dLdX = pickle.load(f_dldx)
f_dldx.close()


#-------------- Plotting ----------------
fig1 = plt.figure()
ax1 = fig1.add_subplot(121)
ax2 = fig1.add_subplot(122)

u,sins_A,v = np.linalg.svd(Ag_exact[2*num_design:, :])   
l1 = ax1.plot(sins_A[:80], 'o--', label='corr %i'%(j))

u_a,sins_A_a,v_a = np.linalg.svd(Ag_approx[2*num_design:, :])    
l2 = ax2.plot(sins_A_a[:80], 'o--', label='corr %i'%(j))

fig1.suptitle('SVs of A_full at mu = 0.0, explicit mat ')
ax1.set_title("exact product")
ax2.set_title("approx product")
plt.legend(bbox_to_anchor=(1.05, 1), loc=1, borderaxespad=0.)


fig2 = pylt.figure()
M = sps.csr_matrix(Ag_exact[2*num_design:, :])
pylt.spy(M, precision=1e-1, marker='.', markersize=1)
pylt.title(case + ' Sparsity Stress A exact 1e-1')

fig22 = pylt.figure()
M = sps.csr_matrix(Ag_exact[2*num_design:, :])
pylt.spy(M, precision=1e-2, marker='.', markersize=1)
pylt.title(case + ' Sparsity Stress A exact 1e-2')

fig23 = pylt.figure()
M = sps.csr_matrix(Ag_exact[2*num_design:, :])
pylt.spy(M, precision=1e-3, marker='.', markersize=1)
pylt.title(case + ' Sparsity Stress A exact 1e-3')


fig3 = pylt.figure()
M = sps.csr_matrix(W_exact)
pylt.spy(M, precision=1e-1, marker='.', markersize=1)
pylt.title(case + ' Sparsity W exact 1e-1')

# fig4 = pylt.figure()
# M = sps.csr_matrix(W_approx)
# pylt.spy(M, precision=1e-1, marker='.', markersize=1)
# pylt.title('sparsity W approx 1e-1')

fig5 = pylt.figure()
M = sps.csr_matrix(W_exact)
pylt.spy(M, precision=1e-2, marker='.', markersize=1)
pylt.title(case + ' Sparsity W exact 1e-2')

fig6 = pylt.figure()
M = sps.csr_matrix(W_exact)
pylt.spy(M, precision=1e-3, marker='.', markersize=1)
pylt.title(case + ' Sparsity W exact 1e-3')

# fig6 = pylt.figure()
# M = sps.csr_matrix(W_exact)
# pylt.spy(M, precision=1e-4, marker='.', markersize=1)
# pylt.title(case + ' Sparsity W exact 1e-4')

# fig6 = pylt.figure()
# M = sps.csr_matrix(W_exact)
# pylt.spy(M, precision=1e-5, marker='.', markersize=1)
# pylt.title(case + ' Sparsity W exact 1e-5')

pylt.show()
plt.show()


pdb.set_trace()

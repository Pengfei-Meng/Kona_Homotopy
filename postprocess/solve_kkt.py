from pyamg.krylov import fgmres
from scipy.sparse.linalg import LinearOperator
import pickle
import pdb 
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt



dir_data = '../test/svd1/'

f_jacobian = open(dir_data+'cnstrA_exact_55','rb')
Ag = pickle.load(f_jacobian)
f_jacobian.close()

f_hessian   = open(dir_data+'hessian_55','rb')
W = pickle.load(f_hessian)
f_hessian.close()

f_dldx = open(dir_data+'dldx_55','rb')
dLdX = pickle.load(f_dldx)
f_dldx.close()


f_design   = open(dir_data+'design55','rb')
at_design = pickle.load(f_design)
f_design.close()

f_slack   = open(dir_data+'slack55','rb')
at_slack = pickle.load(f_slack)
f_slack.close()

f_dual   = open(dir_data+'dual55','rb')
at_dual = pickle.load(f_dual)
f_dual.close()


# -------------------- dimensions -------------------
num_design = 128
num_stress = 128
num_bound = 128*2 

num_dual = 3*num_design

num_kkt = num_design + 2*num_dual

# -----------------  LinearOperator and Solve -------------
def mat_vec_kkt(in_vec):
    in_design = in_vec[ : num_design] 
    in_slack = in_vec[num_design :  num_design+num_dual ] 
    in_dual  = in_vec[num_design + num_dual : ] 


    out_design = np.dot(W, in_design) + np.dot(Ag.transpose(), in_dual)
    out_slack = -at_dual*in_slack - at_slack*in_dual
    out_dual = np.dot(Ag, in_design) - in_slack

    out_vec = np.concatenate( (out_design, out_slack, out_dual), axis=0)
    return out_vec


# ----------------- Defining Preconditioner as LinearOperator and Solve ----------
# With Slack Version!! 
def mat_vec_M(in_vec):
    u_x = in_vec[ : num_design] 
    u_s = in_vec[num_design :  num_design+num_dual ] 
    u_g  = in_vec[num_design + num_dual : ] 

    out_design = np.zeros_like(u_x)
    out_slack  = np.zeros_like(u_s)
    out_dual   = np.zeros_like(u_g)

    Ag_Winv_AgT = np.dot(Ag, np.dot(np.linalg.inv(W), Ag.transpose() ))

    M, gam, N = np.linalg.svd(Ag_Winv_AgT, full_matrices=False)



    sigma = - 1.0/at_slack * at_dual

    gam_N = np.dot(np.diag(gam), N)
    

    # Step 1: solve  v_g -- out_dual 
    rhs_vg = - u_g + 1.0/at_dual * u_s + np.dot(Ag, np.dot( np.linalg.inv(W), u_x ))

    core_mat = np.eye( len(gam) ) + np.dot( gam_N, np.dot( np.diag(sigma), M ) )

    core_inv = np.linalg.inv(core_mat)

    v_g1 = np.dot( gam_N , sigma*rhs_vg ) 
    v_g2 = np.dot( core_inv, v_g1)
    v_g3 = -sigma * np.dot(M, v_g2)
    v_g = sigma * rhs_vg + v_g3

    # Step 2: solve  v_s -- out_slack
    v_s = 1.0/sigma * v_g - 1.0/at_dual*u_s

    # Step 3: solve  v_x -- out_design
    v_x1 = -np.dot( Ag.transpose(), v_g) + u_x 
    v_x = sp.linalg.lu_solve(sp.linalg.lu_factor(W), v_x1) 

    out_vec = np.concatenate( (v_x, v_s, v_g), axis=0)
    return out_vec


K = LinearOperator((num_kkt, num_kkt), matvec=mat_vec_kkt  )

M_pc = LinearOperator((num_kkt, num_kkt), matvec=mat_vec_M  )


#------------------------ Actually solving using the preconditioner --------------

res_hist = []
(x,flag) = fgmres(K, -dLdX,  maxiter=50, tol=1e-6, residuals=res_hist)      

res_hist_pc = []
(x,flag) = fgmres(M_pc, -dLdX, M=M_pc, maxiter=50, tol=1e-6, residuals=res_hist_pc)

print 'Condition of W: ', np.linalg.cond(W)


fig1 = plt.figure()

plt.plot(range(len(res_hist)), res_hist, 'ro', range(len(res_hist_pc)), res_hist_pc, 'bv')
plt.ylabel('Residual History')
plt.axis([0, len(res_hist)+1 , 0, max(res_hist) + 1.0])
plt.show()


# x = mat_vec_M(b)
# print 'norm of design update', np.linalg.norm(x[:128,])
# print 'norm of slack update', np.linalg.norm(x[128:128*4,])
# print 'norm of dual update', np.linalg.norm(x[128*4:,])

# print 'norm of design input', np.linalg.norm(b[:128,])
# print 'norm of slack input', np.linalg.norm(b[128:128*4,])
# print 'norm of dual input', np.linalg.norm(b[128*4:,])

# pdb.set_trace()
# print res_hist
# print len(res_hist)

# res_file = open('res_wtPc_iter500', 'w')
# pickle.dump(res_hist, res_file)
# res_file.close()





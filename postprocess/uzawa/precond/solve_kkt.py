from pyamg.krylov import fgmres
from scipy.sparse.linalg import LinearOperator
import pickle
import pdb 
import numpy as np
import matplotlib.pyplot as plt



f_cnstr_jac = open('DATA/cnstr_jac_wtS','rb')
A_wtS = pickle.load(f_cnstr_jac)
f_cnstr_jac.close()

f_hessian   = open('DATA/hessian_wtS','rb')
W_wtS = pickle.load(f_hessian)
f_hessian.close()

f_cnstr_ja = open('DATA/cnstr_jac_noS','rb')
A_noS = pickle.load(f_cnstr_ja)
f_cnstr_ja.close()

f_hessia   = open('DATA/hessian_noS','rb')
W_noS = pickle.load(f_hessia)
f_hessia.close()

f_rhs = open('DATA/kkt_rhs_wtS','rb')
b = pickle.load(f_rhs)
f_rhs.close()

f_sl = open('DATA/slack','rb')
slack_list = pickle.load(f_sl)
f_sl.close()

f_dl = open('DATA/dual','rb')
dual_list = pickle.load(f_dl)
f_dl.close()

slack = np.concatenate( (slack_list[0], slack_list[1], slack_list[2]), axis=0 )
dual = np.concatenate( (dual_list[0], dual_list[1], dual_list[2]), axis=0 )

Slack_term = W_wtS[128:,128:]
w_eig, v_eig = np.linalg.eigh(Slack_term)


sigma = np.zeros_like(slack)
sigma = np.exp(slack) * dual *(-1.0)

# sigma = -diag(e^s)    sigma_inv = -diag(1/e^s)
sigma_inv = np.zeros_like(slack)
sigma_inv = -1. / np.exp(slack)
sigma_inv.resize((len(sigma_inv), ))
# -------------------- dimensions -------------------
num_design = 128
num_stress = 128
num_bound = 128*2 

num_dual= num_bound + num_stress
num_prime = num_design + num_dual


# -----------------  LinearOperator and Solve -------------
def mat_vec_kkt(in_vec):
    in_prime = in_vec[:num_prime] 
    in_dual  = in_vec[num_prime:] 

    out_prime = np.zeros_like(in_prime)
    out_dual  = np.zeros_like(in_dual)

    out_prime = np.dot(W_wtS, in_prime) + np.dot(A_wtS.T, in_dual)
    out_dual = np.dot(A_wtS, in_prime)
    out_vec = np.concatenate( (out_prime, out_dual), axis=0)
    return out_vec


K = LinearOperator((num_prime+num_dual, num_prime+num_dual), matvec=mat_vec_kkt  )



# ----------------- Defining Preconditioner as LinearOperator and Solve ----------
# With Slack Version!! 
def mat_vec_M(in_vec):
    in_design = in_vec[:num_design] 
    in_slack  = in_vec[num_design:num_prime] 
    in_dual   = in_vec[num_prime:]

    out_design = np.zeros_like(in_design)
    out_slack  = np.zeros_like(in_slack)
    out_dual   = np.zeros_like(in_dual)

    # step 2
    out_dual =  in_slack * sigma_inv

    # step 3, Note   B = I here, approximated

    out_design = in_design - A_noS.T.dot(out_dual) 

    # sub_b = in_design - A_noS.T.dot(out_dual) 
    # sub_res_hist = []
    # (out_design,flag) = fgmres(W_noS, sub_b, maxiter=20, tol=1e-5, residuals=sub_res_hist)


    # step 4 
    out_slack = ( in_dual - A_noS.dot(out_design) ) * sigma_inv

    out_vec = np.concatenate( (out_design, out_slack, out_dual), axis=0)
    return out_vec

M_pc = LinearOperator((num_prime+num_dual, num_prime+num_dual), matvec=mat_vec_M  )


#------------------------ Actually solving using the preconditioner --------------


res_hist = []
(x,flag) = fgmres(K, b, maxiter=100, tol=1e-6, residuals=res_hist)   #M=M_pc,


res_hist_pc = []
(x,flag) = fgmres(K, b, M=M_pc, maxiter=100, tol=1e-6, residuals=res_hist_pc)


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

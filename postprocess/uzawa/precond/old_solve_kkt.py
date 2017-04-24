from pyamg.krylov import fgmres
from scipy.sparse.linalg import LinearOperator
import pickle
import pdb 
import numpy as np


noS = False

if noS:
    appdix = 'noS'
    folder = 'EQU_DATA'
else:
    appdix = 'wtS'
    folder = 'INEQU_DATA'


f_cnstr_jac = open(folder+'/cnstr_jac_'+appdix,'rb')
A = pickle.load(f_cnstr_jac)
f_cnstr_jac.close()

f_hessian   = open(folder+'/hessian_'+appdix,'rb')
W = pickle.load(f_hessian)
f_hessian.close()

if noS:
    f_rhs1 = open(folder+'/rhs_dldx_'+appdix,'rb')
    dldx = pickle.load(f_rhs1)
    f_rhs1.close()

    f_con = open(folder+'/rhs_cnstr_'+appdix,'rb')
    con = pickle.load(f_con)
    f_con.close()
    
    b = np.concatenate( (dldx, con), axis=0)

else: 
    f_rhs = open(folder+'/kkt_rhs_'+appdix,'rb')
    b = pickle.load(f_rhs)
    f_rhs.close()

    f_sl = open(folder+'/slack','rb')
    slack_list = pickle.load(f_sl)
    f_sl.close()

    slack = np.concatenate( (slack_list[0], slack_list[1], slack_list[2]), axis=0 )

    # sigma = -diag(e^s)    sigma_inv = -diag(1/e^s)
    sigma_inv = np.zeros_like(slack)
    sigma_inv = -1. / np.exp(slack)

# -------------------- dimensions -------------------
num_design = 128
num_stress = 128
num_bound = 128*2 

num_dual= num_bound + num_stress

if noS:                         # pure KKT, no slack variables
    num_prime = num_design
else:                           # with Slack variables 
    num_prime = num_design + num_dual


# -----------------  LinearOperator and Solve -------------
def mat_vec_kkt(in_vec):
    in_prime = in_vec[:num_prime] 
    in_dual  = in_vec[num_prime:] 

    out_prime = np.zeros_like(in_prime)
    out_dual  = np.zeros_like(in_dual)

    out_prime = np.dot(W, in_prime) + np.dot(A.T, in_dual)
    out_dual = np.dot(A, in_prime)
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
    # out_design = in_design - A.T.dot(out_dual) 
    pdb.set_trace()
    
    sub_b = in_design - A.T.dot(out_dual) 
    (out_design,flag) = fgmres(W, sub_b, maxiter=50, tol=1e-6, residuals=res_hist)


    # step 4 
    out_slack = ( in_dual - A.dot(out_design) ) * sigma_inv

    out_vec = np.concatenate( (out_design, out_slack, out_dual), axis=0)
    return out_vec

M_pc = LinearOperator((num_prime+num_dual, num_prime+num_dual), matvec=mat_vec_M  )


#------------------------ Actually solving using the preconditioner --------------

res_hist = []
(x,flag) = fgmres(K, b, M=M_pc, maxiter=50, tol=1e-6, residuals=res_hist)

print res_hist
print len(res_hist)















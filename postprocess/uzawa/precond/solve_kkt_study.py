import pickle
import numpy as np 

f1 = open('DATA/hessian_noS','rb')
W = pickle.load(f1)
f1.close()

f2 = open('DATA/hessian_wtS','rb')
Ws = pickle.load(f2)
f2.close()

condW = np.linalg.cond(W)
condWs = np.linalg.cond(Ws[:128,:128])
print 'Condition number of Hessian', condW
print 'Condition number of Hessian', condWs

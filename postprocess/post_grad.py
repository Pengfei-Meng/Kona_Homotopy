# plotting the norm of constraint Jacobians
# ['vol', 'Euler_CRM_cl', 'thick', 'Euler_CRM_cd', 'Euler_CRM_cmy']
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import pdb
import pprint, pickle
import numpy as np

niter = 3
dirname = './scale'

clr = ['b','r','c','k','g']
mark = ['^', 'o', 'v', 'd', 'x']

label_thk = 'd(Thick)/dX'
label_vol = 'd(Vol)/dX'
label_cl  = 'd(Cl)/dX'
label_cm  = 'd(Cm)/dX'
label_cd  = 'd(Cd)/dX'

norm_thk = np.zeros(niter)
norm_vol = np.zeros(niter)
norm_cl = np.zeros(niter)
norm_cm = np.zeros(niter)
norm_cd = np.zeros(niter)

for k in range(0,niter):
    filename = dirname + '/sens_' + str(k) + '.pkl'
    pk = open(filename, 'rb')
    dat = pickle.load(pk)

    thk = np.zeros(dat['thick']['shape'].shape[0])
    for ic in range(0, dat['thick']['shape'].shape[0]):
        thk[ic] = np.linalg.norm( dat['thick']['shape'][ic,:])

    pdb.set_trace()    

    norm_thk[k] = np.linalg.norm(dat['thick']['shape'])
    norm_vol[k] = np.linalg.norm(dat['vol']['shape'])
    norm_cl[k]  = np.linalg.norm(dat['Euler_CRM_cl']['shape'])
    norm_cm[k]  = np.linalg.norm(dat['Euler_CRM_cmy']['shape'])
    norm_cd[k]  = np.linalg.norm(dat['Euler_CRM_cd']['shape'])

    # pdb.set_trace()

kitr = range(0, niter)

plt.subplot(111)
l_thk, = plt.semilogy(kitr, norm_thk, marker=mark[0], linestyle='-', color=clr[0], label=label_thk)
l_vol, = plt.semilogy(kitr, norm_vol, marker=mark[1], linestyle='-', color=clr[1], label=label_vol)
l_cl,  = plt.semilogy(kitr, norm_cl, marker=mark[2], linestyle='-', color=clr[2], label=label_cl)
l_cm,  = plt.semilogy(kitr, norm_cm, marker=mark[3], linestyle='-', color=clr[3], label=label_cm)
l_cd,  = plt.semilogy(kitr, norm_cd, marker=mark[4], linestyle='-', color=clr[4], label=label_cd)

plt.legend(loc='upper right', prop={'size':8})
plt.xlabel('Iteration')
    
plt.show()


import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import os, sys
import pdb



muinit=2
clr=['b','r','c','k','g']


k=0
for mupow in range(2,6):
    print 'mupow: ', mupow

    dirname = './NOTHICK/L2_init00' + str(muinit) + '_pow0' + str(mupow)
    filename = dirname + '/kona_hist.dat'
    cutline = 'sed -ie "s/Wall time.*//g" %s'%filename
    os.system(cutline)

    data = np.loadtxt(filename)

    kona_iter = data[:,0]
    kona_cost = data[:,1]
    kona_optimality = data[:,2]
    kona_feasibility = data[:,3]
    kona_objective = data[:,4]
    kona_mu = data[:,5]

    data2 = np.loadtxt(dirname + '/kona_timings.dat')
    kona_time = data2[:,2]   #  [:-1]

    if len(kona_time)!=len(kona_iter):
        kona_time = kona_time[:-1]

    label_opt = 'mu init 0.0'+str(muinit) + ' mu pow 0.' +str(mupow) + ' opt'
    label_feas = 'mu init 0.0'+str(muinit) + ' mu pow 0.' +str(mupow) + ' feas'
    label_obj = 'mu init 0.0'+str(muinit) + ' mu pow 0.' +str(mupow) + ' obj'
    label_mu = 'mu init 0.0'+str(muinit) + ' mu pow 0.' +str(mupow) + ' mu'

    plt.subplot(131)
    line1, = plt.semilogy(kona_time, kona_optimality, marker='^', linestyle='-', color=clr[k], label=label_opt)
    line2, = plt.semilogy(kona_time, kona_feasibility, marker='o', linestyle='-', color=clr[k], label=label_feas)
    plt.legend(loc='upper right', prop={'size':6})
    plt.xlabel('cpu time')

    plt.subplot(132)
    line3, = plt.plot(kona_time, kona_objective, marker='v', linestyle='-', color=clr[k], label=label_obj)
    plt.xlabel('cpu time')
    plt.ylabel('objective')
    plt.legend(loc='upper right', prop={'size':8})

    plt.subplot(133)
    line4, = plt.semilogy(kona_time, kona_mu, marker='v', linestyle='-', color=clr[k], label=label_mu)
    plt.xlabel('cpu time')
    plt.ylabel('mu')
    plt.legend(loc='lower right', prop={'size':8})

    k+=1

plt.show()

# muinit=2
# for mupow in range(2,4):
#     filename = './L2_init00' + str(muinit) + '_pow0' + str(mupow) + '/kona_hist.dat'
#     cutline = 'sed -ie "s/Wall time.*//g" %s'%filename
#     os.system(cutline)



# os.system('sed -i   "s/Wall time.*//g" %s'%filename)
# print filename
# print type(filename)
# # data = np.loadtxt(filename)
# data2 = np.loadtxt('./L2_init001_pow02/kona_hist.dat')
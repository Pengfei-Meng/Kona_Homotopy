import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import pdb

num_design = 600

dir_konahist = '../kona_latest/src/kona/test/output/' + str(num_design) + '_posT_cor0.1/'
fname = dir_konahist + 'kona_hist.dat'
dtype_cols = np.dtype([('outer_iter', 'i4'),('inner_iter', 'i4'), ('objective', 'float64'), ('optimality', 'float'), ('feasibility', 'float64')])
kona_datas = np.loadtxt(fname, dtype=dtype_cols, skiprows = 3, usecols = (0,1,3,5,6))

tname = dir_konahist + 'kona_timings.dat'
dtype_cols2 = np.dtype([('outer_iter', 'i4'), ('time', 'float64')])
kona_timings = np.loadtxt(tname, dtype=dtype_cols2, skiprows = 3, usecols = (0,2))

# when inner_iter > 1 before mu = 0, display only one inner_iter, to make plots clean
iter_unique, indices = np.unique(kona_datas['outer_iter'], return_index=True)
last_indices = indices[-1]
last_inners = range(last_indices+1, len(kona_datas['outer_iter']))
new_indices = np.hstack([indices, np.array(last_inners)])

kona_time = kona_timings['time'][new_indices]
kona_data = kona_datas[new_indices]

# ----- Processing "SNOPT_summary.out" file 

# run ./README under '~/Developer/kona_mengp2/kona_latest/src/kona/test/output' in Linux system

snopt_name = dir_konahist + 'SNOPT_summary.awk'
dtype_sn = np.dtype([('outer_iter', 'i4'),('nCon', 'i4'), ('feasibility', 'float64'),('optimality', 'float'), ('merit', 'float')])
snopt_data = np.loadtxt(snopt_name, dtype=dtype_sn, skiprows = 1, usecols = (0,3,4,5,6))
nCon_idx = snopt_data['nCon']

tname = dir_konahist + 'SNOPT_timings.dat'
dtype_cols2 = np.dtype([('outer_qqiter', 'i4'), ('time', 'float64')])
snopt_time_s = np.loadtxt(tname, dtype=dtype_cols2, skiprows = 2, usecols = (0,2))

snopt_time = snopt_time_s['time'][nCon_idx-1]

plt.subplot(121)
line1, = plt.semilogy(kona_time, kona_data['optimality'], marker='^', linestyle='-', color='b', label='kona_optimality') 
line2, = plt.semilogy(kona_time, kona_data['feasibility'], marker='o', linestyle='-', color='r', label='kona_feasibility')  
plt.legend([line1, line2], ['kona_optimality', 'kona_feasibility'], prop={'size':12})
plt.xlabel('cpu time seconds', fontsize=18)


plt.subplot(122)
line3, = plt.semilogy(snopt_time, snopt_data['optimality'], marker='^', linestyle='-.', color='b', label='snopt_optimality')    
line4, = plt.semilogy(snopt_time, snopt_data['feasibility'], marker='o', linestyle='-.', color='r', label='snopt_feasibility')

plt.xlabel('cpu time seconds', fontsize=18)
plt.legend([line3, line4], ['snopt_optimality', 'snopt_feasibility'], prop={'size':12})


# plt.suptitle('Num design = 100, diffX = 2.71e-06')
# plt.suptitle('Num design = 200, diffX = 6.41e-07')
# plt.suptitle('Num design = 300, diffX = 0.023')
# plt.suptitle('Num design = 400, diffX = 0.0098')
# plt.suptitle('Num design = 500, diffX = 0.0028')
plt.suptitle('Num design = 600, diffX = 0.0044')
plt.show()     #  max( abs(konaX - pyoptX)/norm(pyoptX) )

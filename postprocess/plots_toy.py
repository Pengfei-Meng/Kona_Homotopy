import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import pdb

num_design = 300
diffX_eye = 0.074
diffX_svd = 0.025
diffX_adj = 0.035

#-------------- Identity PCD ------------
# 100
# diffX_eye = 0.030
# diffX_svd = 3.2e-6
# diffX_adj = 2.2e-6

# num_design = 200
# diffX_eye = 0.046
# diffX_svd = 8.0e-7
# diffX_adj = 2.6e-6



pcd = 'eye'

dir_konahist = '../kona_latest/src/kona/test/output/' + str(num_design) + '_' + pcd + '/'
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

plt.subplot(221)
line1, = plt.semilogy(kona_time, kona_data['optimality']/kona_data['optimality'][0], marker='^', linestyle='-', color='b', label='kona_optimality') 
line2, = plt.semilogy(kona_time, kona_data['feasibility']/kona_data['feasibility'][0], marker='o', linestyle='-', color='r', label='kona_feasibility')  
plt.legend([line1, line2], ['kona_optimality', 'kona_feasibility'], prop={'size':12})
#plt.xlabel('cpu time seconds', fontsize=12)
plt.tick_params(labelsize=12)
plt.title('pc = Identity  diffX = %.2e'%diffX_eye)

#--------------  SVD PCD --------------
pcd = 'svd'

dir_konahist = '../kona_latest/src/kona/test/output/' + str(num_design) + '_' + pcd + '/'
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

plt.subplot(222)
line1, = plt.semilogy(kona_time, kona_data['optimality']/kona_data['optimality'][0], marker='^', linestyle='-', color='b', label='kona_optimality') 
line2, = plt.semilogy(kona_time, kona_data['feasibility']/kona_data['feasibility'][0], marker='o', linestyle='-', color='r', label='kona_feasibility')  
plt.legend([line1, line2], ['kona_optimality', 'kona_feasibility'], prop={'size':12})
#plt.xlabel('cpu time seconds', fontsize=12)
plt.tick_params(labelsize=12)
plt.title('pc = SVD  diffX = %.2e'%diffX_svd)

#---------------- ADJ PCD ---------------
pcd = 'adj'

dir_konahist = '../kona_latest/src/kona/test/output/' + str(num_design) + '_' + pcd + '/'
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

plt.subplot(223)
line1, = plt.semilogy(kona_time, kona_data['optimality']/kona_data['optimality'][0], marker='^', linestyle='-', color='b', label='kona_optimality') 
line2, = plt.semilogy(kona_time, kona_data['feasibility']/kona_data['feasibility'][0], marker='o', linestyle='-', color='r', label='kona_feasibility')  
plt.legend([line1, line2], ['kona_optimality', 'kona_feasibility'], prop={'size':12})
plt.xlabel('cpu time seconds', fontsize=12)
plt.tick_params(labelsize=12)
plt.title('pc = Approximate Adjoint  diffX = %.2e'%diffX_adj)


# ----- Processing "SNOPT_summary.out" file 

# run ./README under '~/Developer/kona_mengp2/kona_latest/src/kona/test/output' in Linux system
dir_konahist = '../kona_latest/src/kona/test/output/' + str(num_design) + '_eye/'
snopt_name = dir_konahist + 'SNOPT_summary.awk'
dtype_sn = np.dtype([('outer_iter', 'i4'),('nCon', 'i4'), ('feasibility', 'float64'),('optimality', 'float'), ('merit', 'float')])
snopt_data = np.loadtxt(snopt_name, dtype=dtype_sn, skiprows = 1, usecols = (0,3,4,5,6))
nCon_idx = snopt_data['nCon']

tname = dir_konahist + 'SNOPT_timings.dat'
dtype_cols2 = np.dtype([('outer_qqiter', 'i4'), ('time', 'float64')])
snopt_time_s = np.loadtxt(tname, dtype=dtype_cols2, skiprows = 2, usecols = (0,2))

snopt_time = snopt_time_s['time'][nCon_idx-1]

plt.subplot(224)
line3, = plt.semilogy(snopt_time, snopt_data['optimality'], marker='^', linestyle='-.', color='b', label='snopt_optimality')    
line4, = plt.semilogy(snopt_time, snopt_data['feasibility'], marker='o', linestyle='-.', color='r', label='snopt_feasibility')

plt.xlabel('cpu time seconds', fontsize=12)
plt.legend([line3, line4], ['snopt_optimality', 'snopt_feasibility'], prop={'size':12})
plt.tick_params(labelsize=12)
plt.title('SNOPT')

plt.suptitle('No. Design = %d'%num_design)
# plt.suptitle('Num design = 200, diffX = 6.41e-07')
# plt.suptitle('Num design = 300, diffX = 0.023')

#plt.show()     #  max( abs(konaX - pyoptX)/norm(pyoptX) )

fig = plt.gcf()
fig.set_size_inches(8.2, 9.2)
fig_name = './figures/' + str(num_design) + '.eps'
fig.savefig(fig_name, format='eps', dpi=1200)
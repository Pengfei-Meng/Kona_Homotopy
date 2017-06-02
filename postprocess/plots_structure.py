import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import pdb


# # Plotting SVD results
case = 'tiny'
dir_konahist = '../results3/' + case + '/'
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

kona_time = kona_timings['time'] 
kona_data = kona_datas[new_indices]

# --------------- Add the Identity PC for comparison ----------------
case2 = case + '_eye'
dir_konahist2 = '../results3/' + case2 + '/'
fname2 = dir_konahist2 + 'kona_hist.dat'
# dtype_colsX2 = np.dtype([('outer_iter', 'i4'),('inner_iter', 'i4'), ('objective', 'float64'), ('optimality', 'float'), ('feasibility', 'float64')])
kona_datas2 = np.loadtxt(fname2, dtype=dtype_cols, skiprows = 3, usecols = (0,1,3,5,6))

tname2 = dir_konahist2 + 'kona_timings.dat'
# dtype_cols2 = np.dtype([('outer_iter', 'i4'), ('time', 'float64')])
kona_timings2 = np.loadtxt(tname2, dtype=dtype_cols2, skiprows = 3, usecols = (0,2))


# when inner_iter > 1 before mu = 0, display only one inner_iter, to make plots clean
iter_unique2, indices2 = np.unique(kona_datas2['outer_iter'], return_index=True)
last_indices2 = indices2[-1]
last_inners2 = range(last_indices2+1, len(kona_datas2['outer_iter']))
new_indices2 = np.hstack([indices2, np.array(last_inners2)])

kona_time_eye = kona_timings2['time'][:-1] 
kona_data_eye = kona_datas2[new_indices2]



# # --------------------------------------------------------------------
###### result files stored in  ~/Developer/kona_mengp2/results3/SNOPT/

# case = 'medium'
dir_konahist = '../results3/SNOPT/' + case + '/'
snopt_name = dir_konahist + 'SNOPT_summary_processed'
dtype_sn = np.dtype([('outer_iter', 'i4'),('nCon', 'i4'), ('feasibility', 'float64'),('optimality', 'float'), ('merit', 'float')])
snopt_data = np.loadtxt(snopt_name, dtype=dtype_sn, skiprows = 1, usecols = (0,3,4,5,6))
nCon_idx = snopt_data['nCon']

tname = dir_konahist + 'SNOPT_timings.dat'
dtype_cols2 = np.dtype([('outer_qqiter', 'i4'), ('time', 'float64')])
snopt_time_s = np.loadtxt(tname, dtype=dtype_cols2, skiprows = 2, usecols = (0,2))

snopt_time = snopt_time_s['time'][nCon_idx-1]

# plt.subplot(111)
# line3, = plt.semilogy(snopt_time, snopt_data['optimality'], marker='^', linestyle='-.', color='b', label='snopt_optimality')    
# line4, = plt.semilogy(snopt_time, snopt_data['feasibility'], marker='o', linestyle='-.', color='b', label='snopt_feasibility')

# plt.xlabel('cpu time seconds', fontsize=12)
# plt.legend([line3, line4], ['snopt_optimality', 'snopt_feasibility'], prop={'size':12})
# plt.tick_params(labelsize=12)
# plt.title('SNOPT ' + case)

# plt.show()   

plt.subplot(111)

line1, = plt.semilogy(kona_time_eye, kona_data_eye['optimality']/kona_data_eye['optimality'][0], marker='^', linestyle='-', color='g', label='eye_optimality') 
line2, = plt.semilogy(kona_time_eye, kona_data_eye['feasibility']/kona_data_eye['feasibility'][0], marker='o', linestyle='-', color='g', label='eye_feasibility')  

line3, = plt.semilogy(kona_time, kona_data['optimality']/kona_data['optimality'][0], marker='^', linestyle='-', color='r', label='kona_optimality') 
line4, = plt.semilogy(kona_time, kona_data['feasibility']/kona_data['feasibility'][0], marker='o', linestyle='-', color='r', label='kona_feasibility')  

line5, = plt.semilogy(snopt_time, snopt_data['optimality'], marker='^', linestyle='-.', color='b', label='snopt_optimality')    
line6, = plt.semilogy(snopt_time, snopt_data['feasibility'], marker='o', linestyle='-.', color='b', label='snopt_feasibility')

# --------------------------------------------------------------------

plt.legend([line1, line2, line3, line4, line5, line6], ['Eye_optimality', 'Eye_feasibility', 'SVD_optimality', 'SVD_feasibility',  'SNOPT_optimality', 'SNOPT_feasibility'], prop={'size':10})
plt.xlabel('cpu time seconds', fontsize=12)
plt.tick_params(labelsize=12)
plt.title(case)


plt.show()     
# fig = plt.gcf()
# fig.set_size_inches(4.6, 4.6)
# fig_name = './figures/' + case + '.eps'
# fig.savefig(fig_name, format='eps', dpi=1200)   
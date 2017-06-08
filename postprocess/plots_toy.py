import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import pdb

num_design = 500

#-------------- Identity PCD ------------


pcd = 'eye'

dir_konahist = '../kona_latest/src/kona/test/output3/' + str(num_design) + '_' + pcd + '/'
fname = dir_konahist + 'kona_hist.dat'

dtype_cols = np.dtype([('outer_iter', 'i4'),('inner_iter', 'i4'), ('objective', 'float64'), ('optimality', 'float'), ('feasibility', 'float64')])
kona_datas = np.loadtxt(fname, dtype=dtype_cols, skiprows = 2, usecols = (0,1,3,5,6))

tname = dir_konahist + 'kona_timings.dat'
dtype_cols2 = np.dtype([('outer_iter', 'i4'), ('time', 'float64')])
kona_timings = np.loadtxt(tname, dtype=dtype_cols2, skiprows = 2, usecols = (0,2))

# when inner_iter > 1 before mu = 0, display only one inner_iter, to make plots clean
iter_unique, indices = np.unique(kona_datas['outer_iter'], return_index=True)
last_indices = indices[-1]
last_inners = range(last_indices+1, len(kona_datas['outer_iter']))
new_indices = np.hstack([indices, np.array(last_inners)])

kona_time_eye = kona_timings['time'][:-1]
kona_data_eye = kona_datas[new_indices]


# ------------------------ SVD convergence results --------------------------
#--------------  SVD PCD --------------
pcd = 'svd'

dir_konahist = '../kona_latest/src/kona/test/output3/' + str(num_design) + '_' + pcd + '/'
fname = dir_konahist + 'kona_hist.dat'

dtype_cols = np.dtype([('outer_iter', 'i4'),('inner_iter', 'i4'), ('objective', 'float64'), ('optimality', 'float'), ('feasibility', 'float64')])
kona_datas = np.loadtxt(fname, dtype=dtype_cols, skiprows = 2, usecols = (0,1,3,5,6))

tname = dir_konahist + 'kona_timings.dat'
dtype_cols2 = np.dtype([('outer_iter', 'i4'), ('time', 'float64')])
kona_timings = np.loadtxt(tname, dtype=dtype_cols2, skiprows = 2, usecols = (0,2))


# when inner_iter > 1 before mu = 0, display only one inner_iter, to make plots clean
iter_unique, indices = np.unique(kona_datas['outer_iter'], return_index=True)
last_indices = indices[-1]
last_inners = range(last_indices+1, len(kona_datas['outer_iter']))

if len(last_inners) > 0: 
    new_indices = np.hstack([indices, np.array(last_inners)])
else:
    new_indices = indices


kona_time_svd = kona_timings['time']
kona_data_svd = kona_datas[new_indices]



# -------- Processing "SNOPT_summary.out" file -------------

# run ./README under '~/Developer/kona_mengp2/kona_latest/src/kona/test/output' in Linux system
dir_konahist = '../kona_latest/src/kona/test/output3/' + str(num_design) + '_svd/'
snopt_name = dir_konahist + 'SNOPT_summary.awk'
dtype_sn = np.dtype([('outer_iter', 'i4'),('nCon', 'i4'), ('feasibility', 'float64'),('optimality', 'float'), ('merit', 'float')])
snopt_data = np.loadtxt(snopt_name, dtype=dtype_sn, skiprows = 1, usecols = (0,3,4,5,6))
nCon_idx = snopt_data['nCon']

tname = dir_konahist + 'SNOPT_timings.dat'
dtype_cols2 = np.dtype([('outer_qqiter', 'i4'), ('time', 'float64')])
snopt_time_s = np.loadtxt(tname, dtype=dtype_cols2, skiprows = 2, usecols = (0,2))

snopt_time = snopt_time_s['time'][nCon_idx-1]


# -------------------- plotting ---------------------
# plot the data
# ms = markersize
# mfc = markerfacecolor     mec = 'k'
# mew = markeredgewidth
axis_fs = 12 # axis title font size
axis_lw = 1.0 # line width used for axis box, legend, and major ticks
label_fs = 10 # axis labels' font size


fig = plt.figure(figsize=(6,4), facecolor=None)
ax = fig.add_subplot(111)

line1, = ax.semilogy(kona_time_eye, kona_data_eye['optimality']/kona_data_eye['optimality'][0], '-k^', linewidth=1.0, ms=6.0, mfc='w', mew=1.0) 
line2, = ax.semilogy(kona_time_eye, kona_data_eye['feasibility']/kona_data_eye['feasibility'][0], ':k^', linewidth=1.0, ms=6.0, mfc='w', mew=1.0)  

line3, = ax.semilogy(kona_time_svd, kona_data_svd['optimality']/kona_data_svd['optimality'][0], '-ro', linewidth=1.0, ms=6.0, mfc='w', mew=1.0) 
line4, = ax.semilogy(kona_time_svd, kona_data_svd['feasibility']/kona_data_svd['feasibility'][0], ':ro', linewidth=1.0, ms=6.0, mfc='w', mew=1.0)  

line5, = ax.semilogy(snopt_time, snopt_data['optimality']/snopt_data['optimality'][0], '-bs', linewidth=1.0, ms=6.0, mfc='w', mew=1.0)    
line6, = ax.semilogy(snopt_time, snopt_data['feasibility']/snopt_data['feasibility'][0], ':bs', linewidth=1.0, ms=6.0, mfc='w', mew=1.0)


ax.set_position([0.15, 0.13, 0.80, 0.83])                                # position relative to figure edges
ax.set_xlabel('CPU time seconds', fontsize=axis_fs, weight='bold')
ax.set_ylabel('Relative Optimality/Feasibility', fontsize=axis_fs, weight='bold')
ax.grid(which='major', axis='y', linestyle='--')
ax.set_axisbelow(True) # grid lines are plotted below
plt.tick_params(labelsize=axis_fs)
rect = ax.patch # a Rectangle instance
#rect.set_facecolor('white')
#rect.set_ls('dashed')
rect.set_linewidth(axis_lw)
rect.set_edgecolor('k')


# ticks on bottom and left only
ax.xaxis.tick_bottom() # use ticks on bottom only
ax.yaxis.tick_left()
for line in ax.xaxis.get_ticklines():
    line.set_markersize(6) # length of the tick
    line.set_markeredgewidth(axis_lw) # thickness of the tick
for line in ax.yaxis.get_ticklines():
    line.set_markersize(6) # length of the tick
    line.set_markeredgewidth(axis_lw) # thickness of the tick
for label in ax.xaxis.get_ticklabels():
    label.set_fontsize(label_fs)
for label in ax.yaxis.get_ticklabels():
    label.set_fontsize(label_fs)

# define and format the minor ticks
# ax.xaxis.set_ticks(np.arange(0, 3.0, 0.5),minor=True)
# ax.xaxis.set_tick_params(which='minor', length=3, width=2.0*axis_lw/3.0)

xmax = max(kona_time_eye[-1], kona_time_svd[-1], snopt_time[-1])


ax.yaxis.set_ticks(np.logspace(-16, 2, num=10))
ax.yaxis.set_tick_params(which='minor', length=3, width=2.0*axis_lw/3.0)
textstr = 'Number of Design : %i'%num_design 
ax.text(xmax*0.6, 10, textstr, fontsize=label_fs, weight='bold')


leg = ax.legend([line1, line2, line3, line4, line5, line6], ['Eye_optimality', 'Eye_feasibility', 'SVD_optimality', 'SVD_feasibility',  'SNOPT_optimality', 'SNOPT_feasibility'], \
                loc=(0.5, 0.15), numpoints=1, prop={'size':6},  borderpad=0.75, handlelength=4)
rect = leg.get_frame()
rect.set_linewidth(axis_lw)
for t in leg.get_texts():
    t.set_fontsize(10)    # the legend text fontsize

plt.show()     

# fig = plt.gcf()
# # fig.set_size_inches(8.2, 9.2)
# fig_name = './figures2/' + str(num_design) + '.eps'
# fig.savefig(fig_name, format='eps', dpi=1200)
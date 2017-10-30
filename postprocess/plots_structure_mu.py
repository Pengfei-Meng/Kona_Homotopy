import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import pdb


# # Plotting SVD results
case = 'small'

# num_design, nlcost = 128, 9
num_design, nlcost = 512, 10
# num_design, nlcost = 2048, 12

# xax = 'time'
xax = 'cost'

pcd = 'svd'
dir_konahist = '../results4/' + case + '_' + pcd + '/'
fname = dir_konahist + 'kona_hist.dat'
dtype_cols = np.dtype([('outer_iter', 'i4'),('inner_iter', 'i4'), ('cost', 'i4'), ('objective', 'float64'), ('optimality', 'float'), ('feasibility', 'float64')])
kona_datas = np.loadtxt(fname, dtype=dtype_cols, skiprows = 3, usecols = (0,1,2, 3,5,6))

tname = dir_konahist + 'kona_timings.dat'
dtype_cols2 = np.dtype([('outer_iter', 'i4'), ('time', 'float64')])
kona_timings = np.loadtxt(tname, dtype=dtype_cols2, skiprows = 2, usecols = (0,2))

# when inner_iter > 1 before mu = 0, display only one inner_iter, to make plots clean
iter_unique, indices = np.unique(kona_datas['outer_iter'], return_index=True)
last_indices = indices[-1]
last_inners = range(last_indices+1, len(kona_datas['outer_iter']))
new_indices = np.hstack([indices, np.array(last_inners)])

kona_time_svd = kona_timings['time'][:-1] 
kona_data_svd = kona_datas[new_indices]

# --------------- Add the Identity PC for comparison ----------------
pcd = 'eye_long'
dir_konahist2 = '../results4/' + case + '_' + pcd + '/'
fname2 = dir_konahist2 + 'kona_hist.dat'
kona_datas2 = np.loadtxt(fname2, dtype=dtype_cols, skiprows = 2, usecols = (0,1,2, 3,5,6))

tname2 = dir_konahist2 + 'kona_timings.dat'
kona_timings2 = np.loadtxt(tname2, dtype=dtype_cols2, skiprows = 2, usecols = (0,2))

# when inner_iter > 1 before mu = 0, display only one inner_iter, to make plots clean
iter_unique2, indices2 = np.unique(kona_datas2['outer_iter'], return_index=True)
last_indices2 = indices2[-1]
last_inners2 = range(last_indices2+1, len(kona_datas2['outer_iter']))
new_indices2 = np.hstack([indices2, np.array(last_inners2)])

kona_time_eye = kona_timings2['time'][:-1] 
kona_data_eye = kona_datas2[new_indices2]

# ------------- SNOPT data -----------------
# --------------------------------------------------------------------
##### result files stored in  ~/Developer/kona_mengp2/results3/SNOPT/
pcd = 'snopt'
dir_konahist = '../results4/' + case + '_' + pcd + '/'
snopt_name = dir_konahist + 'SNOPT_summary.awk'
dtype_sn = np.dtype([('outer_iter', 'i4'),('nCon', 'i4'), ('feasibility', 'float64'),('optimality', 'float'), ('merit', 'float')])
snopt_data = np.loadtxt(snopt_name, dtype=dtype_sn, skiprows = 1, usecols = (0,3,4,5,6))
nCon_idx = snopt_data['nCon']

tname = dir_konahist + 'SNOPT_timings.dat'
dtype_cols2 = np.dtype([('ncon', 'i4'), ('cost', 'i4'), ('time', 'float64')])
snopt_time_s = np.loadtxt(tname, dtype=dtype_cols2, skiprows = 0, usecols = (0,1,3))

snopt_time = snopt_time_s['time'][nCon_idx-1]
snopt_cost = snopt_time_s['cost'][nCon_idx-1]


# -------------------- Plotting --------------------------
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


if xax is 'time':
    line1, = ax.semilogy(kona_time_eye, kona_data_eye['optimality']/kona_data_eye['optimality'][0], '-k^', linewidth=1.0, ms=6.0, mfc='w', mew=1.0) 
    line2, = ax.semilogy(kona_time_eye, kona_data_eye['feasibility'], ':k^', linewidth=1.0, ms=6.0, mfc='w', mew=1.0)  

    line3, = ax.semilogy(kona_time_svd, kona_data_svd['optimality']/kona_data_svd['optimality'][0], '-ro', linewidth=1.0, ms=6.0, mfc='w', mew=1.0) 
    line4, = ax.semilogy(kona_time_svd, kona_data_svd['feasibility'], ':ro', linewidth=1.0, ms=6.0, mfc='w', mew=1.0)  

    line5, = ax.semilogy(snopt_time, snopt_data['optimality']/snopt_data['optimality'][0], '-bs', linewidth=1.0, ms=6.0, mfc='w', mew=1.0)    
    line6, = ax.semilogy(snopt_time, snopt_data['feasibility'], ':bs', linewidth=1.0, ms=6.0, mfc='w', mew=1.0)

    ax.set_position([0.15, 0.13, 0.80, 0.83])                                # position relative to figure edges
    ax.set_xlabel('CPU time seconds', fontsize=axis_fs, weight='bold')

    xmax = max(kona_time_eye[-1], kona_time_svd[-1], snopt_time[-1])

if xax is 'cost':
    line1, = ax.semilogy(kona_data_eye['cost']/nlcost, kona_data_eye['optimality']/kona_data_eye['optimality'][0], '-k^', linewidth=1.0, ms=6.0, mfc='w', mew=1.0) 
    line2, = ax.semilogy(kona_data_eye['cost']/nlcost, kona_data_eye['feasibility'], ':k^', linewidth=1.0, ms=6.0, mfc='w', mew=1.0)  

    line3, = ax.semilogy(kona_data_svd['cost']/nlcost, kona_data_svd['optimality']/kona_data_svd['optimality'][0], '-ro', linewidth=1.0, ms=6.0, mfc='w', mew=1.0) 
    line4, = ax.semilogy(kona_data_svd['cost']/nlcost, kona_data_svd['feasibility'], ':ro', linewidth=1.0, ms=6.0, mfc='w', mew=1.0)  

    line5, = ax.semilogy(snopt_cost/nlcost, snopt_data['optimality']/snopt_data['optimality'][0], '-bs', linewidth=1.0, ms=6.0, mfc='w', mew=1.0)    
    line6, = ax.semilogy(snopt_cost/nlcost, snopt_data['feasibility'], ':bs', linewidth=1.0, ms=6.0, mfc='w', mew=1.0)

    ax.set_position([0.15, 0.13, 0.80, 0.83])                                # position relative to figure edges    mfc=(0.35, 0.35, 0.35)
    ax.set_xlabel('PDE Solves', fontsize=axis_fs, weight='bold')

    xmax = max(kona_data_eye['cost'][-1]/nlcost, kona_data_svd['cost'][-1]/nlcost, snopt_cost[-1]/nlcost)

ax.set_ylabel('Relative optimality/Feasibility', fontsize=axis_fs, weight='bold')
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


ax.yaxis.set_ticks(np.logspace(-8, 2, num=11))
ax.yaxis.set_tick_params(which='minor', length=3, width=2.0*axis_lw/3.0)
textstr = 'Number of Design : %i'%num_design 
ax.text(xmax*0.6, 15, textstr, fontsize=label_fs, weight='bold')

# if case is 'tiny':
leg_size = 2



leg = ax.legend([line1, line2, line3, line4, line5, line6], ['noPC_opt', 'noPC_feas', 'PC_opt', 'PC_feas',  'SNOPT_opt', 'SNOPT_feas'], \
                loc=(0.01, 0.01), numpoints=1, prop={'size':leg_size},  borderpad=0.75, handlelength=4)
rect = leg.get_frame()
rect.set_linewidth(axis_lw)
for t in leg.get_texts():
    t.set_fontsize(10)    # the legend text fontsize

plt.show()     

 

# plt.subplot(111)

# line1, = plt.semilogy(kona_time_eye, kona_data_eye['optimality']/kona_data_eye['optimality'][0], marker='^', linestyle='-', color='g', label='eye_optimality') 
# line2, = plt.semilogy(kona_time_eye, kona_data_eye['feasibility'], marker='o', linestyle='-', color='g', label='eye_feasibility')  

# line3, = plt.semilogy(kona_time, kona_data['optimality']/kona_data['optimality'][0], marker='^', linestyle='-', color='r', label='kona_optimality') 
# line4, = plt.semilogy(kona_time, kona_data['feasibility'], marker='o', linestyle='-', color='r', label='kona_feasibility')  

# line5, = plt.semilogy(snopt_time, snopt_data['optimality'], marker='^', linestyle='-.', color='b', label='snopt_optimality')    
# line6, = plt.semilogy(snopt_time, snopt_data['feasibility'], marker='o', linestyle='-.', color='b', label='snopt_feasibility')

# # --------------------------------------------------------------------

# plt.legend([line1, line2, line3, line4, line5, line6], ['Eye_optimality', 'Eye_feasibility', 'SVD_optimality', 'SVD_feasibility',  'SNOPT_optimality', 'SNOPT_feasibility'], prop={'size':10})
# plt.xlabel('cpu time seconds', fontsize=12)
# plt.tick_params(labelsize=12)
# plt.title(case)


# plt.show()     
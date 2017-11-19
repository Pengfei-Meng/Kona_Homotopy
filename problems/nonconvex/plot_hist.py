import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import pdb

num_design = 100

dir_kona = './random/'

fname = dir_kona + 'kona_hist.dat'

dtype_cols = np.dtype([('outer_iter', 'i4'),('inner_iter', 'i4'), ('objective', 'float64'), ('optimality', 'float'), ('feasibility', 'float64'), ('mu', 'float64')])
kona_datas = np.loadtxt(fname, dtype=dtype_cols, skiprows = 2, usecols = (0,1,3,5,6,10))

# when inner_iter > 1 before mu = 0, display only one inner_iter, to make plots clean
iter_unique, indices = np.unique(kona_datas['outer_iter'], return_index=True)
last_indices = indices[-1]
last_inners = range(last_indices+1, len(kona_datas['outer_iter']))
new_indices = np.hstack([indices, np.array(last_inners)])

kona_data_eye = kona_datas[new_indices]


# ---------------  kona_timing  -----------------

tname = dir_kona + 'kona_timings.dat'
dtype_cols2 = np.dtype([('outer_iter', 'i4'), ('time', 'float64')])
kona_timings = np.loadtxt(tname, dtype=dtype_cols2, skiprows = 2, usecols = (0,2))

kona_time_eye = kona_timings['time'] #[:-1]

pdb.set_trace()
# -------------------- plotting ---------------------
# plot the data
# ms = markersize
# mfc = markerfacecolor     mec = 'k'
# mew = markeredgewidth
axis_fs = 12 # axis title font size
axis_lw = 1.0 # line width used for axis box, legend, and major ticks
label_fs = 11 # axis labels' font size


fig = plt.figure(figsize=(7,4), facecolor=None)
ax = fig.add_subplot(111)

# line1, = ax.semilogy(kona_data_eye['outer_iter'], kona_data_eye['optimality']/kona_data_eye['optimality'][0], '-k^', linewidth=1.0, ms=6.0, mfc='w', mew=1.0) 
# line2, = ax.semilogy(kona_data_eye['outer_iter'], kona_data_eye['feasibility']/kona_data_eye['feasibility'][0], ':k^', linewidth=1.0, ms=6.0, mfc='w', mew=1.0)  

# line1, = ax.semilogy(kona_data_eye['mu'], kona_data_eye['optimality']/kona_data_eye['optimality'][0], '-k^', linewidth=1.0, ms=6.0, mfc='w', mew=1.0) 
# line2, = ax.semilogy(kona_data_eye['mu'], kona_data_eye['feasibility']/kona_data_eye['feasibility'][0], ':k^', linewidth=1.0, ms=6.0, mfc='w', mew=1.0)  


line1, = ax.semilogy(kona_time_eye, kona_data_eye['optimality']/kona_data_eye['optimality'][0], '-k^', linewidth=1.0, ms=6.0, mfc='w', mew=1.0) 
line2, = ax.semilogy(kona_time_eye, kona_data_eye['feasibility']/kona_data_eye['feasibility'][0], ':k^', linewidth=1.0, ms=6.0, mfc='w', mew=1.0)  


ax.set_position([0.15, 0.13, 0.80, 0.83])                                # position relative to figure edges
# ax.set_xlabel('Homotopy Iteration  $\mu$', fontsize=axis_fs, weight='bold')
# ax.invert_xaxis()

ax.set_xlabel('CPU time', fontsize=axis_fs, weight='bold')

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


# ----------- Set_Ticks -------------- # 


ax.yaxis.set_ticks(np.logspace(-14, 2, num=9))
ax.yaxis.set_tick_params(which='minor', length=3, width=2.0*axis_lw/3.0)

# xtick_range = np.arange(max(kona_data_eye['mu']), min(kona_data_eye['mu'])-0.05, -0.2)

# ax.xaxis.set_ticks( xtick_range, minor=True)
ax.xaxis.set_tick_params(which='minor', length=3, width=2.0*axis_lw/3.0)


leg = ax.legend([line1, line2], ['Opt', 'Feas'], \
                loc=(0.05, 0.05), numpoints=1, prop={'size':6},  borderpad=0.75, handlelength=4)
rect = leg.get_frame()
rect.set_linewidth(axis_lw)
for t in leg.get_texts():
    t.set_fontsize(12)    # the legend text fontsize


# ----------- Showing \mu value ---------------
# ax2 = ax.twiny()
# ax2.set_position(ax.get_position())      
# kona_mu = np.around(kona_data_eye['mu'], decimals=2)

# ax2.set_xlim(ax.get_xlim())
# ax2.set_xticks(ax.get_xticks())
# kona_mu2 = np.around( np.linspace(1.0, 0.0, 7), decimals=2  )
# ax2.set_xticklabels( kona_mu2  )
# ax2.set_xlabel("\mu", fontsize=axis_fs, weight='bold')

plt.show()     

# fig = plt.gcf()
# # fig.set_size_inches(8.2, 9.2)
# fig_name = './figures2/' + str(num_design) + '.eps'
# fig.savefig(fig_name, format='eps', dpi=1200)


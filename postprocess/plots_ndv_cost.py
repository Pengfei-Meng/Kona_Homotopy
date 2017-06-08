import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import pdb


num_designs = np.array([100, 200, 300, 400, 500])

num_case = len(num_designs)

cpu_times_kona = np.zeros(num_case)
cpu_times_snopt = np.zeros(num_case)
i = 0

for num_design in num_designs:
    pcd = 'svd'

    dir_konahist = '../kona_latest/src/kona/test/output3/' + str(num_design) + '_' + pcd + '/'

    tname = dir_konahist + 'kona_timings.dat'
    dtype_cols2 = np.dtype([('outer_iter', 'i4'), ('time', 'float64')])
    kona_timings = np.loadtxt(tname, dtype=dtype_cols2, skiprows = 2, usecols = (0,2))

    cpu_times_kona[i] = kona_timings['time'][-1]


    # -------- Processing "SNOPT_summary.out" file -------------

    # run ./README under '~/Developer/kona_mengp2/kona_latest/src/kona/test/output' in Linux system
    dir_konahist = '../kona_latest/src/kona/test/output3/' + str(num_design) + '_svd/'

    tname = dir_konahist + 'SNOPT_timings.dat'
    dtype_cols2 = np.dtype([('outer_qqiter', 'i4'), ('time', 'float64')])
    snopt_time_s = np.loadtxt(tname, dtype=dtype_cols2, skiprows = 2, usecols = (0,2))

    cpu_times_snopt[i] = snopt_time_s['time'][-1]

    i += 1

# ------------------ make plots ----------------

axis_fs = 12 # axis title font size
axis_lw = 1.0 # line width used for axis box, legend, and major ticks
label_fs = 10 # axis labels' font size


fig = plt.figure(figsize=(6,4), facecolor=None)
ax = fig.add_subplot(111)

line1, = ax.plot(num_designs, cpu_times_kona, '-ro', linewidth=1.0, ms=6.0, mfc='w', mew=1.0) 
line2, = ax.plot(num_designs, cpu_times_snopt, '-bs', linewidth=1.0, ms=6.0, mfc='w', mew=1.0) 

ax.set_position([0.15, 0.13, 0.80, 0.83])                                # position relative to figure edges
ax.set_xlabel('Number of design variables', fontsize=axis_fs, weight='bold')
ax.set_ylabel('Cost (CPU computing time)', fontsize=axis_fs, weight='bold')
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
ax.xaxis.set_ticks(np.arange(100, 550, 100),minor=False)
ax.xaxis.set_tick_params(which='minor', length=3, width=2.0*axis_lw/3.0)
ax.yaxis.set_ticks(np.arange(0,22,5),minor=False)
ax.yaxis.set_tick_params(which='minor', length=3, width=2.0*axis_lw/3.0)

# ax.yaxis.set_ticks(np.logspace(-16, 2, num=10))
# ax.yaxis.set_tick_params(which='minor', length=3, width=2.0*axis_lw/3.0)



leg = ax.legend([line1, line2], ['Kona SVD', 'SNOPT'], \
                loc=(0.1, 0.8), numpoints=1, prop={'size':6},  borderpad=0.75, handlelength=4)
rect = leg.get_frame()
rect.set_linewidth(axis_lw)
for t in leg.get_texts():
    t.set_fontsize(10)    # the legend text fontsize

plt.show()      
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import pdb, argparse


parser = argparse.ArgumentParser()
parser.add_argument("--case", help='tiny, small, medium', type=str, default='tiny')
parser.add_argument("--color", help='True, False', type=int, default=1)
args = parser.parse_args()

case = args.case
pic_color = args.color

if case == 'tiny':
    num_design, nlcost, dt_onesolve = 128, 9, 0.0153
if case == 'small':
    num_design, nlcost, dt_onesolve = 512, 10, 0.0632
if case == 'medium':
    num_design, nlcost, dt_onesolve = 2048, 12, 0.2988

folder = '/home/pengfei/Developer/kona_mengp2/fstopo_output/'


pcd = 'pc4'                             # results from  svd_pc_cmu, or svd_pc4.py 
dir_kona = folder + case + '_' + pcd + '/'

fname = dir_kona + 'kona_hist.dat'

dtype_cols = np.dtype([('outer_iter', 'i4'),('inner_iter', 'i4'), ('objective', 'float64'), 
        ('optim_2', 'float'), ('complem_2', 'float64'), ('feas_2', 'float64'), ('mu', 'float64'),
        ('optim_inf', 'float'), ('complem_inf', 'float64'), ('feas_inf', 'float64')  ])
kona_datas = np.loadtxt(fname, dtype=dtype_cols, skiprows = 2, usecols = (0,1,3,5,6,7,11, 12,13,14))


tname = dir_kona + 'kona_timings.dat'
dtype_cols2 = np.dtype([('outer_iter', 'i4'), ('time', 'float64')])
kona_timings = np.loadtxt(tname, dtype=dtype_cols2, skiprows = 2, usecols = (0,2))

# when inner_iter > 1 before mu = 0, display only one inner_iter, to make plots clean
iter_unique, indices = np.unique(kona_datas['outer_iter'], return_index=True)
last_indices = indices[-1]
last_inners = range(last_indices+1, len(kona_datas['outer_iter']))
new_indices = np.hstack([indices, np.array(last_inners)])

kona_time_svd = kona_timings['time'] / dt_onesolve
kona_data_svd = kona_datas[new_indices.astype(int)]

# --------------- Add the Identity PC for comparison ----------------
pcd = 'eye'
dir_kona = folder + case + '_' + pcd + '/'

fname2 = dir_kona + 'kona_hist.dat'
kona_datas2 = np.loadtxt(fname2, dtype=dtype_cols, skiprows = 2, usecols = (0,1,3,5,6,7,11, 12,13,14))

tname2 = dir_kona + 'kona_timings.dat'
kona_timings2 = np.loadtxt(tname2, dtype=dtype_cols2, skiprows = 2, usecols = (0,2))

# when inner_iter > 1 before mu = 0, display only one inner_iter, to make plots clean
iter_unique2, indices2 = np.unique(kona_datas2['outer_iter'], return_index=True)
last_indices2 = indices2[-1]
last_inners2 = range(last_indices2+1, len(kona_datas2['outer_iter']))
new_indices2 = np.hstack([indices2, np.array(last_inners2)])

kona_data_eye = kona_datas2[new_indices2.astype(int)]

kona_time_eye = kona_timings2['time'][:-1]  / dt_onesolve
# kona_data_eye = np.delete(kona_data_eye, -2, 0)

# ------------- SNOPT data -----------------
# --------------------------------------------------------------------
##### result files stored in  ~/Developer/kona_mengp2/results3/SNOPT/
pcd = 'snopt'

dir_snopt = folder + case + '_' + pcd + '/'

snopt_name = dir_snopt + 'SNOPT_summary.awk'
dtype_sn = np.dtype([('outer_iter', 'i4'),('nCon', 'i4'), ('feasibility', 'float64'),('optimality', 'float'), ('merit', 'float')])
snopt_data = np.loadtxt(snopt_name, dtype=dtype_sn, skiprows = 1, usecols = (0,3,4,5,6))
nCon_idx = snopt_data['nCon']

tname = dir_snopt + 'SNOPT_timings.dat'
dtype_cols2 = np.dtype([('ncon', 'i4'), ('cost', 'i4'), ('time', 'float64')])
snopt_time_s = np.loadtxt(tname, dtype=dtype_cols2, skiprows = 0, usecols = (0,1,3))

snopt_time = snopt_time_s['time'][nCon_idx-1] / dt_onesolve
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


fig = plt.figure(figsize=(7,4), facecolor=None)
# fig = plt.figure(figsize=(7,4), facecolor=None)
ax = fig.add_subplot(111)

if pic_color == 1:
    line1, = ax.semilogy(kona_time_eye, kona_data_eye['optim_inf']/kona_data_eye['optim_inf'][0], '-k^', linewidth=1.0, ms=6.0, mfc='w', mew=1.0) 
    line2, = ax.semilogy(kona_time_eye, kona_data_eye['complem_inf'], '--k^', linewidth=1.0, ms=6.0, mfc='w', mew=1.0)  #/kona_data_eye['complem_inf'][1]
    line3, = ax.semilogy(kona_time_eye, kona_data_eye['feas_inf'], ':k^', linewidth=1.0, ms=6.0, mfc='w', mew=1.0)    # /kona_data_eye['feas_inf'][1]

    line4, = ax.semilogy(kona_time_svd, kona_data_svd['optim_inf']/kona_data_svd['optim_inf'][0], '-ro', linewidth=1.0, ms=6.0, mfc='w', mew=1.0) 
    line5, = ax.semilogy(kona_time_svd, kona_data_svd['complem_inf'], '--ro', linewidth=1.0, ms=6.0, mfc='w', mew=1.0)   # /kona_data_svd['complem_inf'][1]
    line6, = ax.semilogy(kona_time_svd, kona_data_svd['feas_inf'], ':ro', linewidth=1.0, ms=6.0, mfc='w', mew=1.0)   # /kona_data_svd['feas_inf'][1]

    line7, = ax.semilogy(snopt_time, snopt_data['optimality']/snopt_data['optimality'][0], '-bs', linewidth=1.0, ms=6.0, mfc='w', mew=1.0)    
    line8, = ax.semilogy(snopt_time, snopt_data['feasibility'], ':bs', linewidth=1.0, ms=6.0, mfc='w', mew=1.0)

else:
    line1, = ax.semilogy(kona_time_eye, kona_data_eye['optim_inf']/kona_data_eye['optim_inf'][0], '-k^', linewidth=1.0, ms=6.0, mfc='w', mew=1.0) 
    line2, = ax.semilogy(kona_time_eye, kona_data_eye['complem_inf'], '--k^', linewidth=1.0, ms=6.0, mfc='w', mew=1.0)  #/kona_data_eye['complem_inf'][1]
    line3, = ax.semilogy(kona_time_eye, kona_data_eye['feas_inf'], ':k^', linewidth=1.0, ms=6.0, mfc='w', mew=1.0)   #/kona_data_eye['feas_inf'][1]

    line4, = ax.semilogy(kona_time_svd, kona_data_svd['optim_inf']/kona_data_svd['optim_inf'][0], '-ko', linewidth=1.0, ms=6.0, mfc='w', mew=1.0) 
    line5, = ax.semilogy(kona_time_svd, kona_data_svd['complem_inf'], '--ko', linewidth=1.0, ms=6.0, mfc='w', mew=1.0)   #/kona_data_svd['complem_inf'][1]
    line6, = ax.semilogy(kona_time_svd, kona_data_svd['feas_inf'], ':ko', linewidth=1.0, ms=6.0, mfc='w', mew=1.0)   #/kona_data_svd['feas_inf'][1]

    line7, = ax.semilogy(snopt_time, snopt_data['optimality']/snopt_data['optimality'][0], '-ks', linewidth=1.0, ms=6.0, mfc='w', mew=1.0)    
    line8, = ax.semilogy(snopt_time, snopt_data['feasibility'], ':ks', linewidth=1.0, ms=6.0, mfc='w', mew=1.0)



ax.set_position([0.15, 0.13, 0.80, 0.83])                                # position relative to figure edges
ax.set_xlabel('Cost (equivalent DE solutions)', fontsize=axis_fs, weight='bold')

ax.set_ylabel('Rel Opt / Abs Comp, Feas', fontsize=axis_fs, weight='bold')
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


ax.yaxis.set_ticks(np.logspace(-7, 2, num=10))
ax.yaxis.set_tick_params(which='minor', length=3, width=2.0*axis_lw/3.0)
# textstr = 'Number of Design : %i'%num_design 
# ax.text(xmax*0.6, 15, textstr, fontsize=label_fs, weight='bold')

# if case is 'tiny':
leg_size = 2



leg = ax.legend([line1, line2, line3, line4, line5, line6, line7, line8], \
    ['noPC_opt','noPC_comp', 'noPC_feas', 'PC_opt', 'PC_comp', 'PC_feas',  'SNOPT_opt', 'SNOPT_feas'], \
                loc=(0.8, 0.01), numpoints=1, prop={'size':leg_size},  borderpad=0.75, handlelength=4)
rect = leg.get_frame()
rect.set_linewidth(axis_lw)
for t in leg.get_texts():
    t.set_fontsize(10)    # the legend text fontsize

plt.show()     

 

if pic_color == 1:
    fig_name = folder + case + '_color.eps' 
    fig.savefig(fig_name, format='eps', dpi=1200)
else:
    fig_name = folder + case + '_nocolor.eps' 
    fig.savefig(fig_name, format='eps', dpi=1200)    
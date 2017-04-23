import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import pdb


dir_konahist = '../test/approx_adj/'
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


plt.subplot(111)
line1, = plt.semilogy(kona_time, kona_data['optimality']/kona_data['optimality'][0], marker='^', linestyle='-', color='b', label='kona_optimality') 
line2, = plt.semilogy(kona_time, kona_data['feasibility']/kona_data['feasibility'][0], marker='o', linestyle='-', color='r', label='kona_feasibility')  
plt.legend([line1, line2], ['kona_optimality', 'kona_feasibility'], prop={'size':12})
plt.xlabel('cpu time seconds', fontsize=12)
plt.tick_params(labelsize=12)
plt.title('Structure Problem,  pc = Approximate Adjoint')


plt.show()     #  max( abs(konaX - pyoptX)/norm(pyoptX) )
fig = plt.gcf()
fig.set_size_inches(4.6, 4.6)
fig_name = './figures/structure.eps'
fig.savefig(fig_name, format='eps', dpi=1200)
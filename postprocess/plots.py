import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import pdb

dir_konahist = '../kona_latest/src/kona/test/output/'
fname = dir_konahist + 'kona_hist.dat'
dtype_cols = np.dtype([('outer_iter', 'i4'),('inner_iter', 'i4'), ('objective', 'float64'), ('optimality', 'float'), ('feasibility', 'float64')])
kona_data = np.loadtxt(fname, dtype=dtype_cols, skiprows = 3, usecols = (0,1,3,5,6))

tname = dir_konahist + 'kona_timings.dat'
dtype_cols2 = np.dtype([('outer_iter', 'i4'), ('time', 'float64')])
kona_time = np.loadtxt(tname, dtype=dtype_cols2, skiprows = 3, usecols = (0,2))

# ----- Processing "SNOPT_summary.out" file 

# run ./README under '~/Developer/kona_mengp2/kona_latest/src/kona/test/output' in Linux system

snopt_name = dir_konahist + 'SNOPT_summary.awk'
dtype_sn = np.dtype([('outer_iter', 'i4'),('nCon', 'i4'), ('feasibility', 'float64'),('optimality', 'float'), ('merit', 'float')])
snopt_data = np.loadtxt(snopt_name, dtype=dtype_sn, skiprows = 1, usecols = (0,3,4,5,6))
nCon_idx = snopt_data['nCon']


tname = dir_konahist + 'SNOPT_timings.dat'
dtype_cols2 = np.dtype([('outer_iter', 'i4'), ('time', 'float64')])
snopt_time_s = np.loadtxt(tname, dtype=dtype_cols2, skiprows = 2, usecols = (0,2))
snopt_time = snopt_time_s['time'][nCon_idx-1]

plt.subplot(121)
line1, = plt.semilogy(kona_time['time'], kona_data['optimality'], marker='^', linestyle='-', color='b', label='kona_optimality') 
line2, = plt.semilogy(kona_time['time'], kona_data['feasibility'], marker='o', linestyle='-', color='r', label='kona_feasibility')  
plt.legend([line1, line2], ['kona_optimality', 'kona_feasibility'], prop={'size':12})
plt.xlabel('cpu time seconds', fontsize=18)


plt.subplot(122)
line3, = plt.semilogy(snopt_time, snopt_data['optimality'], marker='^', linestyle='-.', color='b', label='snopt_optimality')    
line4, = plt.semilogy(snopt_time, snopt_data['feasibility'], marker='o', linestyle='-.', color='r', label='snopt_feasibility')

plt.xlabel('cpu time seconds', fontsize=18)
plt.legend([line3, line4], ['snopt_optimality', 'snopt_feasibility'], prop={'size':12})

# plt.suptitle('Toy Problem, num design = 500, max( abs(konaX - pyoptX)/norm(pyoptX) ) = 0.000694')
plt.suptitle('Toy Problem, num design = 400, max( abs(konaX - pyoptX)/norm(pyoptX) ) = ')
plt.show()     
import numpy as np
import os, sys

# remove the last line "Wall time ... " in kona_hist.dat file for easier handle with np.loadtxt()

os.system('sed -i   "s/Wall time.*//g" kona_hist.dat')

# merge kona_hist.dat and kona_timings.dat into one file
os.system("pr -J -m -t kona_hist.dat kona_timings.dat > kona.dat")

# load datafile using numpy.loadtxt, and then plot 
# data = np.loadtxt('kona_th.dat')




import matplotlib 
matplotlib.use('agg') 
import matplotlib.pyplot as plt
from matplotlib import rc, rcParams
from matplotlib.pyplot import *
from matplotlib.legend_handler import HandlerLine2D

import numpy as np
from scipy import interpolate
import scipy.stats
from scipy.integrate import quad

from source import *

data = np.genfromtxt('../AGS09.dat')
r = data[:,0]
d = data[:,2]
f = data[:,8]
# print(10**(d))
fraction = scipy.interpolate.interp1d(r,f,bounds_error=False,fill_value=0)
density = scipy.interpolate.interp1d(r,10**(d)*const.NAvo*const.cmINV_to_GeV**3*1e27,bounds_error=False,fill_value=0)
rnew = np.linspace(0,0.5,1000)
I = np.sum(fraction(rnew)*(rnew[1]-rnew[0])*density(rnew))


xue = np.logspace(-3,-1,100)
xumu = np.logspace(-4,-2,100)
en = np.linspace(0,17,100)
for ue42 in xue:
	for umu42 in xumu:
		for enu in en:
			Pmine(enu, -const.nue_to_nue, ue42, umu42)
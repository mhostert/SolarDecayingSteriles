import numpy as np
from scipy import interpolate
import scipy.stats
from scipy.integrate import quad

from source import *

def get_exp_flux(fluxfile,get_3sigma=False):

	if (fluxfile == "fluxes/b8spectrum.txt"):
		E, bf, high3, low3 = np.loadtxt(fluxfile, unpack=True,skiprows=16)
		# E *=1e-3
		bf *= const.B8FLUX
		high3 *= const.B8FLUX
		low3 *= const.B8FLUX
		flux = scipy.interpolate.interp1d(E, bf, fill_value=0.0, bounds_error=False)
		flux3h = scipy.interpolate.interp1d(E, high3, fill_value=0.0, bounds_error=False)
		flux3l = scipy.interpolate.interp1d(E, low3, fill_value=0.0, bounds_error=False)

	else:
		print("ERROR! Could not identify the fluxfile.")
	if get_3sigma:
		return flux,flux3h,flux3l
	else:
		return flux
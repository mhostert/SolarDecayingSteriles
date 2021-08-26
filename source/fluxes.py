import numpy as np
from scipy import interpolate
import scipy.stats
from scipy.integrate import quad

from source import *

def get_neutrino_flux(fluxfile,get_3sigma=False, SSM='B16'):

	if (fluxfile == "fluxes/b8spectrum.txt"):
		E, bf, high3, low3 = np.loadtxt(fluxfile, unpack=True,skiprows=16)
		# E *=1e-3

		if SSM=='AGSS09':
			norm = const.B8FLUX_AGSS09
		elif SSM=='GS98':
			norm = const.B8FLUX_GS98
		elif SSM=='B16':
			norm = const.B8FLUX_B16
		else:
			print(f"ERROR! Could not identify solar model {SSM}.")

		bf *= norm
		high3 *= norm
		low3 *= norm

		flux = scipy.interpolate.interp1d(E, bf, fill_value=0.0, bounds_error=False, kind='linear')
		flux3h = scipy.interpolate.interp1d(E, high3, fill_value=0.0, bounds_error=False, kind='linear')
		flux3l = scipy.interpolate.interp1d(E, low3, fill_value=0.0, bounds_error=False, kind='linear')

	else:
		print(f"ERROR! Could not identify the flux file {fluxfile}.")
	if get_3sigma:
		return flux,flux3h,flux3l
	else:
		return flux




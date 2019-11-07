import numpy as np
from scipy import interpolate
import scipy.stats
from scipy.integrate import quad

def get_nue_CCQE(xsfile):

	if (xsfile == "xsecs/GLOBES/XCC.dat"):
		El, nue, _,_, _,_,_ = np.loadtxt(xsfile, unpack=True)
		Enu = 10**(El)
		xs = scipy.interpolate.interp1d(Enu, nue*Enu*1e-38, fill_value=0.0, bounds_error=False)
		# print "Running MiniBooNE fluxes"
	return xs # cm^2

def get_nuebar_CCQE(xsfile):

	if (xsfile == "xsecs/GLOBES/XCC.dat"):
		El, _, _,_,nuebar,_,_ = np.loadtxt(xsfile, unpack=True)
		Enu = 10**(El)
		xs = scipy.interpolate.interp1d(Enu, nuebar*Enu*1e-38, fill_value=0.0, bounds_error=False)
		# print "Running MiniBooNE fluxes"
	return xs # cm^2

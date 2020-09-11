import numpy as np
from scipy import interpolate
import scipy.stats
from scipy.integrate import quad

from source import *


def get_nue_CCQE(xsfile):
	if (xsfile == "xsecs/GLOBES/XCC.dat"):
		El, nue, _,_, _,_,_ = np.loadtxt(xsfile, unpack=True)
		Enu = 10**(El)
		xs = scipy.interpolate.interp1d(Enu, nue*Enu*1e-38, fill_value=0.0, bounds_error=False)
	return xs # cm^2

def get_nuebar_CCQE(xsfile):
	if (xsfile == "xsecs/GLOBES/XCC.dat"):
		El, _, _,_,nuebar,_,_ = np.loadtxt(xsfile, unpack=True)
		Enu = 10**(El)
		xs = scipy.interpolate.interp1d(Enu, nuebar*Enu*1e-38, fill_value=0.0, bounds_error=False)
	return xs # cm^2

def get_IBD(xsfile):
	Enu, xnuebar = np.loadtxt(xsfile, unpack=True)
	xs = scipy.interpolate.interp1d(Enu, xnuebar*1e-38, fill_value=0.0, bounds_error=False)
	return xs # cm^2

def get_nuES(nu_flag,T0=0.0):
	if nu_flag==pdg.PDG_numu:
		Cv = -0.5 + 2*const.s2w
		Ca = -0.5
		CL = (Cv+Ca)/2.0
		CR = (Cv-Ca)/2.0
	elif nu_flag==pdg.PDG_numubar:
		Cv = -0.5 + 2*const.s2w
		Ca = -0.5
		CL = (Cv-Ca)/2.0
		CR = (Cv+Ca)/2.0
	elif nu_flag==pdg.PDG_nue:
		Cv = +0.5 + 2*const.s2w
		Ca = +0.5
		CL = (Cv+Ca)/2.0
		CR = (Cv-Ca)/2.0
	elif nu_flag==pdg.PDG_nuebar:
		Cv = +0.5 + 2*const.s2w
		Ca = +0.5
		CL = (Cv-Ca)/2.0
		CR = (Cv+Ca)/2.0

	prefactor = 2*const.Me*const.Gf**2/np.pi 
	units=3.9204e-28
	def xs(En):
		E = En*1e-3
		t1 	    = 2*E**2/(const.Me+2*E)
		return units*prefactor*((CL**2+CR**2)*(t1-T0) - CR**2/E*(t1**2 - T0**2) + CR**2/3/E/E*(t1**3 - T0**3) - CL*CR*const.Me*(t1**2-T0**2)/2/E/E)
	
	return xs # cm^2

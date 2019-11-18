import numpy as np
from scipy import interpolate
import scipy.stats
from scipy.integrate import quad
import matplotlib 
matplotlib.use('agg') 
import matplotlib.pyplot as plt
from matplotlib import rc, rcParams
from matplotlib.pyplot import *
from matplotlib.legend_handler import HandlerLine2D

import vegas
import gvar as gv
from scipy.stats import chi2

################################################################
# Compute the chi2 using the model independent limits provided by collaborations
def chi2_model_independent(exp,params,fluxfile):
	bin_e = exp.Enu_bin_e
	fluxlimit = exp.fluxlimit

	###########
	# SET BINS TO BE THE EXPERIMENTAL BINS
	bins = bin_e # bin edges

	###########
	# NUMU FLUX
	flux = fluxes.get_exp_flux(fluxfile)

	############
	# NUE/BAR XS
	xsec = lambda x : np.zeros(np.size(x))
	xsecbar = lambda x : np.ones(np.size(x))

	############
	# efficiencies
	enu_eff= bins
	eff= np.ones((np.size(bins)-1))

	############
	# number of events
	NCASCADE, dNCASCADE = integrands.RATES_dN_HNL_CASCADE_NU_NUBAR(\
												flux=flux,\
												xsec=xsec,\
												xsecbar=xsecbar,\
												dim=3,\
												enumin=0,\
												enumax=16.8,\
												params=params,\
												bins=bins,\
												PRINT=False,\
												enu_eff=enu_eff,\
												eff=eff,
												exp=exp.exp_name)

	chi2 = np.sum( 2.71 * (dNCASCADE)**2/fluxlimit**2 )
	return chi2, chi2/(np.size(bins)-2)


def chi2_binned_rate(NP_MC,back_MC,D,sys):

	err_flux = sys[0]
	err_back = sys[1] 

	chi2bin = lambda beta : 2*np.sum(NP_MC*(1+beta[0]) + back_MC*(1+beta[1]) - D + myXLOG(D, D/(NP_MC*(1+beta[0]) + back_MC*(1+beta[1])) ) ) + beta[0]**2/(err_flux**2) + beta[1]**2 /(err_back**2) 
	
	res = scipy.optimize.minimize(chi2bin, [0.0,0.0])
	f = chi2bin(res.x)
	if np.abs(np.sum(res.x))>1:
		return 1e100
	else:
		return f

def myXLOG(d,x):
	return np.array([ (di*np.log(xi) if xi > 0 else 0) for di,xi in zip(d,x)])

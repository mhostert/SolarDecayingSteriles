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

from source import *

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



################################################################
# Compute the chi2 using the model independent limits provided by collaborations
def chi2_spectral(exp,params,fluxfile):

	###########
	# SET BINS TO BE THE EXPERIMENTAL BINS
	bins = exp.bin_e # bin edges
	bin_c = exp.bin_c # bin edges

	###########
	# NUMU FLUX
	flux = fluxes.get_exp_flux(fluxfile)

	############
	# NUE/BAR XS
	xsec = lambda x : np.zeros(np.size(x))
	xsfile="xsecs/IBD_160106169/TCS_CC_anue_p_1026_SV.txt"
	xsecbar = xsecs.get_IBD(xsfile)

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
												smearing_function=exp.smearing_function)

	err_flux = 0.1
	err_back = exp.err_back

	NP_MC = dNCASCADE[bin_c<17.3]*exp.norm
	back_MC = exp.MCall_binned[bin_c<17.3]
	D = exp.data[bin_c<17.3]

	print D/(NP_MC+back_MC)
	print D
	# print back_MC

	dof = np.size(D)-1

	chi2bin = lambda beta : 2*np.sum(NP_MC*(1+beta[0]) + back_MC*(1+beta[1]) - D + myXLOG(D, D/(NP_MC*(1+beta[0]) + back_MC*(1+beta[1])) ) ) + beta[0]**2 /(err_flux**2) + beta[1]**2 /(err_back**2) 
	
	res = scipy.optimize.minimize(chi2bin, [0.0,0.0])
	return chi2bin(res.x)-chi2bin([-1,0])



def myXLOG(d,x):
	return np.array([ (di*np.log(xi) if xi > 0 else 0) for di,xi in zip(d,x)])
	# print x
	# return d*np.log(x)
############
# NUMU FLUX
fluxfile = "fluxes/b8spectrum.txt"

###########
# DECAY MODEL PARAMETERS
params = model.decay_model_params(const.SCALAR)
params.gx		= 1.0
params.Ue4		= np.sqrt(0.001)
params.Umu4		= np.sqrt(0.002)*0
params.UD4		= np.sqrt(1.0-params.Ue4*params.Ue4-params.Umu4*params.Umu4)
params.m4		= 300e-9 # GeV
params.mBOSON  = 0.1*params.m4 # GeV

KAM = exps.kamland_limit()
BOR = exps.kamland_limit()

if __name__ == "__main__":
	# print chi2_model_independent(KAM,params,fluxfile)
	# print chi2_model_independent(BOR,params,fluxfile)
	print chi2_spectral(exps.kamland_data(), params, fluxfile)
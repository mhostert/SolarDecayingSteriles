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
												eff=eff)

	chi2 = np.sum( 2.71 * (dNCASCADE)**2/fluxlimit**2 )
	return chi2, chi2/(np.size(bins)-2)

############
# NUMU FLUX
fluxfile = "fluxes/b8spectrum.txt"

###########
# DECAY MODEL PARAMETERS
params = model.decay_model_params(const.SCALAR)
params.gx		= 1.0
params.Ue4		= 0.1
params.Umu4		= np.sqrt(0.002)
params.UD4		= np.sqrt(1.0-params.Ue4*params.Ue4-params.Umu4*params.Umu4)
params.m4		= 300e-9 # GeV
params.mBOSON  = 0.1*params.m4 # GeV

KAM = exps.kamland_limit()
BOR = exps.kamland_limit()


print chi2_model_independent(KAM,params,fluxfile)
print chi2_model_independent(BOR,params,fluxfile)
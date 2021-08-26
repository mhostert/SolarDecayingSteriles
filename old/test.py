import numpy as np
from scipy import interpolate
import matplotlib 
matplotlib.use('agg') 
import matplotlib.pyplot as plt
from matplotlib import rc, rcParams
from matplotlib.pyplot import *

import vegas
import gvar as gv

from source import * 


##########
# integration evaluations
rates.NEVALwarmup = 1e4
rates.NEVAL = 1e4

# ############
# # NUMU FLUX
# fluxfile = "fluxes/b8spectrum.txt"
# flux, flux3h, flux3l = fluxes.get_neutrino_flux(fluxfile, get_3sigma=True)
# norm = 1e-55


###########
# NUMU FLUX
fluxfile = "fluxes/b8spectrum.txt"
flux = fluxes.get_neutrino_flux(fluxfile)

############
# NUE/BAR XS
xsfile="xsecs/IBD_160106169/TCS_CC_anue_p_1026_SV.txt"
xsec = lambda x : np.zeros(np.size(x)) 
xsecbar = lambda x: np.ones(np.size(x))*1e-38

bins = np.linspace(0.0,14.5,50)
dx = (bins[1:] - bins[:-1])
bin_c = bins[:-1] + dx/2.0

#############
# efficiencies
enu_eff= bins
eff= np.ones((np.size(bins)-1))
identity = lambda TTT : TTT

############
# DECAY MODEL PARAMETERS
params = model.decay_model_params(const.SCALAR)
params.gx		= 1.0
params.Ue4		= np.sqrt(0.01)
params.Umu4		= np.sqrt(0.001)
params.UD4		= np.sqrt(1.0-params.Ue4*params.Ue4-params.Umu4*params.Umu4)
params.m4		= 300e-9 # GeV

params.mBOSON  = 0.9*params.m4 # GeV
NCASCADE, dNCASCADE = rates.RATES_dN_HNL_CASCADE_NU_NUBAR(\
											flux=flux,\
											xsec=xsec,\
											xsecbar=xsecbar,\
											dim=3,\
											enumin=0,\
											enumax=const.Enu_END_OF_SPECTRUM,\
											params=params,\
											bins=bins,\
											PRINT=True,\
											enu_eff=enu_eff,\
											eff=eff,
											smearing_function=identity)

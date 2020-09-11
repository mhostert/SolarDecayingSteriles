import numpy as np
import matplotlib 
matplotlib.use('agg') 
import matplotlib.pyplot as plt
from matplotlib import rc, rcParams
from matplotlib.pyplot import *
from matplotlib.legend_handler import HandlerLine2D

import vegas
import gvar as gv

from source import *

rates.NEVALwarmup = 5e4
rates.NEVAL = 1e5

############
# DECAY MODEL PARAMETERS
params = model.decay_model_params(const.SCALAR)
params.gx		= 1.0
params.Ue4		= np.sqrt(2e-2)
params.Umu4		= np.sqrt(1e-3)
params.Utau4	= np.sqrt(1e-3)*0
params.UD4		= np.sqrt(1.0-params.Ue4*params.Ue4-params.Umu4*params.Umu4)
params.m4		= 300e-9 # GeV
# params.mBOSON   = 0.1*params.m4 # GeV

fluxfile = "fluxes/b8spectrum.txt"
xsfile="xsecs/IBD_160106169/TCS_CC_anue_p_1026_SV.txt"



############
# NUMU FLUX
flux = fluxes.get_exp_flux(fluxfile)
############
# NUE/BAR XS
xsec = lambda x : np.zeros(np.size(x)) 
xsecbar = xsecs.get_IBD(xsfile)
exp = exps.borexino_limit()
expdata = exps.borexino_data()

# elin = np.linspace(7.8,8.8,1000)
# print np.sum(flux(elin))*(elin[1]-elin[0])
# print np.sum(exp.fluxlimit[(exp.Enu_bin_c<17)&(exp.Enu_bin_c>7.8)])
# print np.sum(flux(elin)/const.B8FLUX*xsecbar(elin)*expdata.norm)*(elin[1]-elin[0])*138/0.36
	

#####################
# BOREXINO
params.mBOSON  = 0.1*params.m4 # GeV
borexino.plot(params,fluxfile,xsfile)
plt.show()
params.mBOSON  = 0.5*params.m4 # GeV
borexino.plot(params,fluxfile,xsfile)
params.mBOSON  = 0.9*params.m4 # GeV
borexino.plot(params,fluxfile,xsfile)



#####################
# KAMLAND AND SUPER -- lower mixing
params.Ue4 = np.sqrt(1e-2)

params.mBOSON  = 0.1*params.m4 # GeV
kamland.plot(params,fluxfile,xsfile)
superk.plot(params,fluxfile,xsfile)

params.mBOSON  = 0.5*params.m4 # GeV
kamland.plot(params,fluxfile,xsfile)
superk.plot(params,fluxfile,xsfile)

params.mBOSON  = 0.9*params.m4 # GeV
kamland.plot(params,fluxfile,xsfile)
superk.plot(params,fluxfile,xsfile)

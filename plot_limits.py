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
# SETUP
################################################################
exp = exps.borexino_limit()
Bbin_c = exp.Enu_bin_c
Bbin_w = exp.Enu_bin_w
Bbin_e = exp.Enu_bin_e
Bfluxlimit = exp.fluxlimit
BfluxlimitATM = exp.fluxlimitATM

exp2 = exps.kamland_limit()
Kbin_c = exp2.Enu_bin_c
Kbin_w = exp2.Enu_bin_w
Kbin_e = exp2.Enu_bin_e
Kfluxlimit = exp2.fluxlimit

############
# NUMU FLUX
fluxfile = "fluxes/b8spectrum.txt"
flux, flux3h, flux3l = fluxes.get_exp_flux(fluxfile, get_3sigma=True)
norm = 1e-55


################################################################
# PLOTTING THE EVENT RATES 
################################################################
fsize=10
rc('text', usetex=True)
rcparams={'axes.labelsize':fsize,'xtick.labelsize':fsize,'ytick.labelsize':fsize,\
				'figure.figsize':(1.2*3.7,1.1*2.3617)	}
rc('font',**{'family':'serif', 'serif': ['computer modern roman']})
matplotlib.rcParams['hatch.linewidth'] = 0.1  # previous pdf hatch linewidth

rcParams.update(rcparams)
axes_form  = [0.15,0.16,0.8,0.74]
fig = plt.figure()
ax = fig.add_axes(axes_form)

E = np.linspace(0.001,16.0,1000)

#########################################################################

###########
# NUMU FLUX
fluxfile = "fluxes/b8spectrum.txt"
flux = fluxes.get_exp_flux(fluxfile)
############
# NUE/BAR XS
xsfile="xsecs/IBD_160106169/TCS_CC_anue_p_1026_SV.txt"
xsec = lambda x : np.zeros(np.size(x)) 
xsecbar = lambda x : np.ones(np.size(x)) 

bins = Bbin_e
dx = Bbin_w
bin_c = Bbin_c

# efficiencies
enu_eff= bins
eff= np.ones((np.size(dx)))


###########
# DECAY MODEL PARAMETERS
params = model.vector_model_params()
params.gx		= 1.0
params.Ue4		= 0.1
params.Umu4		= np.sqrt(0.01)
params.UD4		= np.sqrt(1.0-params.Ue4*params.Ue4-params.Umu4*params.Umu4)
params.m4		= 300e-9 # GeV
params.mzprime  = 0.9*params.m4 # GeV

NCASCADE, dNCASCADE = integrands.RATES_dN_HNL_CASCADE_NU_NUBAR(\
											flux=flux,\
											xsec=xsec,\
											xsecbar=xsecbar,\
											dim=3,\
											enumin=0,\
											enumax=16.8,\
											params=params,\
											bins=bins,\
											PRINT=True,\
											enu_eff=enu_eff,\
											eff=eff)

params = model.vector_model_params()
params.gx		= 1.0
params.Ue4		= 0.1
params.Umu4		= np.sqrt(0.01)
params.UD4		= np.sqrt(1.0-params.Ue4*params.Ue4-params.Umu4*params.Umu4)
params.m4		= 300e-9 # GeV
params.mzprime  = 0.1*params.m4 # GeV

NCASCADE2, dNCASCADE2 = integrands.RATES_dN_HNL_CASCADE_NU_NUBAR(\
											flux=flux,\
											xsec=xsec,\
											xsecbar=xsecbar,\
											dim=3,\
											enumin=0,\
											enumax=16.8,\
											params=params,\
											bins=bins,\
											PRINT=True,\
											enu_eff=enu_eff,\
											eff=eff)

print '%.2g'%np.sum(dNCASCADE[bin_c>1.8]*norm)
print '%.2g'%const.B8FLUX

##########################################################################

ax.step(np.append(Kbin_c-0.5,Kbin_c[-1]+0.5), np.append(Kfluxlimit,1e8), where = 'post', color='indigo', lw=0.5)
ax.fill_between(np.append(Kbin_c-0.5,Kbin_c[-1]+0.5), np.append(Kfluxlimit,1e8), np.ones(np.size(Kbin_c)+1)*1e10, step = 'post', lw=0.0, alpha=0.5, color='indigo')

ax.step(np.append(Bbin_c-0.5,Bbin_c[-1]+0.5), np.append(Bfluxlimit,1e8), where = 'post', color='blue', lw=0.5)
ax.fill_between(np.append(Bbin_c-0.5,Bbin_c[-1]+0.5), np.append(Bfluxlimit,1e8), np.ones(np.size(Bbin_c)+1)*1e10, step = 'post', lw=0.0, alpha=0.5, color='dodgerblue')

ax.step(Bbin_c-0.5, dNCASCADE*1e-55, where='post',  lw=1.5, color='darkorange', label=r'$m_{Z^\prime}/m_4 = 0.9$')
ax.step(Bbin_c-0.5, dNCASCADE2*1e-55, where='post',  lw=1, color='darkgreen', label=r'$m_{Z^\prime}/m_4 = 0.1$')

# r'$\nu_4 \to\overline{\nu_e}$ ($%.1g$ cm$^{-2}$ s$^{-1}$)'%(np.sum(dNCASCADE*1e-55))

ax.set_yscale('log')

ax.text(14,1e2,r'Borexino',fontsize=10,color='blue')
ax.text(14,5,r'KamLAND',fontsize=10,color='indigo')

##############
# STYLE
ax.legend(loc='upper right',frameon=False,ncol=2, fontsize=fsize)
# ax.set_title(r'$m_h = %.0f$ keV,\, $m_{Z^\prime} = %.0f$ keV, \, $|U_{\mu h}| = %.3f$'%(params.m4*1e6,params.mzprime*1e6,params.Umu4), fontsize=fsize)
ax.set_title(r'$m_h = %.0f$ eV, \, $|U_{e 4}|^2 = %.3f$'%(params.m4*1e9, params.Umu4**2), fontsize=fsize)

ax.set_xlim(np.min(Bbin_e),17.3)
ax.set_ylim(1e0, 5e5)

ax.set_ylabel(r'$\Phi$ (cm$^{-2}$ s$^{-1}$)')
ax.set_xlabel(r'$E_\nu/$MeV')
fig.savefig('plots/Fluxlimit.pdf')
plt.show()
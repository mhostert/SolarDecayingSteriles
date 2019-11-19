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

##########
# integration evaluations
rates.NEVALwarmup = 1e4
rates.NEVAL = 1e6

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

exp3 = exps.superk_limit()
Sbin_c = exp3.Enu_bin_c
Sbin_w = exp3.Enu_bin_w
Sbin_e = exp3.Enu_bin_e
Sfluxlimit = exp3.fluxlimit

############
# NUMU FLUX
fluxfile = "fluxes/b8spectrum.txt"
flux, flux3h, flux3l = fluxes.get_exp_flux(fluxfile, get_3sigma=True)
norm = 1


################################################################
# PLOTTING THE EVENT RATES 
################################################################
fsize=11
rc('text', usetex=True)
rcparams={'axes.labelsize':fsize,'xtick.labelsize':fsize,'ytick.labelsize':fsize,\
				'figure.figsize':(1.2*3.7,1.4*2.3617)	}
rc('font',**{'family':'serif', 'serif': ['computer modern roman']})
matplotlib.rcParams['hatch.linewidth'] = 0.1  # previous pdf hatch linewidth

rcParams.update(rcparams)
axes_form  = [0.15,0.15,0.82,0.75]
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

bins = np.linspace(0,16,50)
dx = (bins[1:] - bins[:-1])
bin_c = bins[:-1] + dx/2.0

# efficiencies
enu_eff= bins
eff= np.ones((np.size(dx)))

identity = lambda x : x
###########
# DECAY MODEL PARAMETERS
params = model.decay_model_params(const.SCALAR)
params.gx		= 1.0
params.Ue4		= np.sqrt(0.001)
params.Umu4		= np.sqrt(0.001)*0
params.UD4		= np.sqrt(1.0-params.Ue4*params.Ue4-params.Umu4*params.Umu4)
params.m4		= 300e-9 # GeV
params.mBOSON  = 0.9*params.m4 # GeV

NCASCADE, dNCASCADE = rates.RATES_dN_HNL_CASCADE_NU_NUBAR(\
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
											eff=eff,
											smearing_function=identity)

params = model.decay_model_params(const.SCALAR)
params.gx		= 1.0
params.Ue4		= np.sqrt(0.001)
params.Umu4		= np.sqrt(0.001)*0
params.Utau4		= np.sqrt(0.001)*0
params.UD4		= np.sqrt(1.0-params.Ue4*params.Ue4-params.Umu4*params.Umu4)
params.m4		= 300e-9 # GeV
params.mBOSON  = 0.5*params.m4 # GeV

NCASCADE2, dNCASCADE2 = rates.RATES_dN_HNL_CASCADE_NU_NUBAR(\
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
											eff=eff,
											smearing_function=identity)

params = model.decay_model_params(const.SCALAR)
params.gx		= 1.0
params.Ue4		= np.sqrt(0.001)
params.Umu4		= np.sqrt(0.001)*0
params.UD4		= np.sqrt(1.0-params.Ue4*params.Ue4-params.Umu4*params.Umu4)
params.m4		= 300e-9 # GeV
params.mBOSON  = 0.1*params.m4 # GeV

NCASCADE3, dNCASCADE3 = rates.RATES_dN_HNL_CASCADE_NU_NUBAR(\
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
											eff=eff,
											smearing_function=identity)

print '%.2g'%np.sum(dNCASCADE[bin_c>1.8]*norm)
print '%.2g'%const.B8FLUX

##########################################################################
if params.model == const.VECTOR:
	boson_string = r'$m_{Z^\prime}$'
	boson_file = 'vector'
elif params.model == const.SCALAR:
	boson_string = r'$m_\phi$'
	boson_file = 'scalar'

ax.step(np.append(Sbin_c-0.5,Sbin_c[-1]+0.5), np.append(Sfluxlimit,1e8), where = 'post', color='grey', lw=0.5)
ax.fill_between(np.append(Sbin_c-0.5,Sbin_c[-1]+0.5), np.append(Sfluxlimit,1e8), np.ones(np.size(Sbin_c)+1)*1e10, step = 'post', lw=0.0, alpha=0.5, color='grey')

ax.step(np.append(Kbin_c-0.5,Kbin_c[-1]+0.5), np.append(Kfluxlimit,1e8), where = 'post', color='indigo', lw=0.5)
ax.fill_between(np.append(Kbin_c-0.5,Kbin_c[-1]+0.5), np.append(Kfluxlimit,1e8), np.ones(np.size(Kbin_c)+1)*1e10, step = 'post', lw=0.0, alpha=0.5, color='indigo')

ax.step(np.append(Bbin_c-0.5,Bbin_c[-1]+0.5), np.append(Bfluxlimit,1e8), where = 'post', color='blue', lw=0.5)
ax.fill_between(np.append(Bbin_c-0.5,Bbin_c[-1]+0.5), np.append(Bfluxlimit,1e8), np.ones(np.size(Bbin_c)+1)*1e10, step = 'post', lw=0.0, alpha=0.5, color='dodgerblue')

ax.plot(bin_c-dx/2.0, dNCASCADE/dx, lw=1.5, color='darkorange', label=boson_string+r'/$m_4 = 0.9$')
ax.plot(bin_c-dx/2.0, dNCASCADE2/dx, lw=1, color='darkgreen', label=boson_string+r'/$m_4 = 0.5$')
ax.plot(bin_c-dx/2.0, dNCASCADE3/dx, lw=1, color='firebrick', label=boson_string+r'/$m_4 = 0.1$')

# r'$\nu_4 \to\overline{\nu_e}$ ($%.1g$ cm$^{-2}$ s$^{-1}$)'%(np.sum(dNCASCADE))

ax.set_yscale('log')

ax.text(14,1e2,r'Borexino',fontsize=10,color='blue')
ax.text(11.4,37,r'KamLAND',fontsize=10,color='indigo')
ax.text(14.5,5.5,r'SK-IV',fontsize=10,color='black')

##############
# STYLE
ax.legend(loc='lower left',frameon=False,ncol=1, fontsize=fsize)
# ax.set_title(r'$m_h = %.0f$ keV,\, $m_{Z^\prime} = %.0f$ keV, \, $|U_{\mu h}| = %.3f$'%(params.m4*1e6,params.mzprime*1e6,params.Umu4), fontsize=fsize)
ax.set_title(r'$m_4 = %.0f$ eV, \, $|U_{e 4}|^2 = %.3f$'%(params.m4*1e9, params.Ue4**2), fontsize=fsize)

ax.set_xlim(0,18.3)
ax.set_ylim(0.4, 2e5)

ax.set_ylabel(r'$\frac{{\rm d}\Phi}{{\rm d}E_\nu}$ $\times$ (cm$^{2}$ s MeV)')
ax.set_xlabel(r'$E_\nu/$MeV')
fig.savefig('plots/Fluxlimit_'+boson_file+'.pdf',rasterized=True)
plt.show()
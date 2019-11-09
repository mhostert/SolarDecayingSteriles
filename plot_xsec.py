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

EXP_FLAG = const.MINIBOONE

if EXP_FLAG == const.MINIBOONE:
	Nucleons_per_target = 14.0
	P_per_target = 8.0
	TARGETS = (818e6) * const.NAvo
	POTS = 12.84e20
	A_NUMBER = 12.0
	Enu_BEG_OF_SPECTRUM = 0.0
	Enu_END_OF_SPECTRUM = 2.0
	L =  0.541

############
# NUMU FLUX
fluxfile = "fluxes/b8spectrum.txt"
flux, flux3h, flux3l = fluxes.get_exp_flux(fluxfile, get_3sigma=True)
norm = 1

############
# NUE/BAR XS
xsfile="xsecs/IBD_160106169/TCS_CC_anue_p_1026_SV.txt"
xsecSV = xsecs.get_IBD(xsfile)
xsfile="xsecs/IBD_160106169/TCS_CC_anue_p_VB.txt"
xsecVB = xsecs.get_IBD(xsfile)
xsfile="xsecs/IBD_160106169/TCS_CC_anue_p_1026_CVC.txt"
xsecCVC = xsecs.get_IBD(xsfile)

xsec_nuebar_ES = xsecs.get_nuES(pdg.PDG_nuebar)
xsec_nue_ES = xsecs.get_nuES(pdg.PDG_nue)




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
axes_form  = [0.15,0.15,0.66,0.75]
fig = plt.figure()
ax = fig.add_axes(axes_form)
ax2 = ax.twinx()
 # ax.yaxis.tick_right() 
 # ax2.yaxis.tick_left() 
ax2.set_zorder(ax.get_zorder() + 1)
ax2.patch.set_visible(False)


E = np.linspace(0.001,16.0,1000)




#########################################################################

exp = exps.borexino_data()
###########
# NUMU FLUX
fluxfile = "fluxes/b8spectrum.txt"
flux = fluxes.get_exp_flux(fluxfile)
############
# NUE/BAR XS
xsfile="xsecs/IBD_160106169/TCS_CC_anue_p_1026_SV.txt"
xsec = lambda x : np.zeros(np.size(x)) 
xsecbar = lambda x : np.ones(np.size(x)) 
###########
# DECAY MODEL PARAMETERS
params = model.vector_model_params()
params.gx		= 1.0
params.Ue4		= 0.1
params.Umu4		= np.sqrt(1e-2)
params.UD4		= np.sqrt(1.0-params.Ue4*params.Ue4-params.Umu4*params.Umu4)
params.m4		= 300e-9 # GeV
params.mzprime  = 0.1*params.m4 # GeV
############
# EXPERIMENTAL DATA AND BINS
bins = np.linspace(0.00,16.8,50)
dx = bins[1:]-bins[:-1]
bin_c= bins[:-1] + dx/2.0

# efficiencies
enu_eff= bins
eff= np.ones((np.size(dx)))

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

# ax.plot(E, flux(E)/np.max(flux(E)), color='brown',lw=0.5, linestyle='--')
ax.fill_between(E, flux3l(E)/np.max(flux(E))*1.11,flux3l(E)/np.max(flux(E))*0.89*0, facecolor='orange', edgecolor='darkorange', alpha=0.7)
# ax.fill_between(E, flux3l(E)/np.max(flux(E))*0.813,flux3l(E)/np.max(flux(E))*0.776, color='orange', alpha=0.7)

ax.fill_between(bin_c-dx/2.0, dNCASCADE/np.max(dNCASCADE)*1.11,dNCASCADE/np.max(dNCASCADE)*0.89*0, facecolor='darkgrey',edgecolor='black',lw=0.5, linestyle='-',alpha=0.8)
# ax.fill_between(bin_c-dx/2.0, dNCASCADE/np.max(dNCASCADE)*0.813,dNCASCADE/np.max(dNCASCADE)*0.776, facecolor='None',edgecolor='black',hatch='//////////',lw=0.5, linestyle='-')


##########################################################################

ax2.plot(E, xsecSV(E), color='dodgerblue',lw=0.8)
ax2.plot(E, xsec_nue_ES(E), color='darkgreen',lw=0.8,linestyle='--')
ax2.plot(E, xsec_nuebar_ES(E), color='purple',lw=0.8,linestyle=':')
# ax2.plot(E, xsecVB(E), color='orange',lw=0.8,ls='--')
# ax2.plot(E, xsecCVC(E), color='red',lw=0.8,ls='--')
ax2.set_yscale('log')
ax.set_yscale('log')
# ax2.set_xscale('log')
# ax.set_xscale('log')


ax2.text(12,1.5e-41,r'$\sigma_{\rm IBD}$',color='dodgerblue', rotation=12)
ax2.text(12.5,1.7e-43,r'$\sigma_{\nu_e - e}$',color='darkgreen', rotation=5)
ax2.text(12.5,2.8e-44,r'$\sigma_{\overline{\nu}_e - e}$',color='purple', rotation=5)

ax.text(8.5,1.1,r'$\Phi^{\nu_e}(^8$B$)$',color='darkorange', rotation=0)
ax.text(0.5,1.1,r'$\Phi(\nu_h\to \overline{\nu_e})$',color='black', rotation=0)


##############
# STYLE
ax.legend(loc='upper right',frameon=False,ncol=1)
# ax.set_title(r'$m_h = %.0f$ keV,\, $m_{Z^\prime} = %.0f$ keV, \, $|U_{\mu h}| = %.3f$'%(params.m4*1e6,params.mzprime*1e6,params.Umu4), fontsize=fsize)
ax.set_title(r'$m_h = %.0f$ eV,\, $m_{Z^\prime}/m_h = %.2f$, \, $|U_{e h}|^2 = %.3f$'%(params.m4*1e9,params.mzprime/params.m4,params.Umu4**2), fontsize=fsize)

ax.set_xlim(np.min(E),np.max(E))
ax.set_ylim(5e-3, 2)
ax2.set_ylim(1e-45, 1e-40)

# (?# ax.set_ylabel(r'$\Phi$ $\nu/$cm$^2$/s'))
ax.set_ylabel(r'$\Phi$ (a.u.)')
ax2.set_ylabel(r'$\sigma/$cm$^2$')
ax.set_xlabel(r'$E_\nu/$MeV')
fig.savefig('plots/Spectrum_%.0f_MZ_%.0f.pdf'%(params.m4*1e9,params.mzprime*1e9))
plt.show()
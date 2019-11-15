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

############
# NUMU FLUX
fluxfile = "fluxes/b8spectrum.txt"
flux, flux3h, flux3l = fluxes.get_exp_flux(fluxfile, get_3sigma=True)
norm = 1e-55

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
params = model.decay_model_params(const.SCALAR)
params.gx		= 1.0
params.Ue4		= np.sqrt(0.01)
params.Umu4		= np.sqrt(0.01)*0
params.UD4		= np.sqrt(1.0-params.Ue4*params.Ue4-params.Umu4*params.Umu4)
params.m4		= 300e-9 # GeV
params.mBOSON  = 0.9*params.m4 # GeV

############
# EXPERIMENTAL DATA AND BINS
bins = np.linspace(0.00,16.8,50)
dx = bins[1:]-bins[:-1]
bin_c= bins[:-1] + dx/2.0

# efficiencies
enu_eff= bins
eff= np.ones((np.size(dx)))
identity = lambda x : x

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

ax.fill_between(E, flux3l(E),flux3l(E)/np.max(flux(E))*0.89*0, facecolor='orange', edgecolor='orange', alpha=0.5)
ax.plot(E, flux3l(E), color='darkorange',lw=0.7, linestyle='-')

ax.fill_between(bin_c-dx/2.0, dNCASCADE/dx,dNCASCADE/np.max(dNCASCADE)*0.89*0, facecolor='darkgrey',edgecolor='black',lw=0.5, linestyle='-',alpha=0.8)
ax.plot(bin_c-dx/2.0, dNCASCADE/dx, color='black',lw=0.4, linestyle='-')

print '%.2g'%np.sum(dNCASCADE[bin_c>1.8]*norm)
print '%.2g'%const.B8FLUX

##########################################################################

ax2.plot(E, xsecSV(E), color='dodgerblue',lw=0.8)
ax2.plot(E, xsec_nue_ES(E), color='darkgreen',lw=0.8, dashes=(2,1))
ax2.plot(E, xsec_nuebar_ES(E), color='indigo',lw=1, dashes=(5,1))
# ax2.plot(E, xsecVB(E), color='orange',lw=0.8,ls='--')
# ax2.plot(E, xsecCVC(E), color='red',lw=0.8,ls='--')
ax2.set_yscale('log')
ax.set_yscale('log')
# ax2.set_xscale('log')
# ax.set_xscale('log')


ax2.text(12,1.5e-41,r'$\sigma_{\rm IBD}$',color='dodgerblue', rotation=12)
ax2.text(12.5,1.7e-43,r'$\sigma_{\nu_e - e}$',color='darkgreen', rotation=5)
ax2.text(12.5,2.8e-44,r'$\sigma_{\overline{\nu}_e - e}$',color='indigo', rotation=5)

ax.text(2,7e4,r'$\Phi^{\nu_e}(^8$B$)$',color='saddlebrown', rotation=0)
ax.text(6,2e1,r'$\Phi(\nu_4\to \overline{\nu_e})$',color='black', rotation=0)


##############
# STYLE
##############
# STYLE
if params.model == const.VECTOR:
	boson_string = r'$m_{Z^\prime}$'
	boson_file = 'vector'
elif params.model == const.SCALAR:
	boson_string = r'$m_\phi$'
	boson_file = 'scalar'

ax.legend(loc='upper right',frameon=False,ncol=1)
ax.set_title(r'$m_4 = %.0f$ eV,\, '%(params.m4*1e9)+boson_string+r'$/m_4 = %.1f$, \, $|U_{e 4}|^2 = %.2f$'%(params.mBOSON/params.m4,params.Umu4**2), fontsize=fsize)

ax.set_xlim(np.min(E),np.max(E))
ax.set_ylim(1e1, 1*const.B8FLUX)
ax2.set_ylim(1e-45, 1e-40)

# (?# ax.set_ylabel(r'$\Phi$ $\nu/$cm$^2$/s'))
ax.set_ylabel(r'$\frac{{\rm d}\Phi}{{\rm d}E_\nu}$ $\times$ (cm$^{2}$ s MeV)')
ax2.set_ylabel(r'$\sigma/$cm$^2$')
ax.set_xlabel(r'$E_\nu/$MeV')
fig.savefig('plots/Spectrum_'+boson_file+'_%.0f_MZ_%.0f.pdf'%(params.m4*1e9,params.mBOSON*1e9),rasterized=True)

plt.show()
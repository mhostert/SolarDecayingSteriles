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
# integration evaluations
rates.NEVALwarmup = 1e4
rates.NEVAL = 1e6

############
# NUMU FLUX
fluxfile = "fluxes/b8spectrum.txt"
flux, flux3h, flux3l = fluxes.get_exp_flux(fluxfile, get_3sigma=True)
norm = 1e-55

############
# NUE/BAR XS
xsfile="xsecs/IBD_160106169/TCS_CC_anue_p_1026_SV.txt"
xsecSV = xsecs.get_IBD(xsfile)

xsec_nuebar_ES = xsecs.get_nuES(pdg.PDG_nuebar)
xsec_nue_ES = xsecs.get_nuES(pdg.PDG_nue)

###########
# NUMU FLUX
fluxfile = "fluxes/b8spectrum.txt"
flux = fluxes.get_exp_flux(fluxfile)

############
# NUE/BAR XS
xsfile="xsecs/IBD_160106169/TCS_CC_anue_p_1026_SV.txt"
xsec = lambda x : np.zeros(np.size(x)) 
xsecbar = lambda x : np.ones(np.size(x)) 

############
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

############
# efficiencies -- equal to 1
enu_eff= bins
eff= np.ones((np.size(dx)))
identity = lambda x : x

NCASCADE, dNCASCADE = rates.RATES_dN_HNL_CASCADE_NU_NUBAR(\
											flux=flux,\
											xsec=xsec,\
											xsecbar=xsecbar,\
											dim=3,\
											enumin=0,\
											enumax=const.Enu_END_OF_SPECTRUM,\
											params=params,\
											bins=bins,\
											PRINT=False,\
											enu_eff=enu_eff,\
											eff=eff,
											smearing_function=identity)


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

################
# FIGURE 2 axes
fig = plt.figure()
ax = fig.add_axes(axes_form)
ax2 = ax.twinx()
ax2.set_zorder(ax.get_zorder() + 1)
ax2.patch.set_visible(False)


E = np.linspace(0.001,16.3,1000)

################
# PLOT FLUXES
ax.fill_between(E, flux3l(E),flux3l(E)/np.max(flux(E))*0.89*0, facecolor='orange', edgecolor='', alpha=0.6)
ax.plot(E, flux3l(E), color='black',lw=0.7, linestyle='-')

ax.fill_between(bin_c-dx/2.0, dNCASCADE/dx,dNCASCADE/np.max(dNCASCADE)*0.89*0, facecolor='darkgrey',edgecolor='',lw=0.5, linestyle='-',alpha=0.8)
ax.plot(bin_c-dx/2.0, dNCASCADE/dx, color='black',lw=0.7, linestyle='-')

################
# PLOT XSEC
ax2.plot(E, xsecSV(E), color='dodgerblue',lw=1)
ax2.plot(E, xsec_nue_ES(E), color='darkgreen',lw=1, dashes=(2,1))
ax2.plot(E, xsec_nuebar_ES(E), color='indigo',lw=1, dashes=(6,1))


##############
# STYLE
##############
ax2.set_yscale('log')
ax.set_yscale('log')

# LABELS
ax2.text(12,1.5e-41,r'$\sigma_{\rm IBD}$',color='dodgerblue', rotation=13	, fontsize=11)
ax2.text(12.3,1.7e-43,r'$\sigma_{\nu_e - e}$',color='darkgreen', rotation=5, fontsize=11)
ax2.text(12.5,2.8e-44,r'$\sigma_{\overline{\nu_e} - e}$',color='indigo', rotation=5, fontsize=11)
ax.text(2,1e6,r'$\frac{{\rm d} \Phi^{\nu_e}}{{\rm d} E_\nu}  (^8{\rm B})$',color='darkorange', rotation=0, fontsize=10)
ax.text(7.5,2e1,r'$\frac{{\rm d} \Phi^{\overline{\nu_e}}}{{\rm d} E_\nu}$({\small decay})',color='black', rotation=0, fontsize=10)

# title
if params.model == const.VECTOR:
	boson_string = r'$m_{Z^\prime}$'
	boson_file = 'vector'
elif params.model == const.SCALAR:
	boson_string = r'$m_\phi$'
	boson_file = 'scalar'

def to_scientific_notation(number):
    a, b = '{:.4E}'.format(number).split('E')
    b = int(b)
    a = float(a)
    return r'$%.0f \times 10^{%i}$'%(a,b)
UEQSR = to_scientific_notation(params.Ue4**2)
ax.legend(loc='lower left',frameon=False,ncol=1,markerfirst=True)
ax.set_title(r'$m_4 = %.0f$ eV, '%(params.m4*1e9)+boson_string+r'$/m_4 = %.1f$,\, $|U_{e 4}|^2 = \,$'%(params.mBOSON/params.m4)+UEQSR, fontsize=0.95*fsize)

ax.set_xlim(np.min(E),np.max(E))
ax.set_ylim(1e1, 1*const.B8FLUX)
ax2.set_ylim(1e-45, 1e-40)

ax.set_ylabel(r'$\frac{{\rm d}\Phi}{{\rm d}E_\nu}$ \big[cm$^{-2}$ s$^{-1}$ MeV$^{-1}$\big]')
ax2.set_ylabel(r'$\sigma$ [cm$^2$]')
ax.set_xlabel(r'$E_\nu$ [MeV]')
fig.savefig('plots/Spectrum_'+boson_file+'_%.0f_MZ_%.0f.pdf'%(params.m4*1e9,params.mBOSON*1e9),rasterized=True)
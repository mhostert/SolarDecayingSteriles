import numpy as np
from scipy import interpolate
import scipy.stats
from scipy.integrate import quad

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
flux = fluxes.get_exp_flux(fluxfile)
norm = POTS*TARGETS /1e55

############
# NUE/BAR XS
xsec = xsecs.get_nue_CCQE(xsfile)
xsecbar = xsecs.get_nuebar_CCQE(xsfile)

############
# DECAY MODEL PARAMETERS
params = model.vector_model_params()
params.gx		= 1.0
params.Ue4		= 0.1
params.Umu4		= np.sqrt(4*1e-3)
params.UD4		= np.sqrt(1.0-params.Ue4*params.Ue4-params.Umu4*params.Umu4)
params.m4		= 50e-6 # GeV
params.mzprime  = 1e-6 # GeV

############
# OSC MODEL PARAMETERS
params_osc = model.osc_model_params()
params_osc.Ue4		= np.sqrt(0.1/2.0)
params_osc.Umu4		= np.sqrt(0.1/2.0)
params_osc.UD4		= np.sqrt(1.0-params.Ue4*params.Ue4-params.Umu4*params.Umu4)
params_osc.dm4SQR	= 0.4 # eV^2


############
# MINIBOONE DATA AND BINS
miniboone = exps.miniboone_data()
bins = miniboone.bin_e
dx =bins[1:] - bins[0:-1]
bin_c= bins[0:-1] + dx/2.0
# efficiencies
enu_eff= miniboone.enu_eff
eff= miniboone.eff


################################################################
# COMPUTING THE EVENT RATE INTEGRALS
################################################################

#############
# ONLY HNL DECAY
N,dNdEf = integrands.RATES_dN_HNL_TO_ZPRIME(flux=flux,\
											xsec=xsec,\
											dim=2,\
											enumin=Enu_BEG_OF_SPECTRUM,\
											enumax=Enu_END_OF_SPECTRUM,\
											params=params,\
											bins=bins,\
											PRINT=True,
											enu_eff=enu_eff,
											eff=eff)
N*=norm
dNdEf*=norm

#############
# HNL + ZPRIME DECAYS
NCASCADE, dNCASCADE = integrands.RATES_dN_HNL_CASCADE_NU_NUBAR(\
											flux=flux,\
											xsec=xsec,\
											xsecbar=xsecbar,\
											dim=3,\
											enumin=Enu_BEG_OF_SPECTRUM,\
											enumax=Enu_END_OF_SPECTRUM,\
											params=params,\
											bins=bins,\
											PRINT=True,\
											enu_eff=enu_eff,\
											eff=eff)
NCASCADE*=norm
dNCASCADE*=norm

print N, NCASCADE

#########
# ADD THE ZPRIME DECAYS TO THE HNL -> NU ZPRIME
NCASCADE += N
dNCASCADE += dNdEf
#########


##############
# TEST CASE FOR SBL OSCILLATIONS
N_osc,dNdEf_osc = integrands.RATES_SBL_OSCILLATION(flux=flux,\
											xsec=xsec,\
											dim=1,\
											enumin=Enu_BEG_OF_SPECTRUM,\
											enumax=Enu_END_OF_SPECTRUM,\
											params=params_osc,\
											bins=bins,\
											PRINT=False,\
											L=L,
											enu_eff=enu_eff,
											eff=eff)
N_osc*=norm
dNdEf_osc*=norm


################################################################
# PLOTTING THE EVENT RATES 
################################################################
fsize=11
rc('text', usetex=True)
rcparams={'axes.labelsize':fsize,'xtick.labelsize':fsize,'ytick.labelsize':fsize,\
				'figure.figsize':(1.2*3.7,1.4*2.3617)	}
rc('font',**{'family':'serif', 'serif': ['computer modern roman']})
rcParams.update(rcparams)
axes_form  = [0.15,0.15,0.82,0.76]
fig = plt.figure()
ax = fig.add_axes(axes_form)


##########
# PLOT UNITS
UNITS = 1 

##############
# NORMALIZE THE DISTRIBUTIONS
y= dNdEf/dx/UNITS
# y/=np.sum(y)
# y/=np.max(y)


yCASCADE= dNCASCADE/dx/UNITS
# yCASCADE/=np.sum(yCASCADE)
# yCASCADE/=np.max(yCASCADE)

yosc= dNdEf_osc/dx/UNITS
# yosc/=np.sum(yosc)
# yosc/=np.max(yosc)


# Check that my dNdE distribution against a scaled event distribution
Ef=np.linspace(0,2,1000)
e=flux(Ef)*xsec(Ef)*norm*0.8e-3*1e55/UNITS
h,_ = np.histogram(Ef,weights=e,bins=bins,density=True)
# ax.bar(bin_c,h/np.sum(h)*np.sum(e*2.0/1000),width=dx,facecolor='grey',label=r'$\Phi_{\nu_\mu} \times \sigma^{\nu_e}$')
print np.sum(e*2.0/1000)
#######################
# STYLE ARGUMENTS FOR PLOTTING WITH "STEP"
kwargs={'linewidth':1.5,'where':'post'}
ax.step(bin_c-dx/2.0, y, label=r'$\nu_4 \to \nu_e Z^\prime$ (%.1f events)'%(N),**kwargs)
ax.step(bin_c-dx/2.0, yCASCADE, label=r'$\nu_4 \to \nu_e \nu_e \overline{\nu_e}$ (%.1f events)'%(NCASCADE),**kwargs)
ax.step(bin_c-dx/2.0, yosc, label=r'SBL osc (%.1f events)'%(N_osc), dashes=(3,3), **kwargs)



###################
# MINIBOONE DATA

X =  miniboone.Enu_binc
BINW =  miniboone.binw_enu
DATA =  miniboone.data_MB_enu_nue/miniboone.binw_enu/UNITS
ERRORLOW =  -(miniboone.data_MB_enu_nue_errorlow-miniboone.data_MB_enu_nue)/miniboone.binw_enu/UNITS
ERRORUP = (miniboone.data_MB_enu_nue_errorup-miniboone.data_MB_enu_nue)/miniboone.binw_enu/UNITS

ax.errorbar(X, DATA, yerr= np.array([ERRORLOW,ERRORUP]), xerr = BINW/2.0, \
												marker="o", markeredgewidth=0.5, capsize=2.0,markerfacecolor="white",\
												markeredgecolor="black",ms=3, color='black', lw = 0.0, elinewidth=1.0, zorder=100)




##############
# STYLE
ax.legend(loc='upper right',frameon=False,ncol=1)
ax.set_title(r'$m_h = %.0f$ keV,\, $m_{Z^\prime} = %.0f$ keV, \, $|U_{\mu h}| = %.3f$'%(params.m4*1e6,params.mzprime*1e6,params.Umu4), fontsize=fsize)

ax.set_xlim(np.min(bin_c-dx/2.0),np.max(bin_c-dx/2.0))
# ax.set_ylim(ax.get_ylim()[0], ax.get_ylim()[1]*1.2)
ax.set_ylim(-200, 1900)

ax.set_xlabel(r'$E_\nu/$GeV')
ax.set_ylabel(r'Excess Events/GeV')
fig.savefig('plots/Enu_miniboone_MH_%.0f_MZ_%.0f.pdf'%(params.m4*1e6,params.mzprime*1e6))
# plt.show()

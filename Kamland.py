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

EXP_FLAG = const.KAMLAND



################################################################
# COMPUTING THE EVENT RATE INTEGRALS
################################################################

if EXP_FLAG == const.MINIBOONE:
	Nucleons_per_target = 14.0
	P_per_target = 8.0
	TARGETS = (818e6) * const.NAvo
	POTS = 12.84e20
	A_NUMBER = 12.0
	Enu_BEG_OF_SPECTRUM = 0.0
	Enu_END_OF_SPECTRUM = 2.0
	L =  0.541
	norm = POTS*TARGETS /1e55
	exp = exps.miniboone_data()

if EXP_FLAG == const.BOREXINO:
	Enu_BEG_OF_SPECTRUM = 0.0
	Enu_END_OF_SPECTRUM = 16.8
	N_PROTONS = 1.32e31 
	avg_efficiency = 0.850
	exposure = 2485 * 60*60*24 # seconds
	norm = N_PROTONS*avg_efficiency*exposure/1e55
	exp = exps.borexino_data()

if EXP_FLAG == const.KAMLAND:
	Enu_BEG_OF_SPECTRUM = 0.0
	Enu_END_OF_SPECTRUM = 16.8
	EXPOSURE=2343.0*24*60*60
	year=365*24*60*60
	fid_cut=(6.0/6.50)**3
	efficiency=0.92
	mass=1e9 # grams
	NA=6.022e23
	norm=EXPOSURE*fid_cut*efficiency*mass*NA/1e55
	exp = exps.kamland_data()

############
# NUMU FLUX
fluxfile = "fluxes/b8spectrum.txt"
flux = fluxes.get_exp_flux(fluxfile)

############
# NUE/BAR XS
xsfile="xsecs/IBD_160106169/TCS_CC_anue_p_1026_SV.txt"
xsec = lambda x : np.zeros(np.size(x)) 
xsecbar = xsecs.get_IBD(xsfile)
############
# DECAY MODEL PARAMETERS
params = model.vector_model_params()
params.gx		= 1.0
params.Ue4		= 0.1
params.Umu4		= np.sqrt(1e-2)
params.UD4		= np.sqrt(1.0-params.Ue4*params.Ue4-params.Umu4*params.Umu4)
params.m4		= 300e-9 # GeV
params.mzprime  = 0.9*params.m4 # GeV


############
# EXPERIMENTAL DATA AND BINS
bins = exp.bin_e
dx = exp.bin_w
bin_c= exp.bin_c

# efficiencies
enu_eff= bins
eff= np.ones((np.size(dx)))


################################################################
# COMPUTING THE EVENT RATE INTEGRALS
################################################################
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
axes_form  = [0.15,0.15,0.82,0.76]
fig = plt.figure()
ax = fig.add_axes(axes_form)


##########
# PLOT UNITS
UNITS = 1 

##############
# NORMALIZE THE DISTRIBUTIONS
yCASCADE= dNCASCADE

#######################
# STYLE ARGUMENTS FOR PLOTTING WITH "STEP"
kwargs={'linewidth':1.0,'where':'post','color':'darkorange'}

######################
# Montecarlo 
MCatm = exp.MCatm
MCreactor = exp.MCreactor
MCreactor_spall = exp.MCreactor_spall
MClimit = exp.MClimit


Elin = np.linspace(7.5+0.8, 16.8)
# ax.step(bin_c-dx/2.0, MCtot+yCASCADE, label=r'$\nu_4 \to \nu_e \nu_e \overline{\nu_e}$ (%.1f events)'%(np.sum(dNCASCADE)),**kwargs)
ax.bar(bin_c, yCASCADE, bottom=MCatm, width=dx, lw=0.5, edgecolor='black', facecolor='None',hatch='//////////', label=r'$\nu_4 \to \nu_e \nu_e \overline{\nu_e}$ (%.1f events)'%(np.sum(dNCASCADE)))

ax.bar(Elin,MCatm, lw=0.2,facecolor='orange',edgecolor='orange', width=dx,alpha=0.7, label=r'reactors')
ax.bar(Elin,MCreactor, lw=0.2,facecolor='indigo',edgecolor='indigo', width=dx,alpha=0.7, label=r'spallation')
ax.bar(Elin,MCreactor_spall, lw=0.2,facecolor='dodgerblue',edgecolor='dodgerblue', width=dx,alpha=0.7, label=r'atm+$n$+acc')
ax.plot(Elin,MClimit, lw=0.2,color='dodgerblue', label=r'90\% limit')

# ax.step(bin_c-dx/2.0, yCASCADE, ls='--', label=r'$\nu_4 \to \nu_e \nu_e \overline{\nu_e}$ (%.1f events)'%(np.sum(dNCASCADE)),**kwargs)


###################
# DATA
DATA =  exp.data
ERRORLOW =  np.sqrt(DATA)
ERRORUP = np.sqrt(DATA)

ax.errorbar(bin_c, DATA, yerr= np.array([ERRORLOW,ERRORUP]), xerr = dx/2.0, \
												marker="o", markeredgewidth=0.5, capsize=1.0,markerfacecolor="black",\
												markeredgecolor="black", ms=2, color='black', lw = 0.0, elinewidth=0.8, zorder=100,label=r'data')


##############
# STYLE
ax.legend(loc='upper right',frameon=False,ncol=1)
ax.set_title(r'$m_h = %.0f$ eV,\, $m_{Z^\prime}/m_h = %.2f$, \, $|U_{e h}|^2 = %.3f$'%(params.m4*1e9,params.mzprime/params.m4,params.Umu4**2), fontsize=fsize)

# ax.set_yscale('log')
ax.set_xlim(7.5+0.8,16.8)
# ax.set_ylim(ax.get_ylim()[0], ax.get_ylim()[1]*1.2)
ax.set_ylim(0,)

ax.text(10,15,r'KamLAND',fontsize=14)

ax.set_xlabel(r'$E_\nu/$MeV')
ax.set_ylabel(r'Events/MeV')
fig.savefig('plots/Enu_kamland_MH_%.0f_MZ_%.0f.pdf'%(params.m4*1e9,params.mzprime*1e9))
# fig.savefig('plots/test.pdf')
# plt.show()
print NCASCADE
print np.sum(dNCASCADE)
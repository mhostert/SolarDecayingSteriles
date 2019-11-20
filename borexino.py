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

EXP_FLAG = const.BOREXINO

if EXP_FLAG == const.BOREXINO:
	Enu_BEG_OF_SPECTRUM = 0.0
	Enu_END_OF_SPECTRUM = 20
	N_PROTONS = 1.32e31 
	avg_efficiency = 0.850
	exposure = 2485*60*60*24 # seconds
	norm = N_PROTONS*avg_efficiency*exposure
	exp = exps.borexino_data()

############
# NUMU FLUX
fluxfile = "fluxes/b8spectrum.txt"
flux = fluxes.get_exp_flux(fluxfile)

############
# NUE/BAR XS
xsfile="xsecs/IBD_160106169/TCS_CC_anue_p_1026_SV.txt"
xsec = lambda x : np.zeros(np.size(x)) 
xsecbar = xsecs.get_IBD(xsfile)
# xsecbar = lambda x : 1e-42*x
############
# DECAY MODEL PARAMETERS
params = model.decay_model_params(const.SCALAR)
params.gx		= 1.0
params.Ue4		= np.sqrt(4e-3)
params.Umu4		= np.sqrt(1e-3)*0
params.Utau4	= np.sqrt(1e-3)*0
params.UD4		= np.sqrt(1.0-params.Ue4*params.Ue4-params.Umu4*params.Umu4)
params.m4		= 300e-9 # GeV
params.mBOSON  = 0.9*params.m4 # GeV


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
# HNL + BOSON DECAYS
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



######################
# Montecarlo 
MCatm = exp.MCatm
MCreactor = exp.MCreactor
MCgeo = exp.MCgeo

MCtot = MCatm+MCreactor+MCgeo


# ax.step(bin_c-dx/2.0, MCtot+dNCASCADE, label=r'$\nu_4 \to \nu_e \nu_e \overline{\nu_e}$ (%.1f events)'%(np.sum(dNCASCADE)),**kwargs)
ax.bar(bin_c, dNCASCADE, bottom=MCtot, width=dx, lw=0.5, edgecolor='black', facecolor='None',hatch='//////////', label=r'$\nu_4 \to \nu_e \nu_e \overline{\nu_e}$ (%.1f events)'%(np.sum(dNCASCADE)))

ax.bar(bin_c,MCtot, lw=0.2,facecolor='orange',edgecolor='orange', width=dx,alpha=0.7, label=r'geoneutrinos')
ax.bar(bin_c,MCatm+MCreactor, lw=0.2,facecolor='dodgerblue',edgecolor='dodgerblue', width=dx,alpha=0.7, label=r'reactors')
ax.bar(bin_c,MCatm, lw=0.2,facecolor='indigo',edgecolor='indigo', width=dx,alpha=0.7, label=r'atmospheric')



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
if params.model == const.VECTOR:
	boson_string = r'$m_{Z^\prime}$'
	boson_file = 'vector'
elif params.model == const.SCALAR:
	boson_string = r'$m_\phi$'
	boson_file = 'scalar'

ax.legend(loc='upper right',frameon=False,ncol=1)
ax.set_title(r'$m_4 = %.0f$ eV,\, '%(params.m4*1e9)+boson_string+r'$/m_4 = %.2f$, \, $|U_{e 4}|^2 = %.3f$'%(params.mBOSON/params.m4,params.Ue4**2), fontsize=fsize)

ax.annotate(r'Borexino',xy=(0.45,0.35),xycoords='axes fraction',fontsize=14)

ax.set_xlabel(r'$E_\nu/$MeV')
ax.set_ylabel(r'Events/MeV')
fig.savefig('plots/'+boson_file+'_borexino_MN_%.0f_MB_%.0f.pdf'%(params.m4*1e9,params.mBOSON*1e9))

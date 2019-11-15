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




################################################################
# COMPUTING THE EVENT RATE INTEGRALS
################################################################

Enu_BEG_OF_SPECTRUM = 0.0
Enu_END_OF_SPECTRUM = 16.8
EXPOSURE=2343.0*24*60*60
year=365*24*60*60
fid_cut=(6.0/6.50)**3
efficiency=0.92
mass=1e9 # grams
NA=6.022e23
norm=EXPOSURE*fid_cut*efficiency*mass*NA
exp = exps.kamland_data()

print EXPOSURE*fid_cut/year*efficiency, ' ktons.year'

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
params.Ue4		= np.sqrt(2e-3)
params.Umu4		= np.sqrt(1e-3)
params.Utau4	= np.sqrt(1e-3)*0
params.UD4		= np.sqrt(1.0-params.Ue4*params.Ue4-params.Umu4*params.Umu4)
params.m4		= 300e-9 # GeV
params.mBOSON  = 0.9*params.m4 # GeV


############
# EXPERIMENTAL DATA AND BINS
# bins = exp.bin_e
# dx = exp.bin_w
# bin_c= exp.bin_c
bins = np.linspace(np.min(exp.bin_e)-0.2,17.3,60)
dx = bins[1]-bins[0]
bin_c = bins[:-1] + dx/2.0
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


#######################
# STYLE ARGUMENTS FOR PLOTTING WITH "STEP"
kwargs={'linewidth':1.0,'where':'post','color':'darkorange'}

######################
# Montecarlo 
Elin = np.linspace(np.min(exp.bin_e),17.3,100)
dxlin = Elin[1]-Elin[0]
MCall = exp.MCall(Elin)
MCreactor = exp.MCreactor(Elin)
MCreactor_spall = exp.MCreactor_spall(Elin)
MClimit = exp.MClimit(Elin)


# MCall_binned = np.zeros(np.size(bins)-1)
# for i in xrange(np.size(bins)-1):
	# MCall_binned[i] = np.sum( MCall[ (Elin<bins[i+1]) & (Elin>bins[i]) ]*dxlin ) 
MCall_binned = exp.MCall(bin_c)


# ax.step(bin_c-dx/2.0, yCASCADE, ls='--', label=r'$\nu_4 \to \nu_e \nu_e \overline{\nu_e}$ (%.1f events)'%(np.sum(dNCASCADE)),**kwargs)


# ax.bar(bin_c, dNCASCADE, bottom=MCall_binned, width=dx, lw=0.5, edgecolor='black', facecolor='None',hatch='//////////', label=r'$\nu_4 \to \nu_e \nu_e \overline{\nu_e}$ (%.1f events)'%(np.sum(dNCASCADE)))
ax.fill_between(bin_c, dNCASCADE+MCall_binned, MCall_binned, lw=0.5, facecolor='None', edgecolor='black', hatch='//////', label=r'$\nu_4 \to \nu_e \nu_e \overline{\nu_e}$ (%.1f events)'%(np.sum(dNCASCADE)))
ax.fill_between(Elin,MCall, MCreactor,  lw=0.2,color='orange',alpha=0.7, label=r'reactors')
ax.fill_between(Elin,MCreactor, MCreactor_spall,  lw=0.2,color='dodgerblue',alpha=0.7, label=r'spallation')
ax.fill_between(Elin,MCreactor_spall, 0*MCreactor_spall,  lw=0.2,color='indigo',alpha=0.7, label=r'atm+$n$+acc')
ax.plot(Elin,MClimit,lw=0.8, color='crimson', dashes=(5,1),label=r'90\% limit')

###################
# DATA
DATA =  exp.data
ERRORLOW =  np.sqrt(DATA)
ERRORUP = np.sqrt(DATA)

ax.errorbar(exp.bin_c, DATA, yerr= np.array([ERRORLOW,ERRORUP]), xerr = exp.bin_w/2.0, \
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
ax.set_title(r'$m_4 = %.0f$ eV,\, '%(params.m4*1e9)+boson_string+r'$/m_h = %.2f$, \, $|U_{e 4}|^2 = %.3f$'%(params.mBOSON/params.m4,params.Ue4**2), fontsize=fsize)

ax.annotate(r'KamLAND',xy=(0.45,0.35),xycoords='axes fraction',fontsize=14)
ax.set_xlim(7.5+0.81,17.31)
ax.set_ylim(0,)

ax.set_xlabel(r'$E_\nu/$MeV')
ax.set_ylabel(r'Events/MeV')
fig.savefig('plots/'+boson_file+'_kamland_MN_%.0f_MB_%.0f.pdf'%(params.m4*1e9,params.mBOSON*1e9))

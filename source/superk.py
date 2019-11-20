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

def plot(params,fluxfile,xsfile):
	################################################################
	# SETUP
	################################################################
	exp = exps.superk_data()
	smearing_function=exps.superk_Esmear

	############
	# NUMU FLUX
	flux = fluxes.get_exp_flux(fluxfile)

	############
	# NUE/BAR XS
	xsec = lambda x : np.zeros(np.size(x)) 
	xsecbar = xsecs.get_IBD(xsfile)
	# xsecbar = lambda x : 1e-42*x
	
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
	NCASCADE, dNCASCADE = rates.RATES_dN_HNL_CASCADE_NU_NUBAR(\
												flux=flux,\
												xsec=xsec,\
												xsecbar=xsecbar,\
												dim=3,\
												enumin=const.Enu_BEG_OF_SPECTRUM,\
												enumax=const.Enu_END_OF_SPECTRUM,\
												params=params,\
												bins=bins,\
												PRINT=True,\
												enu_eff=enu_eff,\
												eff=eff,
												smearing_function=smearing_function)

	NCASCADE*=exp.norm
	dNCASCADE*=exp.norm


	################################################################
	# PLOTTING THE EVENT RATES 
	################################################################
	fsize=11
	rc('text', usetex=True)
	rcparams={'axes.labelsize':fsize,'xtick.labelsize':fsize,'ytick.labelsize':fsize,\
					'figure.figsize':(1.2*3.7,1.4*2.3617)	}
	rc('font',**{'family':'serif', 'serif': ['computer modern roman']})
	matplotlib.rcParams['hatch.linewidth'] = 0.5  # previous pdf hatch linewidth
	rcParams.update(rcparams)
	axes_form  = [0.15,0.15,0.82,0.76]
	fig = plt.figure()
	ax = fig.add_axes(axes_form)

	RAST=True
	ax.set_rasterized(True)

	######################
	# Montecarlo 
	MCall = exp.MCall
	MCaccidental = exp.MCaccidental
	MCreactor = exp.MCreactor
	MCreactorLi = exp.MCreactorLi


	# ax.step(bin_c-dx/2.0, MCtot+dNCASCADE, label=r'$\nu_4 \to \nu_e \nu_e \overline{\nu_e}$ (%.1f events)'%(np.sum(dNCASCADE)),**kwargs)
	ax.bar(bin_c, dNCASCADE, bottom=MCall, width=dx, lw=0, edgecolor='None', facecolor='grey', label=r'$\nu_4 \to \nu_e \nu_e \overline{\nu_e}$ (%.1f events)'%(np.sum(dNCASCADE)), rasterized=False)

	ax.bar(bin_c, MCall-MCreactor, bottom=MCreactor, lw=0.5, edgecolor='#FFD500', width=dx, label=r'reactors', rasterized=RAST,facecolor='None', hatch='xxxxxxxxxx')
	ax.bar(bin_c,MCreactor-MCreactorLi, bottom=MCreactorLi, lw=0.5, edgecolor='#5BD355', width=dx, label=r'spall ($^9$Li)', rasterized=RAST,facecolor='None', hatch='xxxxxxxxxx')
	ax.bar(bin_c,MCreactorLi, lw=0.5, edgecolor='#5955D8', width=dx, label=r'atm+NC+acc', rasterized=RAST,facecolor='None', hatch='xxxxxxxxxx')
	# ax.bar(bin_c,MCaccidental, facecolor='lightskyblue',edgecolor='lightskyblue', width=dx, label=r'accidental', rasterized=False)
	ax.bar(bin_c, dNCASCADE, bottom=MCall, width=dx, lw=0.5, edgecolor='black', facecolor='None', rasterized=False)
	ax.bar(bin_c, dNCASCADE+MCall, width=dx, lw=0.6, facecolor='None', edgecolor='black', rasterized=False)

	###################
	# DATA
	DATA =  exp.data
	ERRORLOW =  np.sqrt(DATA)
	ERRORUP = np.sqrt(DATA)

	ax.errorbar(bin_c, DATA, yerr= np.array([ERRORLOW,ERRORUP]), xerr = dx/2.0, \
													marker="o", markeredgewidth=0.5, capsize=1.0,markerfacecolor="white",\
													markeredgecolor="black", ms=3, color='black', lw = 0.0, elinewidth=0.8, zorder=100,label=r'data')

	##############
	# STYLE
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
	ax.legend(loc='upper right',frameon=False,ncol=1,markerfirst=False)
	ax.set_title(r'$m_4 = %.0f$ eV,\, '%(params.m4*1e9)+boson_string+r'$/m_4 = %.2f$, \, $|U_{e 4}|^2 = \,$'%(params.mBOSON/params.m4)+UEQSR, fontsize=fsize)

	ax.annotate(r'SK-IV',xy=(0.8,0.4),xycoords='axes fraction',fontsize=14)
	ax.set_xlim(9.3,17.3)
	ax.set_ylim(0,)
	_,yu = ax.get_ylim()
	ax.set_ylim(0,16)

	ax.set_xlabel(r'$E_\nu/$MeV')
	ax.set_ylabel(r'Events/MeV')
	fig_name = 'plots/'+boson_file+'_SK-IV_MN_%.0f_MB_%.0f.pdf'%(params.m4*1e9,params.mBOSON*1e9)
	fig.savefig(fig_name, dpi=500)
	return fig, fig_name, ax
import numpy as np
import matplotlib 
import matplotlib.pyplot as plt
from matplotlib import rc, rcParams
from matplotlib.pyplot import *
from matplotlib.legend_handler import HandlerLine2D

import vegas
import gvar as gv

from source import *

def plot_IBD_spectrum(params,fluxfile,xsfile, rasterized=True):
	################################################################
	# SETUP
	################################################################
	exp = exps.borexino_data()
	smearing_function=exps.borexino_Esmear

	############
	# NUMU FLUX
	flux = fluxes.get_neutrino_flux(fluxfile)

	############
	# NUE/BAR XS
	xsec = lambda x : np.zeros(np.size(x)) 
	xsecbar = xsecs.get_IBD(xsfile)


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
												PRINT=False,\
												enu_eff=enu_eff,\
												eff=eff,
												smearing_function=lambda x: x)

	NCASCADE*=exp.norm
	dNCASCADE*=exp.norm


	################################################################
	# PLOTTING THE EVENT RATES 
	################################################################
	fsize=11
	rc('text', usetex=True)
	rcparams={'axes.labelsize':fsize,'xtick.labelsize':fsize,'ytick.labelsize':fsize,\
					'figure.figsize':(1.2*3.7,2.3617)	}
	rc('font',**{'family':'serif', 'serif': ['computer modern roman']})
	matplotlib.rcParams['hatch.linewidth'] = 0.5  # previous pdf hatch linewidth
	matplotlib.rcParams['hatch.color'] = 'firebrick'  # previous pdf hatch linewidth
	rcParams.update(rcparams)
	axes_form  = [0.15,0.185,0.82,0.7]
	fig = plt.figure()
	ax = fig.add_axes(axes_form)
	ax.patch.set_alpha(0.0)
	ax.set_rasterization_zorder(0)

	######################
	# Montecarlo 
	MCatm = exp.MCatm
	MCreactor = exp.MCreactor
	MCgeo = exp.MCgeo
	MCtot = exp.MCall

	# ax.step(bin_c-dx/2.0, MCtot+dNCASCADE, label=r'$\nu_4 \to \nu_e \nu_e \overline{\nu_e}$ (%.1f events)'%(np.sum(dNCASCADE)),**kwargs)
	ax.bar(bin_c, dNCASCADE, bottom=MCtot, width=dx, lw=0, facecolor='grey', edgecolor='None', label=r'$\nu_4 \to \nu_e \nu_e \overline{\nu_e}$ (%.1f events)'%(np.sum(dNCASCADE)),
				zorder=1, rasterized=rasterized)

	ax.bar(bin_c,MCgeo,bottom=MCreactor+MCatm, lw=0.75,facecolor='None',edgecolor='dodgerblue',width=dx, label=r'geoneutrinos', hatch='xxxxxxxxxx',
				zorder=1, rasterized=rasterized)
	ax.bar(bin_c,MCreactor,bottom=MCatm, lw=0.75,facecolor='None',edgecolor='#FFD500', width=dx, label=r'reactors', hatch='xxxxxxxxxx',
				zorder=1, rasterized=rasterized)
	ax.bar(bin_c,MCatm, lw=0.75,facecolor='None',edgecolor='#5955D8', width=dx, label=r'atm', hatch='xxxxxxxxxx',
				zorder=1, rasterized=rasterized)
	ax.bar(bin_c, dNCASCADE+MCtot, width=dx, lw=0.75, edgecolor='black', facecolor='None',
				zorder=1, rasterized=rasterized)

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
	ax.legend(loc='upper right',frameon=False,ncol=1,markerfirst=False,fontsize=9)
	ax.set_title(r'$m_4 = %.0f$ eV,\, '%(params.m4*1e9)+boson_string+r'$/m_4 = %.2f$, \, $|U_{e 4}|^2 = \,$'%(params.mBOSON/params.m4)+UEQSR, fontsize=0.9*fsize)

	ax.annotate(r'Borexino',xy=(0.7,0.2),xycoords='axes fraction',fontsize=14)
	ax.set_xlim(1.8,15.8)
	ax.set_ylim(0,48)
	ax.set_xlabel(r'$E_\nu/$MeV')
	ax.set_ylabel(r'Events/MeV')
	fig_name = 'plots/IBD_spectra/'+boson_file+'_borexino_MN_%.0f_MB_%.0f.pdf'%(params.m4*1e9,params.mBOSON*1e9)
	fig.savefig(fig_name,dpi=300)
	fig.savefig(fig_name.replace("pdf","png"),dpi=300)
	plt.close()
	return fig, fig_name, ax
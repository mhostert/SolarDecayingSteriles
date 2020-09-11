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

def plot(params,fluxfile,xsfile,style='binned'):
	################################################################
	# SETUP
	# style = 'smooth'
	exp = exps.kamland_data()
	smearing_function=exps.kamland_Esmear
	Enu_BEG_OF_SPECTRUM = 0.0
	Enu_END_OF_SPECTRUM = 17.0

	############
	# NUMU FLUX
	flux = fluxes.get_exp_flux(fluxfile)

	############
	# NUE/BAR XS
	xsec = lambda x : np.zeros(np.size(x)) 
	xsecbar = xsecs.get_IBD(xsfile)

	##############
	# Plot style -- use differential or bins
	if style=='binned':
		bins = exp.bin_e
		dx = exp.bin_w
		bin_c= exp.bin_c
	elif style=='smooth':
		bins = np.linspace(np.min(exp.bin_e)-0.2,17.3,60)
		dx = bins[1]-bins[0]
		bin_c = bins[:-1] + dx/2.0
	enu_eff= bins
	eff= np.ones((np.size(dx)))

	################################################################
	# COMPUTING THE EVENT RATE INTEGRALS
	################################################################
	# # HNL + BOSON DECAYS
	# NCASCADE, dNCASCADE = rates.RATES_dN_HNL_CASCADE_NU_NUBAR(\
	# 											flux=flux,\
	# 											xsec=xsec,\
	# 											xsecbar=xsecbar,\
	# 											dim=3,\
	# 											enumin=Enu_BEG_OF_SPECTRUM,\
	# 											enumax=Enu_END_OF_SPECTRUM,\
	# 											params=params,\
	# 											bins=bins,\
	# 											PRINT=False,\
	# 											enu_eff=enu_eff,\
	# 											eff=eff,
	# 											smearing_function=smearing_function)
	# NCASCADE*=exp.norm
	# dNCASCADE*=exp.norm/dx
	KAM = exps.kamland_data()
	bK, npK, backK, dK = rates.fill_bins(KAM,params,fluxfile,endpoint=16.3)
	dbK = (bK[1:]-bK[:-1])
	bKc = bK[:-1]+dbK/2.0

	print(bK, npK)
	# elin = np.linspace(8.3,17,1000)
	# print np.sum(flux(elin)/const.B8FLUX)*(elin[1]-elin[0])*315.13
	# print np.sum(flux(elin)*xsecbar(elin)*315.13)*(elin[1]-elin[0])*exp.norm/const.B8FLUX

	################################################################
	# PLOTTING THE EVENT RATES 
	################################################################
	fsize=11
	rc('text', usetex=True)
	rcparams={'axes.labelsize':fsize,'xtick.labelsize':fsize,'ytick.labelsize':fsize,\
					'figure.figsize':(1.2*3.7,2.3617)	}
	rc('font',**{'family':'serif', 'serif': ['computer modern roman']})
	matplotlib.rcParams['hatch.linewidth'] = 0.5  # previous pdf hatch linewidth
	rcParams.update(rcparams)
	axes_form  = [0.15,0.175,0.82,0.7]
	fig = plt.figure()
	ax = fig.add_axes(axes_form)

	RAST = True
	ax.set_rasterized(True)

	######################
	# Montecarlo 
	Elin = np.linspace(np.min(exp.bin_e),17.3,1000)
	dxlin = Elin[1]-Elin[0]

	MCall = exp.MCall(Elin)
	MCreactor = exp.MCreactor(Elin)
	MCreactor_spall = exp.MCreactor_spall(Elin)
	MClimit = exp.MClimit(Elin)

	if style=='binned':
		MCall_binned = np.zeros(np.size(bins)-1)
		MCreactor_binned = np.zeros(np.size(bins)-1)
		MCreactor_spall_binned = np.zeros(np.size(bins)-1)
		MClimit_binned = np.zeros(np.size(bins)-1)
		for i in range(0,np.size(bins)-1):
			MCall_binned[i] = np.sum( MCall[ (Elin<bins[i+1]) & (Elin>bins[i]) ]*dxlin ) 
			MCreactor_binned[i] = np.sum( MCreactor[ (Elin<bins[i+1]) & (Elin>bins[i]) ]*dxlin ) 
			MCreactor_spall_binned[i] = np.sum( MCreactor_spall[ (Elin<bins[i+1]) & (Elin>bins[i]) ]*dxlin ) 
			MClimit_binned[i] = np.sum( MClimit[ (Elin<bins[i+1]) & (Elin>bins[i]) ]*dxlin ) 

		# ax.bar(bKc, npK, bottom=backK, width=dbK, lw=0, facecolor='grey', edgecolor='None', label=r'$\nu_4 \to \nu_e \nu_e \overline{\nu_e}$ (%.1f events)'%(np.sum(npK)), rasterized=False)
		ax.bar(bKc, npK, bottom=backK, width=dbK, lw=0, facecolor='grey', edgecolor='None', label=r'$\nu_4 \to \nu_e \nu_e \overline{\nu_e}$ (%.1f events)'%(np.sum(npK)))

		# ax.step(bin_c-dx/2.0,MClimit_binned,lw=1, where='post',dashes=(5,1), color='crimson', label=r'90\% limit', rasterized=False)
		ax.bar(bin_c,MCall_binned-MCreactor_binned, bottom=MCreactor_binned, width=dx,  lw=0.5,edgecolor='#FFD500', label=r'reactors', rasterized=RAST,facecolor='None',hatch='xxxxxxxxxx')
		ax.bar(bin_c,MCreactor_binned-MCreactor_spall_binned, bottom=MCreactor_spall_binned, width=dx,  lw=0.5,edgecolor='#5BD355', label=r'spallation', rasterized=RAST,facecolor='None',hatch='xxxxxxxxxx')
		ax.bar(bin_c,MCreactor_spall_binned, bottom=0*MCreactor_spall_binned, width=dx,  lw=0.5,edgecolor='#5955D8', label=r'atm+$n$+acc', rasterized=RAST,facecolor='None',hatch='xxxxxxxxxx')

		ax.bar(bKc, npK+backK, width=dbK, lw=0.6, facecolor='None', edgecolor='black', rasterized=False)

	elif style=='smooth':
		MCall_binned = exp.MCall(bin_c)
		ax.fill_between(bin_c, dNCASCADE+MCall_binned, MCall_binned, lw=0.5, facecolor='None', edgecolor='black', hatch='////', label=r'$\nu_4 \to \nu_e \nu_e \overline{\nu_e}$ (%.1f events)'%(np.sum(dNCASCADE*dx)))
		ax.fill_between(Elin,MCall, MCreactor,  lw=0.2,color='dodgerblue',alpha=0.7, label=r'reactors')
		ax.fill_between(Elin,MCreactor, MCreactor_spall,  lw=0.2,color='pink',alpha=0.7, label=r'spallation')
		ax.fill_between(Elin,MCreactor_spall, 0*MCreactor_spall,  lw=0.2,color='indigo',alpha=0.7, label=r'atm+$n$+acc')
		ax.plot(Elin,MClimit,lw=0.8, color='crimson', dashes=(5,1),label=r'90\% limit')

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
	ax.set_title(r'$m_4 = %.0f$ eV,\, '%(params.m4*1e9)+boson_string+r'$/m_4 = %.2f$, \, $|U_{e 4}|^2 = \,$'%(params.mBOSON/params.m4)+UEQSR, fontsize=9)

	ax.annotate(r'KamLAND',xy=(0.7,0.2),xycoords='axes fraction',fontsize=14)
	ax.set_xlim(7.5+0.81,17.31)
	ax.set_ylim(0,)

	_,yu = ax.get_ylim()
	if yu < 16:
		ax.set_ylim(0,20)

	ax.set_xlabel(r'$E_\nu/$MeV')
	ax.set_ylabel(r'Events/MeV')
	fig_name='plots/'+style+'_'+boson_file+'_kamland_MN_%.0f_MB_%.0f.pdf'%(params.m4*1e9,params.mBOSON*1e9)
	fig.savefig(fig_name,dpi=500)
	return fig, fig_name, ax

import numpy as np

import vegas
import gvar as gv

from source import *
from source.plot_tools import *

################################################################
def kamland_21(params, fluxfile, xsfile, style='binned', rasterized=True):
	# fill bins
	exp = exps.kamland21_data()
	ENDPOINT = exp.fit_endpoint
	STARTPOINT = exp.fit_startpoint
	bK, npK, backK, dK = rates.fill_bins(exp,params,fluxfile,endpoint=18.3)
	dbK = (bK[1:]-bK[:-1])
	bKc = bK[:-1]+dbK/2.0

	##############
	# Plot style -- use differential or bins
	bins = exp.bin_e
	dx = exp.bin_w
	bin_c= exp.bin_c

	######################
	# Montecarlo 
	Elin = np.linspace(np.min(exp.bin_e),ENDPOINT,1000)
	dxlin = Elin[1]-Elin[0]

	MCall = exp.MCall(Elin)

	################################################################
	# PLOTTING THE EVENT RATES 
	fig, ax = get_std_fig()
		
	nbins=np.size(bins)-1
	MCall_binned = np.zeros(nbins)
	
	for i in range(0,nbins):
		in_this_bin = (Elin<bins[i+1]) & (Elin>bins[i])
		MCall_binned[i] = np.sum( MCall[  in_this_bin	]*dxlin ) 
		
	ax.bar(bKc, npK, bottom=backK, width=dbK, lw=0, facecolor='grey', edgecolor='None', label=r'$\nu_4 \to \nu_e \nu_e \overline{\nu_e}$ (%.1f events)'%(np.sum(npK)),
		zorder=1, rasterized=rasterized)

	ax.bar(bin_c,MCall_binned, width=dx,  lw=0.5,edgecolor='#FFD500', label=r'all bkgs', facecolor='None', hatch=my_hatch,
		zorder=1, rasterized=rasterized)
	ax.bar(bKc, npK+backK, width=dbK, lw=0.6, facecolor='None', edgecolor='black',
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
	# what boson
	if params.model == const.VECTOR:
		boson_string = r'$m_{Z^\prime}$'
		boson_file = 'vector'
	elif params.model == const.SCALAR:
		boson_string = r'$m_\phi$'
		boson_file = 'scalar'


	UEQSR = to_scientific_notation(params.Ue4**2)
	ax.legend(loc='upper right',frameon=False,ncol=1,markerfirst=False,fontsize=9)
	ax.set_title(r'$m_4 = %.0f$ eV,\, '%(params.m4*1e9)+boson_string+r'$/m_4 = %.2f$, \, $|U_{e 4}|^2 = \,$'%(params.mBOSON/params.m4)+UEQSR, fontsize=9)

	ax.annotate(r'KamLAND 2021',xy=(0.63,0.25),xycoords='axes fraction',fontsize=14)
	ax.set_xlim(STARTPOINT,ENDPOINT)
	ax.set_ylim(0,)

	_,yu = ax.get_ylim()
	if yu < 16:
		ax.set_ylim(0,20)

	ax.set_xlabel(r'$E_\nu/$MeV')
	ax.set_ylabel(r'Events/2 MeV')

	fig_name='plots/IBD_spectra/'+style+'_'+boson_file+'_kamland21_MN_%.0f_MB_%.0f.pdf'%(params.m4*1e9,params.mBOSON*1e9)
	save_figs(fig_name)
	plt.close()

	return fig, fig_name, ax

################################################################
def kamland(params, fluxfile, xsfile, style='binned', rasterized=True):
	# fill bins
	exp = exps.kamland_data()
	ENDPOINT = exp.fit_endpoint
	STARTPOINT = exp.fit_startpoint
	bK, npK, backK, dK = rates.fill_bins(exp,params,fluxfile,endpoint=16.3)
	dbK = (bK[1:]-bK[:-1])
	bKc = bK[:-1]+dbK/2.0

	##############
	# Plot style -- use differential or bins
	if style=='binned':
		bins = exp.bin_e
		dx = exp.bin_w
		bin_c= exp.bin_c
	elif style=='smooth':
		bins = np.linspace(np.min(exp.bin_e)-0.2,ENDPOINT,60)
		dx = bins[1]-bins[0]
		bin_c = bins[:-1] + dx/2.0

	######################
	# Montecarlo 
	Elin = np.linspace(np.min(exp.bin_e),ENDPOINT,1000)
	dxlin = Elin[1]-Elin[0]

	MCall = exp.MCall(Elin)
	MCreactor = exp.MCreactor(Elin)
	MCreactor_spall = exp.MCreactor_spall(Elin)
	MClimit = exp.MClimit(Elin)


	################################################################
	# PLOTTING THE EVENT RATES 
	fig, ax = get_std_fig()

	if style=='binned':
		
		nbins=np.size(bins)-1
		MCall_binned = np.zeros(nbins)
		MCreactor_binned = np.zeros(nbins)
		MCreactor_spall_binned = np.zeros(nbins)
		MClimit_binned = np.zeros(nbins)
		
		for i in range(0,nbins):
			in_this_bin = (Elin<bins[i+1]) & (Elin>bins[i])
			MCall_binned[i] = np.sum( MCall[  in_this_bin	]*dxlin ) 
			MCreactor_binned[i] = np.sum( MCreactor[ in_this_bin ]*dxlin ) 
			MCreactor_spall_binned[i] = np.sum( MCreactor_spall[ in_this_bin ]*dxlin ) 
			MClimit_binned[i] = np.sum( MClimit[ in_this_bin ]*dxlin ) 

		# ax.bar(bKc, npK, bottom=backK, width=dbK, lw=0, facecolor='grey', edgecolor='None', label=r'$\nu_4 \to \nu_e \nu_e \overline{\nu_e}$ (%.1f events)'%(np.sum(npK)), rasterized=False)
		ax.bar(bKc, npK, bottom=backK, width=dbK, lw=0, facecolor='grey', edgecolor='None', label=r'$\nu_4 \to \nu_e \nu_e \overline{\nu_e}$ (%.1f events)'%(np.sum(npK)),
			zorder=1, rasterized=rasterized)

		# ax.step(bin_c-dx/2.0,MClimit_binned,lw=1, where='post',dashes=(5,1), color='crimson', label=r'90\% limit', rasterized=False)
		ax.bar(bin_c,MCall_binned-MCreactor_binned, bottom=MCreactor_binned, width=dx,  lw=0.5,edgecolor='#FFD500', label=r'reactors', facecolor='None', hatch=my_hatch,
			zorder=1, rasterized=rasterized)
		ax.bar(bin_c,MCreactor_binned-MCreactor_spall_binned, bottom=MCreactor_spall_binned, width=dx,  lw=0.5,edgecolor='#5BD355', label=r'spallation', facecolor='None', hatch=my_hatch,
			zorder=1, rasterized=rasterized)
		ax.bar(bin_c,MCreactor_spall_binned, bottom=0*MCreactor_spall_binned, width=dx,  lw=0.5,edgecolor='#5955D8', label=r'atm+$n$+acc', facecolor='None', hatch=my_hatch,
			zorder=1, rasterized=rasterized)

		ax.bar(bKc, npK+backK, width=dbK, lw=0.6, facecolor='None', edgecolor='black',
			zorder=1, rasterized=rasterized)

	elif style=='smooth': 
		MCall_binned = exp.MCall(bin_c)
		ax.fill_between(bin_c, npK+MCall_binned, MCall_binned, lw=0.5, facecolor='None', edgecolor='black', hatch='////', label=r'$\nu_4 \to \nu_e \nu_e \overline{\nu_e}$ (%.1f events)'%(np.sum(dNCASCADE*dx)),
			zorder=1, rasterized=rasterized)
		ax.fill_between(Elin,MCall, MCreactor,  lw=0.2,color='dodgerblue',alpha=0.7, label=r'reactors',
			zorder=1, rasterized=rasterized)
		ax.fill_between(Elin,MCreactor, MCreactor_spall,  lw=0.2,color='pink',alpha=0.7, label=r'spallation',
			zorder=1, rasterized=rasterized)
		ax.fill_between(Elin,MCreactor_spall, 0*MCreactor_spall,  lw=0.2,color='indigo',alpha=0.7, label=r'atm+$n$+acc',
			zorder=1, rasterized=rasterized)
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
	# what boson
	if params.model == const.VECTOR:
		boson_string = r'$m_{Z^\prime}$'
		boson_file = 'vector'
	elif params.model == const.SCALAR:
		boson_string = r'$m_\phi$'
		boson_file = 'scalar'


	UEQSR = to_scientific_notation(params.Ue4**2)
	ax.legend(loc='upper right',frameon=False,ncol=1,markerfirst=False,fontsize=9)
	ax.set_title(r'$m_4 = %.0f$ eV,\, '%(params.m4*1e9)+boson_string+r'$/m_4 = %.2f$, \, $|U_{e 4}|^2 = \,$'%(params.mBOSON/params.m4)+UEQSR, fontsize=9)

	ax.annotate(r'KamLAND',xy=(0.7,0.2),xycoords='axes fraction',fontsize=14)
	ax.set_xlim(STARTPOINT,ENDPOINT)
	ax.set_ylim(0,)

	_,yu = ax.get_ylim()
	if yu < 16:
		ax.set_ylim(0,20)

	ax.set_xlabel(r'$E_\nu/$MeV')
	ax.set_ylabel(r'Events/MeV')

	fig_name='plots/IBD_spectra/'+style+'_'+boson_file+'_kamland_MN_%.0f_MB_%.0f.pdf'%(params.m4*1e9,params.mBOSON*1e9)
	save_figs(fig_name)
	plt.close()

	return fig, fig_name, ax



################################################################
def borexino(params, fluxfile ,xsfile, rasterized=True):
	# fill bins
	exp = exps.borexino_data()
	ENDPOINT = exp.fit_endpoint
	STARTPOINT = exp.fit_startpoint
	bins, NP_MC, back_MC, data = rates.fill_bins(exp,params,fluxfile,endpoint=15.8)
	ERRORLOW =  np.sqrt(data)
	ERRORUP = np.sqrt(data)

	mask = exp.bin_c < ENDPOINT
	bins = exp.bin_e[np.append(mask,True)]
	bin_c = exp.bin_c[mask]
	dx = exp.bin_w[mask]
	MCatm = exp.MCatm[mask]
	MCreactor = exp.MCreactor[mask]
	MCgeo = exp.MCgeo[mask]
	MCtot = exp.MCall[mask]


	################################################################
	# PLOTTING THE EVENT RATES 
	fig, ax = get_std_fig()

	# MC
	ax.bar(bin_c, NP_MC, bottom=MCtot, width=dx, lw=0, facecolor='grey', edgecolor='None', label=r'$\nu_4 \to \nu_e \nu_e \overline{\nu_e}$ (%.1f events)'%(np.sum(NP_MC)),
				zorder=1, rasterized=rasterized)
	ax.bar(bin_c,MCgeo,bottom=MCreactor+MCatm, lw=0.75,facecolor='None',edgecolor='dodgerblue',width=dx, label=r'geoneutrinos', hatch=my_hatch,
				zorder=1, rasterized=rasterized)
	ax.bar(bin_c,MCreactor,bottom=MCatm, lw=0.75,facecolor='None',edgecolor='#FFD500', width=dx, label=r'reactors', hatch=my_hatch,
				zorder=1, rasterized=rasterized)
	ax.bar(bin_c,MCatm, lw=0.75,facecolor='None',edgecolor='#5955D8', width=dx, label=r'atm', hatch=my_hatch,
				zorder=1, rasterized=rasterized)
	ax.bar(bin_c, NP_MC+MCtot, width=dx, lw=0.75, edgecolor='black', facecolor='None',
				zorder=1, rasterized=rasterized)
	# data
	ax.errorbar(bin_c, data, yerr= np.array([ERRORLOW,ERRORUP]), xerr = dx/2.0, \
													marker="o", markeredgewidth=0.5, capsize=1.0,markerfacecolor="white",\
													markeredgecolor="black", ms=3, color='black', lw = 0.0, elinewidth=0.8, zorder=100,label=r'data')

	##############
	# what boson
	if params.model == const.VECTOR:
		boson_string = r'$m_{Z^\prime}$'
		boson_file = 'vector'
	elif params.model == const.SCALAR:
		boson_string = r'$m_\phi$'
		boson_file = 'scalar'

	UEQSR = to_scientific_notation(params.Ue4**2)
	ax.legend(loc='upper right',frameon=False,ncol=1,markerfirst=False,fontsize=9)
	ax.set_title(r'$m_4 = %.0f$ eV,\, '%(params.m4*1e9)+boson_string+r'$/m_4 = %.2f$, \, $|U_{e 4}|^2 = \,$'%(params.mBOSON/params.m4)+UEQSR, fontsize=0.9*fsize)

	ax.annotate(r'Borexino',xy=(0.7,0.2),xycoords='axes fraction',fontsize=14)
	ax.set_xlim(STARTPOINT,ENDPOINT)
	ax.set_ylim(0,48)
	ax.set_xlabel(r'$E_\nu/$MeV')
	ax.set_ylabel(r'Events/MeV')

	fig_name = 'plots/IBD_spectra/'+boson_file+'_borexino_MN_%.0f_MB_%.0f.pdf'%(params.m4*1e9,params.mBOSON*1e9)
	save_figs(fig_name)
	plt.close()
	
	return fig, fig_name, ax
################################################################
# SUPER-K IV new -- using latest article
def superk(params,fluxfile,xsfile, rasterized=True):
	
	################################################################
	# fill bins
	exp = exps.superk_data()
	ENDPOINT = exp.fit_endpoint
	STARTPOINT = exp.fit_startpoint
	bins, NP_MC, back_MC, data = rates.fill_bins(exp,params,fluxfile,endpoint=ENDPOINT)
	ERRORLOW =  np.sqrt(data)
	ERRORUP = np.sqrt(data)

	mask = exp.bin_c < ENDPOINT
	bins = exp.bin_e[np.append(mask,True)]
	bin_c = exp.bin_c[mask]
	dx = exp.bin_w[mask]
	
	back_MC = back_MC/dx
	data  = data/dx

	NP_MC = NP_MC/dx
	MCall = exp.MCall[mask]/dx
	MCreactor = exp.MCreactor[mask]/dx
	MCacc = exp.MCacc[mask]/dx
	MCatm = exp.MCatm[mask]/dx

	################################################################
	# PLOTTING THE EVENT RATES 
	fig, ax = get_std_fig()

	# MC
	ax.bar(bin_c, NP_MC, bottom=MCall, width=dx, lw=0, facecolor='grey', edgecolor='None', label=r'$\nu_4 \to \nu_e \nu_e \overline{\nu_e}$ (%.1f events)'%(np.sum(NP_MC)),
		rasterized=rasterized, zorder=1)

	ax.bar(bin_c, MCreactor, bottom=MCatm+MCacc, lw=0.5, edgecolor='#FFD500', width=dx, label=r'reactors', facecolor='None', hatch=my_hatch,
		rasterized=rasterized, zorder=1)
	ax.bar(bin_c, MCacc, bottom=MCatm, lw=0.5, edgecolor='#5BD355', width=dx, label=r'atm', facecolor='None', hatch=my_hatch,
		rasterized=rasterized, zorder=1)
	ax.bar(bin_c, MCatm, lw=0.5, edgecolor='#5955D8', width=dx, label=r'atm', facecolor='None', hatch=my_hatch,
		rasterized=rasterized, zorder=1)
	# ax.bar(bin_c,MCaccidental, facecolor='lightskyblue',edgecolor='lightskyblue', width=dx, label=r'accidental', rasterized=False)
	ax.bar(bin_c, NP_MC, bottom=MCall, width=dx, lw=0.5, edgecolor='black', facecolor='None',
		rasterized=rasterized, zorder=1)

	ax.bar(bin_c, NP_MC+MCall, width=dx, lw=0.6, facecolor='None', edgecolor='black',
		rasterized=rasterized, zorder=1)

	# DATA
	ax.errorbar(bin_c, data, yerr= np.array([ERRORLOW,ERRORUP]), xerr = dx/2.0, \
													marker="o", markeredgewidth=0.5, capsize=1.0,markerfacecolor="white",\
													markeredgecolor="black", ms=3, color='black', lw = 0.0, elinewidth=0.8, zorder=100,label=r'data')

	##############
	# what boson
	if params.model == const.VECTOR:
		boson_string = r'$m_{Z^\prime}$'
		boson_file = 'vector'
	elif params.model == const.SCALAR:
		boson_string = r'$m_\phi$'
		boson_file = 'scalar'

	
	UEQSR = to_scientific_notation(params.Ue4**2)
	ax.legend(loc='upper right',frameon=False,ncol=1,markerfirst=False,fontsize=9)
	ax.set_title(r'$m_4 = %.0f$ eV,\, '%(params.m4*1e9)+boson_string+r'$/m_4 = %.2f$, \, $|U_{e 4}|^2 = \,$'%(params.mBOSON/params.m4)+UEQSR, fontsize=9)

	ax.annotate(r'SK-IV',xy=(0.8,0.26),xycoords='axes fraction',fontsize=13)
	ax.set_xlim(STARTPOINT,ENDPOINT)
	ax.set_ylim(0,)

	ax.set_xlabel(r'$E_\nu/$MeV')
	ax.set_ylabel(r'Events/MeV')
	fig_name = 'plots/IBD_spectra/'+boson_file+'_SK-IV_MN_%.0f_MB_%.0f.pdf'%(params.m4*1e9,params.mBOSON*1e9)

	save_figs(fig_name)

	plt.close()
	return fig, fig_name, ax

################################################################
# SUPER-K IV from thesis -- bad assumptions on efficiencies in v1
def superk_outdated(params,fluxfile,xsfile, rasterized=True):
	
	################################################################
	# fill bins
	exp = exps.superk_outdated_data()
	ENDPOINT = exp.fit_endpoint
	STARTPOINT = exp.fit_startpoint
	bins, NP_MC, back_MC, data = rates.fill_bins(exp,params,fluxfile,endpoint=ENDPOINT)
	ERRORLOW =  np.sqrt(data)
	ERRORUP = np.sqrt(data)

	mask = exp.bin_c < ENDPOINT
	bins = exp.bin_e[np.append(mask,True)]
	bin_c = exp.bin_c[mask]
	dx = exp.bin_w[mask]
	
	MCtot = exp.MCall[mask]
	MCaccidental = exp.MCaccidental[mask]
	MCreactor = exp.MCreactor[mask]
	MCreactorLi = exp.MCreactorLi[mask]

	################################################################
	# PLOTTING THE EVENT RATES 
	fig, ax = get_std_fig()

	# MC
	ax.bar(bin_c, NP_MC, bottom=MCtot, width=dx, lw=0, facecolor='grey', edgecolor='None', label=r'$\nu_4 \to \nu_e \nu_e \overline{\nu_e}$ (%.1f events)'%(np.sum(NP_MC)),
		rasterized=rasterized, zorder=1)

	ax.bar(bin_c, MCtot-MCreactor, bottom=MCreactor, lw=0.5, edgecolor='#FFD500', width=dx, label=r'reactors', facecolor='None', hatch=my_hatch,
		rasterized=rasterized, zorder=1)
	ax.bar(bin_c,MCreactor-MCreactorLi, bottom=MCreactorLi, lw=0.5, edgecolor='#5BD355', width=dx, label=r'spall ($^9$Li)', facecolor='None', hatch=my_hatch,
		rasterized=rasterized, zorder=1)
	ax.bar(bin_c,MCreactorLi, lw=0.5, edgecolor='#5955D8', width=dx, label=r'atm+NC+acc', facecolor='None', hatch=my_hatch,
		rasterized=rasterized, zorder=1)
	# ax.bar(bin_c,MCaccidental, facecolor='lightskyblue',edgecolor='lightskyblue', width=dx, label=r'accidental', rasterized=False)
	ax.bar(bin_c, NP_MC, bottom=MCtot, width=dx, lw=0.5, edgecolor='black', facecolor='None',
		rasterized=rasterized, zorder=1)
	ax.bar(bin_c, NP_MC+MCtot, width=dx, lw=0.6, facecolor='None', edgecolor='black',
		rasterized=rasterized, zorder=1)

	# DATA
	ax.errorbar(bin_c, data, yerr= np.array([ERRORLOW,ERRORUP]), xerr = dx/2.0, \
													marker="o", markeredgewidth=0.5, capsize=1.0,markerfacecolor="white",\
													markeredgecolor="black", ms=3, color='black', lw = 0.0, elinewidth=0.8, zorder=100,label=r'data')

	##############
	# what boson
	if params.model == const.VECTOR:
		boson_string = r'$m_{Z^\prime}$'
		boson_file = 'vector'
	elif params.model == const.SCALAR:
		boson_string = r'$m_\phi$'
		boson_file = 'scalar'

	
	UEQSR = to_scientific_notation(params.Ue4**2)
	ax.legend(loc='upper right',frameon=False,ncol=1,markerfirst=False,fontsize=9)
	ax.set_title(r'$m_4 = %.0f$ eV,\, '%(params.m4*1e9)+boson_string+r'$/m_4 = %.2f$, \, $|U_{e 4}|^2 = \,$'%(params.mBOSON/params.m4)+UEQSR, fontsize=9)

	ax.annotate(r'SK-IV',xy=(0.8,0.26),xycoords='axes fraction',fontsize=13)
	ax.set_xlim(STARTPOINT,ENDPOINT)
	ax.set_ylim(0,)

	ax.set_xlabel(r'$E_\nu/$MeV')
	ax.set_ylabel(r'Events/MeV')
	fig_name = 'plots/IBD_spectra/'+boson_file+'_SK-IV_OUTDATED_MN_%.0f_MB_%.0f.pdf'%(params.m4*1e9,params.mBOSON*1e9)

	save_figs(fig_name)

	plt.close()
	return fig, fig_name, ax
import numpy as np
from scipy import interpolate
import matplotlib 
matplotlib.use('agg') 
import matplotlib.pyplot as plt
from matplotlib import rc, rcParams
from matplotlib.pyplot import *

import vegas
import gvar as gv

from source import *

##########
# integration evaluations
rates.NEVALwarmup = 1e6
rates.NEVAL = 1e6

################################################################
# SETUP
################################################################
exp = exps.borexino_limit()
Bbin_c = exp.Enu_bin_c
Bbin_w = exp.Enu_bin_w
Bbin_e = exp.Enu_bin_e
Bfluxlimit = exp.fluxlimit
BfluxlimitATM = exp.fluxlimitATM

exp2 = exps.kamland_limit()
Kbin_c = exp2.Enu_bin_c
Kbin_w = exp2.Enu_bin_w
Kbin_e = exp2.Enu_bin_e
Kfluxlimit = exp2.fluxlimit

exp3 = exps.superk_limit()
Sbin_c = exp3.Enu_bin_c
Sbin_w = exp3.Enu_bin_w
Sbin_e = exp3.Enu_bin_e
Sfluxlimit = exp3.fluxlimit

###########
# NUMU FLUX
fluxfile = "fluxes/b8spectrum.txt"
flux = fluxes.get_exp_flux(fluxfile)

############
# NUE/BAR XS
xsfile="xsecs/IBD_160106169/TCS_CC_anue_p_1026_SV.txt"
xsec = lambda x : np.zeros(np.size(x)) 
xsecbar = lambda x: np.ones(np.size(x))*1e-38

bins = np.linspace(0.0,14.5,50)
dx = (bins[1:] - bins[:-1])
bin_c = bins[:-1] + dx/2.0

#############
# efficiencies
enu_eff= bins
eff= np.ones((np.size(bins)-1))
identity = lambda TTT : TTT

###########
# DECAY MODEL PARAMETERS
params = model.decay_model_params(const.SCALAR)
params.gx		= 1.0
params.Ue4		= np.sqrt(0.001)
params.Umu4		= np.sqrt(0.001)
params.UD4		= np.sqrt(1.0-params.Ue4*params.Ue4-params.Umu4*params.Umu4)
params.m4		= 300e-9 # GeV

params.mBOSON  = 0.9*params.m4 # GeV
NCASCADE, dNCASCADE = rates.RATES_dN_HNL_CASCADE_NU_NUBAR(\
											flux=flux,\
											xsec=xsec,\
											xsecbar=xsecbar,\
											dim=3,\
											enumin=0,\
											enumax=const.Enu_END_OF_SPECTRUM,\
											params=params,\
											bins=bins,\
											PRINT=True,\
											enu_eff=enu_eff,\
											eff=eff,
											smearing_function=identity)

params.mBOSON  = 0.5*params.m4 # GeV
NCASCADE2, dNCASCADE2 = rates.RATES_dN_HNL_CASCADE_NU_NUBAR(\
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

params.mBOSON  = 0.1*params.m4 # GeV
NCASCADE3, dNCASCADE3 = rates.RATES_dN_HNL_CASCADE_NU_NUBAR(\
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

###################################3
params.Ue4		= np.sqrt(0.01)
params.mBOSON  = 0.9*params.m4 # GeV
NCASCADE_001, dNCASCADE_001 = rates.RATES_dN_HNL_CASCADE_NU_NUBAR(\
											flux=flux,\
											xsec=xsec,\
											xsecbar=xsecbar,\
											dim=3,\
											enumin=0,\
											enumax=const.Enu_END_OF_SPECTRUM,\
											params=params,\
											bins=bins,\
											PRINT=True,\
											enu_eff=enu_eff,\
											eff=eff,
											smearing_function=identity)

params.mBOSON  = 0.5*params.m4 # GeV
NCASCADE2_001, dNCASCADE2_001 = rates.RATES_dN_HNL_CASCADE_NU_NUBAR(\
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

params.mBOSON  = 0.1*params.m4 # GeV
NCASCADE3_001, dNCASCADE3_001 = rates.RATES_dN_HNL_CASCADE_NU_NUBAR(\
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

print(dNCASCADE3)
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
axes_form  = [0.15,0.15,0.82,0.75]
fig = plt.figure()
ax = fig.add_axes(axes_form)

# E = np.linspace(0.001,13.5,1000)


##########################################################################
if params.model == const.VECTOR:
	boson_string = r'$m_{Z^\prime}$'
	boson_file = 'vector'
elif params.model == const.SCALAR:
	boson_string = r'$m_\phi$'
	boson_file = 'scalar'

ax.fill_between(np.append(Sbin_c-0.5,Sbin_c[-1]+0.5), np.append(Sfluxlimit,1e8), np.ones(np.size(Sbin_c)+1)*1e10, step = 'post', lw=0.0, alpha=0.5, color='darkgreen')
ax.fill_between(np.append(Kbin_c-0.5,Kbin_c[-1]+0.5), np.append(Kfluxlimit,1e8), np.ones(np.size(Kbin_c)+1)*1e10, step = 'post', lw=0.0, alpha=0.5, color='indigo')
ax.fill_between(np.append(Bbin_c-0.5,Bbin_c[-1]+0.5), np.append(Bfluxlimit,1e8), np.ones(np.size(Bbin_c)+1)*1e10, step = 'post', lw=0.0, alpha=0.5, color='dodgerblue')

ax.step(np.append(0.0,np.append(Sbin_c-0.5,Sbin_c[-1]+0.5)), np.append(1e8,np.append(Sfluxlimit,1e8)), where = 'post', color='grey', lw=0.5)
ax.step(np.append(0.0,np.append(Kbin_c-0.5,Kbin_c[-1]+0.5)), np.append(1e8,np.append(Kfluxlimit,1e8)), where = 'post', color='indigo', lw=0.5)
ax.step(np.append(0.0,np.append(Bbin_c-0.5,Bbin_c[-1]+0.5)), np.append(1e8,np.append(Bfluxlimit,1e8)), where = 'post', color='blue', lw=0.5)

ax.plot(bin_c, dNCASCADE/dx*1e38, lw=1.2, color='black', label=boson_string+r'/$m_4 = 0.9$')
ax.plot(bin_c, dNCASCADE2/dx*1e38, lw=1, dashes = (6,1), color='black', label=boson_string+r'/$m_4 = 0.5$')
ax.plot(bin_c, dNCASCADE3/dx*1e38, lw=1, dashes = (2,1), color='black', label=boson_string+r'/$m_4 = 0.1$')

ax.plot(bin_c, dNCASCADE_001/dx*1e38, lw=1.2, color='darkorange')
ax.plot(bin_c, dNCASCADE2_001/dx*1e38, lw=1, dashes = (6,1), color='darkorange')
ax.plot(bin_c, dNCASCADE3_001/dx*1e38, lw=1, dashes = (2,1), color='darkorange')

ax.text(9.5,2.2e2,r'Borexino',fontsize=10,color='white')
ax.text(11.4,37,r'KamLAND',fontsize=10,color='white')
ax.text(14.5,5.5,r'SK-IV',fontsize=10,color='white')
ax.text(0.4,0.4e4,r'$|U_{e4}|^2=10^{-2}$',fontsize=8.5,color='chocolate',rotation=-20)
ax.text(0.3,3e2,r'$|U_{e4}|^2=10^{-3}$',fontsize=8.5,color='black',rotation=-18)

##############
# STYLE
ax.set_yscale('log')

def to_scientific_notation(number):
    a, b = '{:.4E}'.format(number).split('E')
    b = int(b)
    a = float(a)
    return r'$%.0f \times 10^{%i}$'%(a,b)
UQSR = to_scientific_notation(params.Umu4**2)
ax.legend(loc='lower left',frameon=False,ncol=1,markerfirst=True,fontsize=8.5)
ax.set_title(r'$m_4 = %.0f$ eV, $|U_{\mu 4}|^2 = \,$'%(params.m4*1e9)+UQSR, fontsize=8.5)

ax.set_xlim(0,18.3)
ax.set_ylim(0.4, 2e5)

ax.set_ylabel(r'$\frac{{\rm d}\Phi}{{\rm d}E_\nu}$ [cm$^{-2}$ s$^{-1}$ MeV$^{-1}$]')
ax.set_xlabel(r'$E_\nu$ [MeV]')

fig.savefig('plots/Fluxlimit_'+boson_file+'.pdf',rasterized=True)
import numpy as np
import matplotlib 
matplotlib.use('agg') 
import matplotlib.pyplot as plt
from matplotlib import rc, rcParams
from matplotlib.pyplot import *
from matplotlib.legend_handler import HandlerLine2D

import vegas
import gvar as gv
from scipy.stats import chi2
from source import *


############
# NUMU FLUX
fluxfile = "fluxes/b8spectrum.txt"

###########
# DECAY MODEL PARAMETERS
params = model.decay_model_params(const.SCALAR)
params.gx		= 1.0
params.Ue4		= np.sqrt(0.01)
params.Umu4		= np.sqrt(0.01)
params.Utau4    = np.sqrt(0)
params.UD4		= np.sqrt(1.0-params.Ue4*params.Ue4-params.Umu4*params.Umu4)
params.m4		= 300e-9 # GeV
params.mBOSON  = 0.9*params.m4 # GeV

KAM = exps.kamland_data()
BOR = exps.kamland_data()

err_flux = 0.1
err_back = KAM.err_back


bK, npK, backK, dK = rates.fill_bins(KAM,params,fluxfile,endpoint=17)


NPOINTS = 33
UE4SQR =np.logspace(-5,-1,NPOINTS)
UMU4SQR =np.logspace(-4,-1,NPOINTS)
L = np.zeros((NPOINTS,NPOINTS))

dof = np.size(dK)
old_factor = (params.Ue4**2*std_osc.Padiabatic(bK, -const.nue_to_nue) + params.Umu4**2*std_osc.Padiabatic(bK, -const.numu_to_nue))/(params.Ue4**2 + params.Umu4**2 +params.Utau4**2)
for i in range(np.size(UE4SQR)):
	for j in range(np.size(UMU4SQR)):
		new_factor = (UE4SQR[i]*std_osc.Padiabatic(bK, -const.nue_to_nue) + UMU4SQR[j]*std_osc.Padiabatic(bK, -const.numu_to_nue))/(UE4SQR[i] + UMU4SQR[j] +params.Utau4**2)
		np_new = new_factor/old_factor*npK
		L[j,i] = np.sum(np_new)#stats.chi2_binned_rate(np_new, backK, dK, [err_flux,err_back])

print L

################################################################
# PLOTTING THE LIMITS
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

X,Y = np.meshgrid(UE4SQR,UMU4SQR)
# ax.contour(X,Y,L, [chi2.ppf(0.90, dof)], color='black')
ax.contour(X,Y,L, 20, color='black')

############
# GET THE FIT REGIONS FROM DENTLER ET AL
DentlerPath='digitized/Dentler_et_al/'
MB_ue_b,MB_umu_b = np.genfromtxt(DentlerPath+'bottom_MiniBooNE_300.txt',unpack=True)
MB_ue_t,MB_umu_t = np.genfromtxt(DentlerPath+'top_MiniBooNE_300.txt',unpack=True)

MB_ue_f=np.logspace( np.log10(np.min([MB_ue_b])), np.log10(np.max([MB_ue_b])), 100)
MB_umu_b_f = np.interp(MB_ue_f,MB_ue_b,MB_umu_b)

MB_umu_t_f = np.interp(MB_ue_f,MB_ue_t,MB_umu_t)

ax.fill_between(MB_ue_f,MB_umu_b_f,MB_umu_t_f,facecolor='yellow',alpha=0.5,lw=0)
ax.fill_between(MB_ue_f,MB_umu_b_f,MB_umu_t_f,edgecolor='darkorange',facecolor='None',lw=0.8)


y,x = np.genfromtxt(DentlerPath+'right_LSND_300.txt',unpack=True)
yl,xl = np.genfromtxt(DentlerPath+'left_LSND_300.txt',unpack=True)
x_f=np.logspace( np.log10(np.min([x])), np.log10(np.max([x])), 100)
y_f = np.interp(x_f,x,y)
yl_f = np.interp(x_f,xl,yl)

ax.fill_betweenx(x_f,y_f,yl_f,facecolor='firebrick',alpha=0.5,lw=0)
ax.fill_betweenx(x_f,y_f,yl_f,edgecolor='firebrick',facecolor='None',lw=0.8)

y,x = np.genfromtxt(DentlerPath+'right_combined_300.txt',unpack=True)
yl,xl = np.genfromtxt(DentlerPath+'left_combined_300.txt',unpack=True)
x_f=np.logspace( np.log10(np.min([x])), np.log10(np.max([x])), 100)
y_f = np.interp(x_f,x,y)
yl_f = np.interp(x_f,xl,yl)

ax.fill_betweenx(x_f,y_f,yl_f,facecolor='darkorange',alpha=0.5,lw=0)
ax.fill_betweenx(x_f,y_f,yl_f,edgecolor='firebrick',facecolor='None',lw=0.8)



ax.set_xscale('log')
ax.set_yscale('log')
##############
# STYLE
if params.model == const.VECTOR:
	boson_string = r'$m_{Z^\prime}$'
	boson_file = 'vector'
elif params.model == const.SCALAR:
	boson_string = r'$m_\phi$'
	boson_file = 'scalar'

ax.legend(loc='upper right',frameon=False,ncol=1)
ax.set_title(r'$m_4 = %.0f$ eV,\, '%(params.m4*1e9)+boson_string+r'$/m_4 = %.2f$'%(params.mBOSON/params.m4), fontsize=fsize)

# ax.annotate(r'Borexino',xy=(0.55,0.35),xycoords='axes fraction',fontsize=14)
ax.set_xlim(1e-5,0.1)
ax.set_ylim(1e-4,0.1)
# ax.set_ylim(0,)
ax.set_xlabel(r'$|U_{e 4}|^2$')
ax.set_ylabel(r'$|U_{\mu 4}|^2$')
fig.savefig('plots/limits_MN_%.0f_MB_%.0f.pdf'%(params.m4*1e9,params.mBOSON*1e9),rasterized=True)
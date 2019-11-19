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
import matplotlib.colors as colors

rates.NEVALwarmup = 1e4
rates.NEVAL = 1e5

############
# NUMU FLUX
fluxfile = "fluxes/b8spectrum.txt"

###########
# DECAY MODEL PARAMETERS
params = model.decay_model_params(const.SCALAR)
params.gx		= 1.0
params.Ue4		= np.sqrt(0.0003)
params.Umu4		= np.sqrt(0.0003)
params.Utau4    = np.sqrt(0)
params.UD4		= np.sqrt(1.0-params.Ue4*params.Ue4-params.Umu4*params.Umu4)
params.m4		= 300e-9 # GeV
params.mBOSON  = 0.9*params.m4 # GeV

KAM = exps.kamland_data()
BOR = exps.borexino_data()
SK = exps.superk_data()

print stats.chi2_model_independent(exps.borexino_limit(),params,fluxfile)

err_flux = 0.1
err_backK = KAM.err_back/2.0
err_backB = BOR.err_back
err_backS = SK.err_back/2.0


bK, npK, backK, dK = rates.fill_bins(KAM,params,fluxfile,endpoint=17)
bB, npB, backB, dB = rates.fill_bins(BOR,params,fluxfile,startpoint=0,endpoint=17)
bS, npS, backS, dS = rates.fill_bins(SK,params,fluxfile,endpoint=17)

# print backB, dB, npB

NPOINTS = 33
UE4SQR =np.logspace(-4,-1,NPOINTS)
UMU4SQR =np.logspace(-4,-1,NPOINTS)
LK = np.zeros((NPOINTS,NPOINTS))
LB = np.zeros((NPOINTS,NPOINTS))
LS = np.zeros((NPOINTS,NPOINTS))

dofK = np.size(dK)
dofB = np.size(dB)
dofS = np.size(dS)
old_factorK = params.Ue4**2*(params.Ue4**2*std_osc.Padiabatic(bK, -const.nue_to_nue) + params.Umu4**2*std_osc.Padiabatic(bK, -const.numu_to_nue))/(params.Ue4**2 + params.Umu4**2 +params.Utau4**2)
old_factorB = params.Ue4**2*(params.Ue4**2*std_osc.Padiabatic(bB, -const.nue_to_nue) + params.Umu4**2*std_osc.Padiabatic(bB, -const.numu_to_nue))/(params.Ue4**2 + params.Umu4**2 +params.Utau4**2)
old_factorS = params.Ue4**2*(params.Ue4**2*std_osc.Padiabatic(bS, -const.nue_to_nue) + params.Umu4**2*std_osc.Padiabatic(bS, -const.numu_to_nue))/(params.Ue4**2 + params.Umu4**2 +params.Utau4**2)

for i in range(np.size(UE4SQR)):
	for j in range(np.size(UMU4SQR)):
		new_factorK = UE4SQR[i]*(UE4SQR[i]*std_osc.Padiabatic(bK, -const.nue_to_nue) + UMU4SQR[j]*std_osc.Padiabatic(bK, -const.numu_to_nue))/(UE4SQR[i] + UMU4SQR[j] +params.Utau4**2)
		new_factorB = UE4SQR[i]*(UE4SQR[i]*std_osc.Padiabatic(bB, -const.nue_to_nue) + UMU4SQR[j]*std_osc.Padiabatic(bB, -const.numu_to_nue))/(UE4SQR[i] + UMU4SQR[j] +params.Utau4**2)
		new_factorS = UE4SQR[i]*(UE4SQR[i]*std_osc.Padiabatic(bS, -const.nue_to_nue) + UMU4SQR[j]*std_osc.Padiabatic(bS, -const.numu_to_nue))/(UE4SQR[i] + UMU4SQR[j] +params.Utau4**2)
		np_newK = new_factorK/old_factorK*npK
		np_newB = new_factorB/old_factorB*npB
		np_newS = new_factorS/old_factorS*npS
		# print new_factorK,new_factorB
		LK[j,i] = stats.chi2_binned_rate(np_newK, backK, dK, [err_flux,err_backK])#np.sum(np_newK)
		LB[j,i] = stats.chi2_binned_rate(np_newB, backB, dB, [err_flux,err_backB])#np.sum(np_newB)
		LS[j,i] = stats.chi2_binned_rate(np_newS, backS, dS, [err_flux,err_backS])#np.sum(np_newS)
print np.min(LK), dofK
LK = LK - np.min(LK)
print np.min(LB), dofB
LB = LB - np.min(LB)
print np.min(LS), dofS
LS = LS - np.min(LS)

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
# ax.contourf(X,Y,LK, [chi2.ppf(0.90, dofK),1e100], colors=['black'],alpha=0.1, linewidths=[0.1])
ax.contour(X,Y,LK, [chi2.ppf(0.90, dofK)], colors=['blue'],linewidths=[1.0])
# ax.contour(X,Y,L, 20, color='black')

# ax.contourf(X,Y,LB, [chi2.ppf(0.90, dofB),1e100], colors=['black'],alpha=0.1, linewidths=[0.1])
ax.contour(X,Y,LB, [chi2.ppf(0.90, dofB)], colors=['red'],linewidths=[1.0])
Z = LB
# pcm = ax.pcolor(X, Y, Z,
#                    norm=colors.LogNorm(vmin=Z.min(), vmax=Z.max()),
#                    cmap='PuBu_r')# ax.contour(X,Y,L, 20, color='black')
# fig.colorbar(pcm, ax=ax, extend='max')

ax.contour(X,Y,LS, [chi2.ppf(0.90, dofS)], colors=['green'],linewidths=[1.0], ls='--')

ax.annotate(r'MiniBooNE',xy=(0.55,0.71),xycoords='axes fraction',color='firebrick',fontsize=10,rotation=-32)
ax.annotate(r'LSND',xy=(0.85,0.9),xycoords='axes fraction',color='firebrick',fontsize=10,rotation=0)
ax.annotate(r'\noindent All w/o\\LSND',xy=(0.4,0.85),xycoords='axes fraction',color='darkred',fontsize=10,rotation=0)
ax.annotate(r'', fontsize=fsize, xy=(1.3e-3,2e-4), xytext=(5.2e-4,2e-4),color='blue',
            arrowprops=dict(arrowstyle="-|>", mutation_scale=5, color='blue', lw = 0.5),
            )
ax.annotate(r'KamLAND $90\%$ C.L.', fontsize=0.8*fsize, xy=(0.45,0.17), xytext=(0.3,0.18),xycoords='axes fraction', color='blue')
ax.annotate(r'excluded', fontsize=0.8*fsize, xy=(0.45,0.2), xytext=(0.3,0.08),xycoords='axes fraction', color='blue')



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
ax.set_xlim(1e-4,0.04)
ax.set_ylim(1e-4,0.012)
# ax.set_ylim(0,)
ax.set_xlabel(r'$|U_{e 4}|^2$')
ax.set_ylabel(r'$|U_{\mu 4}|^2$')
fig.savefig('plots/limits_MN_%.0f_MB_%.0f.pdf'%(params.m4*1e9,params.mBOSON*1e9),rasterized=True)
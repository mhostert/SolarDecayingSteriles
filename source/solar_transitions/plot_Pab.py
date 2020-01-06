import numpy as np
import matplotlib 
# matplotlib.use('agg') 
import matplotlib.pyplot as plt
from matplotlib import rc, rcParams
from matplotlib.pyplot import *
from matplotlib.legend_handler import HandlerLine2D

sys.path.insert(1, '../source')

import standard_oscillations as std_osc
import const

a = np.genfromtxt('Pab_noCP.dat', unpack=True)
aCP = np.genfromtxt('Pab_CP.dat', unpack=True)
b = np.genfromtxt('Pab_bar_noCP.dat', unpack=True)
bCP = np.genfromtxt('Pab_bar_CP.dat', unpack=True)

Enu = a[0,:]

def Pab(i,CP):
	if i>0 and not CP:
		return a[i,:]
	if i>0 and CP:
		return aCP[i,:]
	if i<0 and not CP:
		return b[-i,:]
	if i<0 and CP:
		return bCP[-i,:]

################################################################
# PLOTTING THE EVENT RATES 
################################################################
fsize=11
rc('text', usetex=True)
rcparams={'axes.labelsize':fsize,'xtick.labelsize':fsize,'ytick.labelsize':fsize,\
				'figure.figsize':(1.2*3.7,2.2*2.3617)	}
rc('font',**{'family':'serif', 'serif': ['computer modern roman']})
matplotlib.rcParams['hatch.linewidth'] = 0.1  # previous pdf hatch linewidth

rcParams.update(rcparams)
axes_form1  = [0.15,0.13+0.46,0.78,0.38]
axes_form2  = [0.15,0.11,0.78,0.38]
fig = plt.figure()
ax = fig.add_axes(axes_form1)
ax2 = fig.add_axes(axes_form2)


# ax.plot(Enu, Pab(2), color='black', dashes=(5,1), label=r'$P_{ee}$')
# ax.plot(Enu, Pab(4), color='indigo', dashes=(5,1), label=r'$P_{e\mu}$')
# ax.plot(Enu, Pab(6), color='dodgerblue',dashes=(5,1), label=r'$P_{e\tau}$')
ax.plot([],[],lw=0.8,c='black', label=r'$\delta = 0$')
ax.plot([],[],lw=0.8,c='black',dashes=(2,1),label=r'$\delta = \pi$')
ax2.plot([],[],lw=0.8,c='black', label=r'$\delta = 0$')
ax2.plot([],[],lw=0.8,c='black',dashes=(2,1),label=r'$\delta = \pi$')
ax.legend(loc='upper center', frameon=False, markerfirst=False,ncol=3)
ax2.legend(loc='upper center', frameon=False, markerfirst=False,ncol=3)

alpha=0.5
i=0
ax.fill_between(Enu, Pab(-(1+i),False), Pab(-(1+i),True), color='black', alpha=alpha, lw=0.0)
ax.plot(Enu, Pab(-(1+i),False),  lw=0.8, color='black')
ax.plot(Enu, Pab(-(1+i),True), lw=0.8, color='black', label=r'$P_{\overline{ee}}$', dashes=(2,1))

ax2.fill_between(Enu, Pab((1+i),False), Pab((1+i),True), color='black', alpha=alpha, lw=0.0)
ax2.plot(Enu, Pab((1+i),False),  lw=0.8, color='black')
ax2.plot(Enu, Pab((1+i),True), lw=0.8, color='black', label=r'$P_{ee}$', dashes=(2,1))
# ax.plot(Enu, Pab(-2), color='black', dashes=(5,1), label=r'$\overline{P_{ee}}$')

ax.fill_between(Enu, Pab(-(7+i),False), Pab(-(7+i),True), color='darkorange', alpha=alpha, lw=0.0)
ax.plot(Enu, Pab(-(7+i),False),  lw=0.8, color='darkorange')
ax.plot(Enu, Pab(-(7+i),True), lw=0.8, color='darkorange', label=r'$P_{\overline{\mu e}}$', dashes=(2,1))

ax2.fill_between(Enu, Pab((7+i),False), Pab((7+i),True), color='darkorange', alpha=alpha, lw=0.0)
ax2.plot(Enu, Pab((7+i),False),  lw=0.8, color='darkorange')
ax2.plot(Enu, Pab((7+i),True), lw=0.8, color='darkorange', label=r'$P_{\mu e}$', dashes=(2,1))
# ax.plot(Enu, Pab(-4), color='indigo', dashes=(5,1), label=r'$\overline{P_{e\mu}}$')

ax.fill_between(Enu, Pab(-(13+i),False), Pab(-(13+i),True), color='dodgerblue', alpha=alpha, lw=0.0)
ax.plot(Enu, Pab(-(13+i),False),  lw=0.8, color='dodgerblue')
ax.plot(Enu, Pab(-(13+i),True), lw=0.8, color='dodgerblue',label=r'$P_{\overline{\tau e}}$', dashes=(2,1))

ax2.fill_between(Enu, Pab((13+i),False), Pab((13+i),True), color='dodgerblue', alpha=alpha, lw=0.0)
ax2.plot(Enu, Pab((13+i),False),  lw=0.8, color='dodgerblue')
ax2.plot(Enu, Pab((13+i),True), lw=0.8, color='dodgerblue',label=r'$P_{\tau e}$', dashes=(2,1))
# ax.plot(Enu, Pab(-6), color='dodgerblue',dashes=(5,1), label=r'$\overline{P_{e\tau}}$')

ax.text(0.12,0.6,r'$P_{\overline{ee}}$')
ax.text(0.12,0.3,r'$P_{\overline{\mu e}}$',color='darkorange')
ax.text(0.12,0.08,r'$P_{\overline{\tau e}}$', color='dodgerblue')

ax2.text(0.12,0.57,r'$P_{{ee}}$')
ax2.text(0.12,0.3,r'$P_{{\mu e}}$',color='darkorange')
ax2.text(0.12,0.08,r'$P_{{\tau e}}$', color='dodgerblue')

# ax.text(6,0.85,r'$\delta \in [0,\pi]$', color='black')
# ax2.text(6,0.85,r'$\delta \in [0,\pi]$', color='black')

ax.set_ylim(0,1)
ax2.set_ylim(0,1)

ax.set_xlim(0.1,16.5)
ax2.set_xlim(0.1,16.5)

ax.set_xscale('log')
ax2.set_xscale('log')

ax.set_ylabel(r'Probability')
ax2.set_ylabel(r'Probability')

ax.set_xlabel(r'$E_\nu/$MeV')
ax2.set_xlabel(r'$E_\nu/$MeV')
# ax.set_xticks([])


# e,p = np.loadtxt('BOREXINO_PEE.dat',unpack=True)
# ax2.plot(e,p,c='pink')
# ax2.plot(Enu, std_osc.Padiabatic(Enu,const.nue_to_nue),c='red', lw=1.0, label=r'$\nu_e \to \nu_e$')
# ax.plot(Enu, std_osc.Padiabatic(Enu,-const.nue_to_nue),c='red', lw=1.0, label=r'$\overline{\nu_e} \to \overline{\nu_e}$')


fig.savefig('Psolar.pdf')
plt.show()

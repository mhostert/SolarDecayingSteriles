import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc, rcParams
import matplotlib.lines as mlines
from matplotlib.pyplot import *
from scipy import interpolate
import scipy.stats
import sys
import scipy.ndimage as ndimage

Emu, f_mu = np.genfromtxt('Flux_LAr1ND_numu.txt', unpack=True, skip_header=0)
Emub, f_mub = np.genfromtxt('Flux_LAr1ND_numubar.txt', unpack=True, skip_header=0)
Ee, f_e = np.genfromtxt('Flux_LAr1ND_nue.txt', unpack=True, skip_header=0)
Eeb, f_eb = np.genfromtxt('Flux_LAr1ND_nuebar.txt', unpack=True, skip_header=0)

Delta_E = 0.05


Eint = np.linspace(0.0,3, (3)/Delta_E)

f_mu_int = np.interp(Eint, Emu, f_mu, left = 0.0)
f_mub_int = np.interp(Eint, Emub, f_mub, left = 0.0)
f_e_int = np.interp(Eint, Ee, f_e, left = 0.0)
f_eb_int = np.interp(Eint, Eeb, f_eb, left = 0.0)

f = open("Flux_SBN_LAr1ND.txt", 'w')
f.write('# flux / 50 MeV / m^2 / 10^6 POTs @ LAr1ND (110 m)\n \n')

np.savetxt("Flux_SBN_LAr1ND.txt", zip(Eint,f_e_int, f_mu_int, np.zeros(np.size(Eint)), f_eb_int, f_mub_int, np.zeros(np.size(Eint))),\
					delimiter = '\t', fmt='%1.2f %2.8f %2.8f %2.8f %2.8f %2.8f %2.8f' )

fig = plt.figure(figsize=(5,3.4))
fsize = 9
rc('text', usetex=True)
params={'axes.labelsize':fsize,'xtick.labelsize':fsize+1,'ytick.labelsize':fsize+1}
rcParams['font.family'] = 'serif'
rcParams['font.sans-serif'] = ['Computer Modern']
rcParams.update(params)

plt.step(Eint, f_mu_int, lw=1.0,label=r'$\nu_{\mu}$')
# plt.step(Emu, f_mu, ls='--')
plt.step(Eint, f_e_int, lw=1.0,label=r'$\nu_{e}$')
# plt.step(Ee, f_e, ls='--')
plt.step(Eint,f_mub_int, lw=1.0,label=r'$\overline{\nu}_{\mu}$')
# plt.step(Emub, f_mub,ls='--')
plt.step(Eint, f_eb_int, lw=1.0,label=r'$\overline{\nu}_{e}$')
plt.legend(loc='upper right', fontsize=9)
plt.text(1.5, 5,'LAr1ND')

plt.xlim(0.0, 3)
plt.xlabel(r'$E$ (GeV)')
plt.ylabel(r"$\Phi (\nu)/50$MeV$/$m$^2/10^6$POT$")

plt.yscale('log')

fig.savefig('SBN_interp_fluxes.pdf')
plt.show()
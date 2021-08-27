import matplotlib.pyplot as plt
from matplotlib import rc, rcParams
import matplotlib.lines as mlines
from matplotlib.pyplot import *
import scipy.ndimage as ndimage
from scipy import interpolate
import numpy as np
import scipy.stats
import sys


Umu4_2 = pow(1,2)

def cm2inch(*tupl):
    inch = 2.54
    if isinstance(tupl[0], tuple):
        return tuple(i/inch for i in tupl[0])
    else:
        return tuple(i/inch for i in tupl)

def plot_comparison(S,theta):
	
	thick =1.2
	rc('text', usetex=True)
	rc('font',**{'family':'serif','serif':['Computer Modern']})
	rc('axes', linewidth=thick)
	fsize = 11
	params={'axes.labelsize':fsize,'xtick.labelsize':fsize,'ytick.labelsize':fsize,\
			'figure.figsize':(cm2inch(40.0/2.0, 16.15/2.0))}
	rc('font',**{'family':'serif','serif':['Computer Modern Roman']})
	rcParams.update(params)


	fig = plt.figure()
	ax = fig.add_axes([0.085,0.15,0.9,0.8])

	# ax.plot([31.5281,1e8], [2.24611e-5/Umu4_2,2.24611e-5/Umu4_2], c='RoyalBlue', lw=thick*1.5, linestyle="--", dashes=(6,4))
	# ax.plot([31.5281,1e8], [3.16216e-5/Umu4_2,3.16216e-5/Umu4_2], c='RoyalBlue', lw=thick*1.5, linestyle="--", dashes=(6,4))


	if (S=='edis'):
		# ax.plot([31.63,1e8], [0.179542,0.179542], c='RoyalBlue', lw=thick*1.3, linestyle="--")
		# ax.plot([31.63,1e8], [0.1758,0.1758], c='RoyalBlue', lw=thick*1.3, linestyle="--")

		kev_labels = [[100.90, 0.9*np.sqrt(8e-5)],
									[193, np.sqrt(1.304e-4)],
									[1.5*439.299,0.8*np.sqrt(7.65e-5)],
									[1.5*1883.26,0.6*np.sqrt(0.6e-5)],
									[2250.49,np.sqrt(10e-5)],
									[12343.4,np.sqrt(6.96e-8)],
									[23334.6,np.sqrt(10.634e-7)],
									[40277,np.sqrt(2.1919e-6)],
									[348374,1.*np.sqrt(9.46617e-6)],
									[636073,np.sqrt(4.10313e-6)],
									[0.6*1883.26,1.3*np.sqrt(0.2*1.050e-5)],
									[8e4,0.5e-3]]

		for x in xrange(1,13):
			temp_x, temp_y = np.genfromtxt('digitized/Datasets/kev_bounds/Bound'+str(x)+'.txt', unpack=True, skip_header=0)
			one_line = np.ones(np.size(temp_x))
			if x == 12:
				temp_y*=1e-1
				temp_x*=1e3
			ax.fill_between(temp_x,(temp_y/100.0), one_line,\
							color="lightblue", linestyle="-", lw=0, alpha = 0.4, zorder = -1)
			ax.plot(temp_x,(temp_y/100.0),\
							color="darkblue", linestyle="-", lw=thick*0.3, alpha = 1.0, zorder = -1)
			ax.text(kev_labels[x-1][0],kev_labels[x-1][1], r''+str(x),\
								fontsize=fsize, horizontalalignment='center', verticalalignment='top',color='blue')
			
		ax.plot([0,1e8],[4e-3,4e-3],ls='--',color='black')
		ax.text(10,1.6e-3,r'$|U_{e 4}|^2 = 4\times 10^{-3}$')
		# xSBN, ySBN = np.genfromtxt('digitized/Datasets/SBN/Marks_Paper/E_Disappearance_only_3plus1.txt', unpack=True, skip_header=0)
		# lSBN, = ax.plot(np.sqrt(ySBN),(np.sin(np.arcsin(np.sqrt(xSBN))/2.0))**2, color="orange", lw=thick*1.3, dashes=(6,2), label=r'90\% C.L. SBN',zorder=5)

		ySK = 0.065*np.ones((2))
		xSK = [0.1,1e8]
		lSK, = ax.plot(xSK,ySK, color="darkorange", lw=thick*1, label=r'95\% C.L. Solar exps')

		xBUG, yBUG = np.genfromtxt('digitized/Datasets/SBN/Marks_Paper/Bugey_90CL.txt', unpack=True, skip_header=0)
		lBUG, = ax.plot(np.sqrt(yBUG),(np.sin(np.arcsin(np.sqrt(xBUG))/2.0))**2, c="red", lw=thick*1, label=r'90\% C.L. Bugey')
		
		xCarbon, yCarbon = np.genfromtxt('digitized/Datasets/KARMEN+LSND/LSND-KARMEN.dat', unpack=True, skip_header=0)
		lCarbon, = ax.plot(np.sqrt(yCarbon),(np.sin(np.arcsin(np.sqrt(xCarbon))/2.0))**2, c="violet", lw=thick*1, label=r'95\% C.L. LSND+KARMEN')
		
		xNEOS, yNEOS = np.genfromtxt('digitized/Datasets/Updated_Kopp/NEOS+DayaBay.dat', unpack=True, skip_header=0)
		lNEOS, = ax.plot(np.sqrt(yNEOS),xNEOS, c="green", lw=thick, label=r'95\% C.L. NEOS+DayaBay', zorder=-1)
		# xNuSTORM,yNuSTORM = np.genfromtxt('digitized/Datasets/NuSTORM_dis_99CL.txt', unpack=True, skip_header=0)
		# ax.plot(xNuSTORM,yNuSTORM, c="green", linestyle="-", lw="2.0", label=r'arXiv:1402.5250 99\% C.L.')
			
		plt.xlim(3e-4, 0.9)
		ax.legend(loc='upper left', shadow=0, fontsize=fsize, frameon=0)
		# ax.text(0.0013, 60,r"$\ell_{p} = N \times 226$ m", fontsize=fsize)
		# ax.text(0.0013, 35,r"$\Gamma_{\mu} = 3 \times 10^{-19}$ GeV", fontsize=fsize)
		# ax.text(0.0013, 20.7,r"$v_{\mu} = 1.0$", fontsize=fsize)
		# leg1 = ax.legend(handles=[l1,lSBN],loc='lower right',  shadow=0, fontsize=fsize, frameon=0)
	 	# plt.gca().add_artist(leg1)
		leg2 = ax.legend(handles=[lNEOS,lBUG,lSK,lCarbon],loc='lower center',  shadow=0, fontsize=fsize*0.8, ncol=2, frameon=0)

		plt.xlim(10**(-0.5),2.3*10**(6))
		plt.ylim(10**(-5),np.sqrt(0.3))
		plt.grid(True,which='both',lw=0.4,alpha=0.5)

	
		# ax.text(8.5e-3,0.6, r'FD stat', rotation=-20, fontsize=fsize)
		# ax.text(1.3e-3,17.25, r'ND stat', rotation=-20, fontsize=fsize)
		# ax.text(0.22,0.42, r'FD ', rotation=-20, fontsize=fsize)
		# ax.text(0.37,23.03,r'ND', rotation=-20, fontsize=fsize)


	if (S=='dis'):
		
		# ax.plot([31.6361, 1e8], [0.061518, 0.061518], c='RoyalBlue', lw=thick*1.3, linestyle="--", dashes=(6,2),zorder=5)
		# ax.plot([31.6361, 1e8], [0.059221, 0.059221], c='RoyalBlue', lw=thick*1.3, linestyle="--", dashes=(6,2))
		ax.plot([0,1e8],[4e-3,4e-3],ls='--',color='black')
		ax.text(10,2.3e-3,r'$|U_{\mu 4}|^2 = 4\times 10^{-3}$')
		plt.grid(True,which='both',lw=0.4,alpha=0.5)

		xIC,yIC = np.genfromtxt('digitized/Datasets/IceCUBE99CL.txt', unpack=True, skip_header=0)
		lIC, = ax.plot(np.sqrt(yIC),(np.sin(np.arcsin(np.sqrt(xIC))/2.0))**2, c="r", lw=thick*1, label=r'99\% C.L. IceCUBE',zorder=-1)
		
		# xMINOSp,yMINOSp = np.genfromtxt('digitized/Datasets/MINOS/MINOS+.txt', unpack=True, skip_header=0)
		# lMINOSp, = ax.plot(np.sqrt(yMINOSp),(np.sin(np.arcsin(np.sqrt(xMINOSp))/2.0))**2, c="black", lw=thick*1.3, label=r'90\% C.L. MINOS \& MINOS+')
		
		xMINOSp,yMINOSp = np.genfromtxt('digitized/Datasets/MINOS-MINOS+/MINOS-MINOS+.dat', unpack=True, skip_header=0)
		lMINOSp, = ax.plot(np.sqrt(yMINOSp),xMINOSp, c="black", lw=thick*1, label=r'90\% C.L. MINOS \& MINOS+',zorder=-1)
		
		ax.fill_between(np.sqrt(yMINOSp[-8:]),xMINOSp[-8:],xMINOSp[-8:]*1.3, facecolor='None', edgecolor='black', hatch='//////', linewidth=0, label=r'90\% C.L. MINOS \& MINOS+',zorder=-1)
		
		ySK = 0.037*np.ones((2))
		xSK = [0.1,1e8]
		lSK, = ax.plot(xSK,ySK, color="g", lw=thick*1.2, label=r'90\% C.L. SK+DC')

		# xSBN, ySBN = np.genfromtxt('digitized/Datasets/SBN/Marks_Paper/Dissapearance_only_3plus1.txt', unpack=True, skip_header=0)
		# lSBN, = ax.plot(np.sqrt(ySBN),(np.sin(np.arcsin(np.sqrt(xSBN))/2.0))**2, color="orange", dashes=(6,2), lw=thick*1, label=r'90\% C.L. SBN',zorder=5)

		# xBUG, yBUG = np.genfromtxt('digitized/Datasets/SBN/Marks_Paper/Bugey_90CL.txt', unpack=True, skip_header=0)
		# lBUG, = ax.plot(np.sqrt(yBUG),(np.sin(np.arcsin(np.sqrt(xBUG))/2.0))**2, c="darkgreen",dashes=((3.0,3.0)), lw=thick*1.2, label=r'90\% C.L. Bugey')
		# xNuSTORM,yNuSTORM = np.genfromtxt('digitized/Datasets/NuSTORM_dis_99CL.txt', unpack=True, skip_header=0)
		# ax.plot(xNuSTORM,yNuSTORM, c="green", linestyle="-", lw="2.0", label=r'arXiv:1402.5250 99\% C.L.')
			
		ax.legend(loc='upper left', shadow=0, fontsize=fsize, frameon=0)
		# ax.text(0.0013, 60,r"$\ell_{p} = N \times 226$ m", fontsize=fsize)
		# ax.text(0.0013, 35,r"$\Gamma_{\mu} = 3 \times 10^{-19}$ GeV", fontsize=fsize)
		# ax.text(0.0013, 20.7,r"$v_{\mu} = 1.0$", fontsize=fsize)
		# leg1 = ax.legend(handles=[l1,lSBN],loc='lower right',  shadow=0, fontsize=fsize, frameon=0)
	 	# plt.gca().add_artist(leg1)
		leg2 = ax.legend(handles=[lIC, lSK, lMINOSp],loc='upper center', ncol=2,  shadow=0, fontsize=fsize*0.8, frameon=0)

		plt.xlim(10**(-0.5),10**(6))
		plt.ylim(10**(-3),np.sqrt(0.25))


	
		# ax.text(8.5e-3,0.6, r'FD stat', rotation=-20, fontsize=fsize)
		# ax.text(1.3e-3,17.25, r'ND stat', rotation=-20, fontsize=fsize)
		# ax.text(0.22,0.42, r'FD ', rotation=-20, fontsize=fsize)
		# ax.text(0.37,23.03,r'ND', rotation=-20, fontsize=fsize)


	# leg1 = ax.legend(handles=[l1,l2,l3],loc='upper right',  shadow=0, fontsize=fsize, frameon=0)
	# plt.gca().add_artist(leg1)
	# leg2 = ax.legend(handles=[l4,l5,l6],loc='lower left',  shadow=0, fontsize=fsize, frameon=0)

	ax.set_xscale("log")
	ax.set_yscale("log")

	# ax.text(56583,np.sqrt(4.32e-12),r'$|U_{\mu 4}| = 1$', fontsize=fsize*1.2)

	plt.ylabel(theta)
	plt.xlabel(r'$m_4$ [eV]')
	
	plt.show()
	fig.savefig('plots/Bounds_'+S+'.pdf')




plot_comparison("dis",r'$|U_{\mu 4}|^2$')
plot_comparison("edis",r'$|U_{e 4}|^2$')




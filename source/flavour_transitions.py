import numpy as np
from numpy import ma
import scipy
from pathlib import Path
local_dir = Path(__file__).parent

from source import *
from source.const import *

#############################################
# get neutrino fluxes from AGSS09 solar model
data = np.genfromtxt(f'{local_dir}/../SSM/agss09_nu_fluxes.dat')
r = data[:,0]
d = data[:,2]
f = data[:,8]
fraction = scipy.interpolate.interp1d(r,f,bounds_error=False,fill_value=0)
density = scipy.interpolate.interp1d(r,10**(d)*NAvo*cmINV_to_GeV**3*1e27,bounds_error=False,fill_value=0.0)
rnew = np.linspace(0,0.5,1000)
flux_weighted_integral = np.sum(fraction(rnew)*(rnew[1]-rnew[0])*density(rnew))

#############################
# approx -- > uses pproximation with V_NC --> 0, leading order in all theta_i4, and taking Utau4 --> 0
def Usi_matter(Enu, channel, r, USQR=[0,0,0], approx=False):
	
	Ue4SQR, Umu4SQR, Utau4SQR = USQR
	Ue4  = np.sqrt(Ue4SQR)
	Umu4 = np.sqrt(Umu4SQR)
	Utau4 = np.sqrt(Utau4SQR)

	Enu = Enu*1e6 # in eV

	# matter potentials -- reverse sign for antineutrinos
	VCC = np.sqrt(2)*Gf*1e-18*density(r) * (1 - 2*is_antineutrino(channel)) # eV
	VNC = -0.25*VCC

	rx = ma.masked_array(data=-VNC/VCC, mask= ~(VCC != 0), fill_value=0.0)
	rx = rx.filled()

	ACC = 2*Enu*VCC # solar_core_Ne  # eV^2
	vx = ACC/dmSQR21

	## from mixing elements to rotation angles
	s14 = Ue4
	c14 = np.sqrt(1-s14**2)
	
	s24 = np.sqrt(2)*(Umu4*c23 - Utau4*s23)
	s24 /= np.sqrt(2 - 2*Ue4SQR - Umu4SQR - Utau4SQR + (Umu4SQR - Utau4SQR)*np.cos(2*theta23) - 2*Umu4*Utau4*np.sin(2*theta23))
	c24 = np.sqrt(1-s24**2)

	s34 = Utau4*c23 + Umu4*s23
	s34 /= np.sqrt(1-Ue4SQR)
	c34 = np.sqrt(1-s34**2)

	# aux parameters from 1105.1705
	alpha = c24*(s34*s13 - c34*s14*c13)
	beta  = -s24
	gamma = c13*c14

	if approx: 
		# theta_12 in matter
		theta12MATTER = 0.5 * np.arctan2( np.tan(2*theta12), (1.0 - ACC/dmSQR21/np.cos(2*theta12) )  )

		s12MATTER = np.sin(theta12MATTER)
		c12MATTER = np.cos(theta12MATTER)
		Us1m = Umu4*s12MATTER*c23+c12MATTER*(-Ue4*c13+Umu4*s13*s23) 
		Us2m = Umu4*c12MATTER*c23+s12MATTER*( Ue4*c13-Umu4*s13*s23) 
		Us3m = Ue4*s13+Umu4*c13*s23
	else:
		# theta_12 in matter
		num = np.sin(2*theta12) + 2*vx*rx*alpha*beta
		den = np.cos(2*theta12) - vx*gamma**2 - vx*rx*(alpha**2 - beta**2)
		theta12MATTER = 0.5 * np.arctan2(num, den)
		s12MATTER = np.sin(theta12MATTER)
		c12MATTER = np.cos(theta12MATTER)
		
		Us1m = s12MATTER*s24 + c12MATTER*c24*(s13*s34 - c13*c34*s14)
		Us2m = -c12MATTER*s24 + s12MATTER*c24*(s13*s34 - c13*c34*s14)
		Us3m = -c24*(c13*s34 + s13*c34*s14)
		
	return Us1m, Us2m, Us3m

def Pse_r(Enu, channel, r, USQR=[0,0,0], approx=False):

	## relevant matrix elements for solar transition
	Us1m, Us2m, Us3m = Usi_matter(Enu, channel, r, USQR=USQR, approx=approx)

	Ue1 = c12 * c13
	Ue2 = s12 * c13
	Ue3 = s13

	Ue4SQR, Umu4SQR, Utau4SQR = USQR

	Pall = Us1m**2*Ue1**2 + Us2m**2*Ue2**2 + Us3m**2*Ue3**2
	Pall /= (Ue4SQR+Umu4SQR+Utau4SQR) # normalized 

	return Pall


################################
# average over production region
def Usi_matter_avg(Enu, channel, USQR=[0,0,0], approx=False):
	
	Ue4SQR, Umu4SQR, Utau4SQR = USQR
	
	r=np.linspace(0,0.5,100)
	dr = r[1]-r[0]
	rgrid,egrid = np.meshgrid(r, Enu)

	UsiSQR_grid = np.array([fraction(rgrid)*density(rgrid)*Usi_matter(egrid, channel, rgrid, USQR=USQR, approx=approx)[i]**2 for i in [0,1,2]])
	
	# integrate over radius (last index)
	return np.sum(UsiSQR_grid, axis = -1)*dr/flux_weighted_integral/(Ue4SQR + Umu4SQR + Utau4SQR)

def Pse_avg(Enu, channel, USQR=[0,0,0], approx=False):
	
	r=np.linspace(0,0.5,100)
	dr = r[1]-r[0]
	rgrid,egrid = np.meshgrid(r, Enu)

	Pse_grid = fraction(rgrid)*density(rgrid)*Pse_r(egrid, channel, r, USQR=USQR, approx=approx)
	
	# integrate over radius (last index)
	return np.sum(Pse_grid, axis = -1)*dr/flux_weighted_integral


#############################################
# ## Get splines and use them for computation on the fly
spline_approx= np.load(f'{local_dir}/../prob_splines/Pse_nu_approx.npy',allow_pickle=True)
func_approx_nu = spline_approx.item()['interp']

spline= np.load(f'{local_dir}/../prob_splines/Pse_nu.npy',allow_pickle=True)
func_nu = spline.item()['interp']

spline_approx= np.load(f'{local_dir}/../prob_splines/Pse_nubar_approx.npy',allow_pickle=True)
func_approx_nubar = spline_approx.item()['interp']

spline= np.load(f'{local_dir}/../prob_splines/Pse_nubar.npy',allow_pickle=True)
func_nubar = spline.item()['interp']

def Pse_spline_nu(Enu, Ue4SQR, Umu4SQR):
	one = np.ones((np.size(Enu),))
	return func_nu(np.array([Ue4SQR*one,Umu4SQR*one,Enu]).T)

def Pse_approx_spline_nu(Enu, Ue4SQR, Umu4SQR):
	one = np.ones(np.size(Enu))
	return func_approx_nu(np.array([Ue4SQR*one,Umu4SQR*one,Enu]).T)

def Pse_spline_nubar(Enu, Ue4SQR, Umu4SQR):
	one = np.ones((np.size(Enu),))
	return func_nubar(np.array([Ue4SQR*one,Umu4SQR*one,Enu]).T)

def Pse_approx_spline_nubar(Enu, Ue4SQR, Umu4SQR):
	one = np.ones(np.size(Enu))
	return func_approx_nubar(np.array([Ue4SQR*one,Umu4SQR*one,Enu]).T)

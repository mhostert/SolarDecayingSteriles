import numpy as np
import numpy.ma as ma

import vegas
import gvar as gv

import scipy.stats
from scipy.stats import chi2
from scipy import interpolate
from source import *


def get_likelihood_in_ratio_of_masses(EXP, params, fluxfile, endpoint, xpoints=20, ypoints=30):

    LK = np.zeros((ypoints,xpoints))


    ############# 
    # 2D grid in Umu4 and Ue4 space
    UE4SQR =np.logspace(-3.5,-1.5,ypoints,endpoint=True)
    RATIOS =np.linspace(0.1,0.999,xpoints,endpoint=True)

    for j in range(np.size(RATIOS)):    
        # set ratio
        params.mBOSON = RATIOS[j]*params.m4
        # recompute
        beK, npK, backK, dK = rates.fill_bins(EXP,params,fluxfile,endpoint=endpoint)
        bK = const.get_centers(beK)

        dofK = np.size(dK)-2
        old_factorK = params.Ue4**2*(osc.Pse_spline_nubar(bK, params.Ue4**2, params.Umu4**2))

        for i in range(np.size(UE4SQR)):
            
            new_factorK = UE4SQR[i]*osc.Pse_spline_nubar(bK, UE4SQR[i], params.Umu4**2)
            np_newK = new_factorK/old_factorK*npK

            LK[i,j] = chi2_binned_rate(np_newK, backK, dK, [EXP.err_flux,EXP.err_back])

    print("(Lmin, dof) = ",np.min(LK), dofK)
    LK = LK - np.min(LK)

    return UE4SQR, RATIOS, LK


def get_likelihood(EXP, params, fluxfile, endpoint, NPOINTS =33):
    
    bK, npK, backK, dK = rates.fill_bins(EXP,params,fluxfile,endpoint=endpoint)
    bK = bK[:-1] + (bK[1:] - bK[:-1])/2
    LK = np.zeros((NPOINTS,NPOINTS))
    dofK = np.size(dK)-2

    old_factorK = params.Ue4**2*(osc.Pse_spline_nubar(bK, params.Ue4**2, params.Umu4**2))


    ############# 
    # 2D grid in Umu4 and Ue4 space
    UE4SQR =np.linspace(0,0.1,NPOINTS)
    UMU4SQR =np.linspace(0,0.020,NPOINTS)
    for i in range(np.size(UE4SQR)):
        for j in range(np.size(UMU4SQR)):
            new_factorK = UE4SQR[i]*osc.Pse_spline_nubar(bK, UE4SQR[i], UMU4SQR[j])
            np_newK = new_factorK/old_factorK*npK

            LK[j,i] = chi2_binned_rate(np_newK, backK, dK, [EXP.err_flux,EXP.err_back])

    print("(Lmin, dof) = ",np.min(LK), dofK)
    LK = LK - np.min(LK)
    return UE4SQR, UMU4SQR, LK

def chi2_binned_rate(NP_MC,back_MC,D,sys):
    err_flux = sys[0]
    err_back = sys[1]
    
    # mask = D>0
    # NP_MC = NP_MC[mask]
    # back_MC = back_MC[mask]
    # D = D[mask]
    
    dpoints = len(D)
    def chi2bin(nuis):
        alpha=nuis[:dpoints]
        beta = nuis[dpoints:]

        mu = NP_MC*(1+alpha) + back_MC*(1+beta)
        return 2*np.sum(mu - D + safe_log(D, mu) ) + np.sum(alpha**2/(err_flux**2)) + np.sum(beta**2 /(err_back**2))
        # return 2*np.sum(   (NP_MC*(1+beta[0]) + back_MC*(1+beta[1]) - D)**2/(NP_MC*(1+beta[0]) + back_MC*(1+beta[1]))) + beta[0]**2/(err_flux**2) + beta[1]**2 /(err_back**2)

    # max std devs allowed for nuissance params -- hard wall
    # max_std=100
    # bounds = np.ones((dpoints*2,2))
    # bounds[:dpoints,0] = -max_std*err_flux
    # bounds[:dpoints,1] = max_std*err_flux
    # bounds[dpoints:,0] = -max_std*err_back
    # bounds[dpoints:,1] = max_std*err_back
    # res = scipy.optimize.minimize(chi2bin, np.zeros(np.size(D)*2), bounds=bounds)
    
    res = scipy.optimize.minimize(chi2bin, np.zeros(np.size(D)*2))
    
    return chi2bin(res.x)

def chi2_total_rate(NP_MC,back_MC,D,sys):
    err_flux = sys[0]
    err_back = sys[1]
    
    def chi2bin(nuis):
        alpha=nuis[0]
        beta = nuis[1]
        mu = NP_MC*(1+alpha) + back_MC*(1+beta)
        return 2*np.sum((mu - D + safe_log(D, mu))) + alpha**2/(err_flux**2) + beta**2 /(err_back**2)
    
    res = scipy.optimize.minimize(chi2bin, [0.0,0.0])
    
    return chi2bin(res.x)


def safe_log(di,xi):
    mask = (di*xi>0)
    d = np.empty_like(di*xi)
    d[mask] = di[mask]*np.log(di[mask]/xi[mask])
    d[~mask] = di[~mask]*1e100
    return d



def get_model_indep_bound_flux(PRED, BKG, DATA, err_flux=0.1, err_bkg=0.1, DOF=1, CL=0.9):
    fluxes = np.logspace(-1,5, 100)
    my_chi2_table = np.empty_like(fluxes)

    for i in range(len(fluxes)):
        c = chi2_total_rate(fluxes[i]*PRED, BKG, DATA, [err_flux,err_bkg])
        my_chi2_table[i] = c

    my_chi2_table = my_chi2_table - np.min(my_chi2_table)
    
    ibound = const.find_nearest_arg(my_chi2_table, chi2.ppf(CL, DOF))
    
    return fluxes[ibound]


# feldman-cousins with b=0
def get_model_indep_bound_flux_FC(PRED, BKG, DATA, err_flux=0.1, err_bkg=0.1, DOF=1, CL=0.9):
    fluxes = np.logspace(-1,5, 10000)
    rate_table = np.empty_like(fluxes)
    
    if CL == 0.9:
        upper_bound = CL_90_upper
    elif CL == 0.99:
        upper_bound = CL_99_upper

    for i in range(len(fluxes)):
        c = fluxes[i]*PRED
        rate_table[i] = c
    

    ibound = const.find_nearest_arg(rate_table, upper_bound(DATA, BKG)) # b = 0
    
    return fluxes[ibound]


###  deprecated deprecated deprecated
def chi2_model_independent(exp,params,fluxfile):
    bin_e = exp.Enu_bin_e
    fluxlimit = exp.fluxlimit

    ###########
    # SET BINS TO BE THE EXPERIMENTAL BINS
    bins = bin_e # bin edges

    ###########
    # NUMU FLUX
    flux = fluxes.get_neutrino_flux(fluxfile)

    ############
    # NUE/BAR XS
    xsec = lambda x : np.zeros(np.size(x))
    xsecbar = lambda x : np.ones(np.size(x))

    ############
    # efficiencies
    enu_eff= bins
    eff= np.ones((np.size(bins)-1))

    ############
    # number of events
    NCASCADE, dNCASCADE = rates.RATES_dN_HNL_CASCADE_NU_NUBAR(\
                                                flux=flux,\
                                                xsec=xsec,\
                                                xsecbar=xsecbar,\
                                                dim=3,\
                                                enumin=0,\
                                                enumax=16.8,\
                                                params=params,\
                                                bins=bins,\
                                                PRINT=False,\
                                                enu_eff=enu_eff,\
                                                eff=eff,
                                                smearing_function=lambda x: x)

    chi2 = np.sum( 2.71 * (dNCASCADE)**2/fluxlimit**2 )
    return chi2, chi2/(np.size(bins)-2)




##############################
# Feldman-Cousins tables
CL_90 = np.array([[ [0.00, 2.44], [0.00, 1.94],  [0.00, 1.61],  [0.00, 1.33],  [0.00, 1.26], [0.00, 1.18], [0.00, 1.08], [0.00, 1.06], [0.00, 1.01], [0.00, 0.98]],
        [ [0.11, 4.36], [0.00, 3.86],  [0.00, 3.36],  [0.00, 2.91],  [0.00, 2.53], [0.00, 2.19], [0.00, 1.88], [0.00, 1.59], [0.00, 1.39], [0.00, 1.22]],
        [ [0.53, 5.91], [0.03, 5.41],  [0.00, 4.91],  [0.00, 4.41],  [0.00, 3.91], [0.00, 3.45], [0.00, 3.04], [0.00, 2.67], [0.00, 2.33], [0.00, 1.73]],
        [ [1.10, 7.42], [0.60, 6.92],  [0.10, 6.42],  [0.00, 5.92],  [0.00, 5.42], [0.00, 4.92], [0.00, 4.42], [0.00, 3.95], [0.00, 3.53], [0.00, 2.78]],
        [ [1.47, 8.60], [1.17, 8.10],  [0.74, 7.60],  [0.24, 7.10],  [0.00, 6.60], [0.00, 6.10], [0.00, 5.60], [0.00, 5.10], [0.00, 4.60], [0.00, 3.60]]])

n0 = np.arange(0, 5, 1)
b = [0.0,0.5,1.0,1.5,2.0,2.5,3.0,3.5,4.0,5.0]
nn, bb = np.meshgrid(n0, b)
bb
CL_90_lower = interpolate.interp2d(n0, b, CL_90[:,:,0].T, kind='linear')
CL_90_upper = interpolate.interp2d(n0, b, CL_90[:,:,1].T, kind='linear')

##############################
# Feldman-Cousins tables
CL_99_first=[[[0.00, 4.74], [0.00, 4.24], [0.00, 3.80], [0.00, 3.50], [0.00, 3.26], [0.00, 3.26], [0.00, 3.05], [0.00, 3.05], [0.00, 2.98], [0.00, 2.94]],
[[0.01, 6.91], [0.00, 6.41], [0.00, 5.91], [0.00, 5.41], [0.00, 4.91], [0.00, 4.48], [0.00, 4.14], [0.00, 4.09], [0.00, 3.89], [0.00, 3.59]],
[[0.15, 8.71], [0.00, 8.21], [0.00, 7.71], [0.00, 7.21], [0.00, 6.71], [0.00, 6.24], [0.00, 5.82], [0.00, 5.42], [0.00, 5.06], [0.00, 4.37]],
[[0.44,10.47], [0.00, 9.97], [0.00, 9.47], [0.00, 8.97], [0.00, 8.47], [0.00, 7.97], [0.00, 7.47], [0.00, 6.97], [0.00, 6.47], [0.00, 5.57]],
[[0.82,12.23], [0.32,11.73], [0.00,11.23], [0.00,10.73], [0.00,10.23], [0.00, 9.73], [0.00, 9.23], [0.00, 8.73], [0.00, 8.23], [0.00, 7.30]],
[[1.28,13.75], [0.78,13.25], [0.28,12.75], [0.00,12.25], [0.00,11.75], [0.00,11.25], [0.00,10.75], [0.00,10.25], [0.00, 9.75], [0.00, 8.75]],
[[1.79,15.27], [1.29,14.77], [0.79,14.27], [0.29,13.77], [0.00,13.27], [0.00,12.77], [0.00,12.27], [0.00,11.77], [0.00,11.27], [0.00,10.27]],
[[2.33,16.77], [1.83,16.27], [1.33,15.77], [0.83,15.27], [0.33,14.77], [0.00,14.27], [0.00,13.77], [0.00,13.27], [0.00,12.77], [0.00,11.77]],
[[2.91,18.27], [2.41,17.77], [1.91,17.27], [1.41,16.77], [0.91,16.27], [0.41,15.77], [0.00,15.27], [0.00,14.77], [0.00,14.27], [0.00,13.27]],
[[3.31,19.46], [3.00,18.96], [2.51,18.46], [2.01,17.96], [1.51,17.46], [1.01,16.96], [0.51,16.46], [0.01,15.96], [0.00,15.46], [0.00,14.46]],
[[3.68,20.83], [3.37,20.33], [3.07,19.83], [2.63,19.33], [2.13,18.83], [1.63,18.33], [1.13,17.83], [0.63,17.33], [0.13,16.83], [0.00,15.83]],
[[4.05,22.31], [3.73,21.81], [3.43,21.31], [3.14,20.81], [2.77,20.31], [2.27,19.81], [1.77,19.31], [1.27,18.81], [0.77,18.31], [0.00,17.31]],
[[4.41,23.80], [4.10,23.30], [3.80,22.80], [3.50,22.30], [3.22,21.80], [2.93,21.30], [2.43,20.80], [1.93,20.30], [1.43,19.80], [0.43,18.80]],
[[5.83,24.92], [5.33,24.42], [4.83,23.92], [4.33,23.42], [3.83,22.92], [3.33,22.42], [3.02,21.92], [2.60,21.42], [2.10,20.92], [1.10,19.92]],
[[6.31,26.33], [5.81,25.83], [5.31,25.33], [4.86,24.83], [4.46,24.33], [4.10,23.83], [3.67,23.33], [3.17,22.83], [2.78,22.33], [1.78,21.33]],
[[6.70,27.81], [6.20,27.31], [5.70,26.81], [5.24,26.31], [4.84,25.81], [4.48,25.31], [4.14,24.81], [3.82,24.31], [3.42,23.81], [2.48,22.81]],
[[7.76,28.85], [7.26,28.35], [6.76,27.85], [6.26,27.35], [5.76,26.85], [5.26,26.35], [4.76,25.85], [4.26,25.35], [3.89,24.85], [3.15,23.85]],
[[8.32,30.33], [7.82,29.83], [7.32,29.33], [6.82,28.83], [6.32,28.33], [5.85,27.83], [5.42,27.33], [5.03,26.83], [4.67,26.33], [3.73,25.33]],
[[8.71,31.81], [8.21,31.31], [7.71,30.81], [7.21,30.31], [6.71,29.81], [6.24,29.31], [5.82,28.81], [5.42,28.31], [5.06,27.81], [4.37,26.81]],
[[9.88,32.85], [9.38,32.35], [8.88,31.85], [8.38,31.35], [7.88,30.85], [7.38,30.35], [6.88,29.85], [6.40,29.35], [5.97,28.85], [5.01,27.85]],
[[10.28,34.32], [9.78,33.82], [9.28,33.32], [8.78,32.82], [8.28,32.32], [7.78,31.82], [7.28,31.32], [6.81,30.82], [6.37,30.32], [5.57,29.32]]]
CL_99_second=[[[0.00, 2.91], [0.00, 2.90],[0.00, 2.89], [0.00, 2.88], [0.00, 2.88], [0.00, 2.87], [0.00, 2.87], [0.00, 2.86], [0.00, 2.86], [0.00, 2.86]],
[[0.00, 3.42], [0.00, 3.31],[0.00, 3.21], [0.00, 3.18], [0.00, 3.15], [0.00, 3.11], [0.00, 3.09], [0.00, 3.07], [0.00, 3.06], [0.00, 3.03]],
[[0.00, 4.13], [0.00, 3.89],[0.00, 3.70], [0.00, 3.56], [0.00, 3.44], [0.00, 3.39], [0.00, 3.35], [0.00, 3.32], [0.00, 3.26], [0.00, 3.23]],
[[0.00, 5.25], [0.00, 4.59],[0.00, 4.35], [0.00, 4.06], [0.00, 3.89], [0.00, 3.77], [0.00, 3.65], [0.00, 3.56], [0.00, 3.51], [0.00, 3.47]],
[[0.00, 6.47], [0.00, 5.73],[0.00, 5.04], [0.00, 4.79], [0.00, 4.39], [0.00, 4.17], [0.00, 4.02], [0.00, 3.91], [0.00, 3.82], [0.00, 3.74]],
[[0.00, 7.81], [0.00, 6.97],[0.00, 6.21], [0.00, 5.50], [0.00, 5.17], [0.00, 4.67], [0.00, 4.42], [0.00, 4.24], [0.00, 4.11], [0.00, 4.01]],
[[0.00, 9.27], [0.00, 8.32],[0.00, 7.47], [0.00, 6.68], [0.00, 5.96], [0.00, 5.46], [0.00, 5.05], [0.00, 4.83], [0.00, 4.63], [0.00, 4.44]],
[[0.00,10.77], [0.00, 9.77],[0.00, 8.82], [0.00, 7.95], [0.00, 7.16], [0.00, 6.42], [0.00, 5.73], [0.00, 5.48], [0.00, 5.12], [0.00, 4.82]],
[[0.00,12.27], [0.00,11.27],[0.00,10.27], [0.00, 9.31], [0.00, 8.44], [0.00, 7.63], [0.00, 6.88], [0.00, 6.18], [0.00, 5.83], [0.00, 5.29]],
[[0.00,13.46], [0.00,12.46],[0.00,11.46], [0.00,10.46], [0.00, 9.46], [0.00, 8.50], [0.00, 7.69], [0.00, 7.34], [0.00, 6.62], [0.00, 5.95]],
[[0.00,14.83], [0.00,13.83],[0.00,12.83], [0.00,11.83], [0.00,10.83], [0.00, 9.87], [0.00, 8.98], [0.00, 8.16], [0.00, 7.39], [0.00, 7.07]],
[[0.00,16.31], [0.00,15.31],[0.00,14.31], [0.00,13.31], [0.00,12.31], [0.00,11.31], [0.00,10.35], [0.00, 9.46], [0.00, 8.63], [0.00, 7.84]],
[[0.00,17.80], [0.00,16.80],[0.00,15.80], [0.00,14.80], [0.00,13.80], [0.00,12.80], [0.00,11.80], [0.00,10.83], [0.00, 9.94], [0.00, 9.09]],
[[0.10,18.92], [0.00,17.92],[0.00,16.92], [0.00,15.92], [0.00,14.92], [0.00,13.92], [0.00,12.92], [0.00,11.92], [0.00,10.92], [0.00, 9.98]],
[[0.78,20.33], [0.00,19.33],[0.00,18.33], [0.00,17.33], [0.00,16.33], [0.00,15.33], [0.00,14.33], [0.00,13.33], [0.00,12.33], [0.00,11.36]],
[[1.48,21.81], [0.48,20.81],[0.00,19.81], [0.00,18.81], [0.00,17.81], [0.00,16.81], [0.00,15.81], [0.00,14.81], [0.00,13.81], [0.00,12.81]],
[[2.18,22.85], [1.18,21.85],[0.18,20.85], [0.00,19.85], [0.00,18.85], [0.00,17.85], [0.00,16.85], [0.00,15.85], [0.00,14.85], [0.00,13.85]],
[[2.89,24.33], [1.89,23.33],[0.89,22.33], [0.00,21.33], [0.00,20.33], [0.00,19.33], [0.00,18.33], [0.00,17.33], [0.00,16.33], [0.00,15.33]],
[[3.53,25.81], [2.62,24.81],[1.62,23.81], [0.62,22.81], [0.00,21.81], [0.00,20.81], [0.00,19.81], [0.00,18.81], [0.00,17.81], [0.00,16.81]],
[[4.13,26.85], [3.31,25.85],[2.35,24.85], [1.35,23.85], [0.35,22.85], [0.00,21.85], [0.00,20.85], [0.00,19.85], [0.00,18.85], [0.00,17.85]],
[[4.86,28.32], [3.93,27.32],[3.08,26.32], [2.08,25.32], [1.08,24.32], [0.08,23.32], [0.00,22.32], [0.00,21.32], [0.00,20.32], [0.00,19.32]]]
CL_99 = np.concatenate( (np.array(CL_99_first),np.array(CL_99_second)) ,axis=1)
n0 = np.arange(0, 21, 1)
b = [0.0,0.5,1.0,1.5,2.0,2.5,3.0,3.5,4.0,5.0]+[i for i in range(6,16)]
# nn, bb = np.meshgrid(n0, b)
CL_99_lower = interpolate.interp2d(n0, b, CL_99[:,:,0].T, kind='linear')
CL_99_upper = interpolate.interp2d(n0, b, CL_99[:,:,1].T, kind='linear')


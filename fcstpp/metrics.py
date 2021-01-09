
import numba as nb
import numpy as np

from fcstpp.utils import *

@nb.njit(fastmath=True)
def CRPS_1d(y_true, y_ens):
    '''
    Given one-dimensional ensemble forecast, compute its CRPS and corresponded two-term decomposition.
    
    CRPS, MAE, pairwise_abs_diff = CRPS_1d(y_true, y_ens)
    
    Grimit, E.P., Gneiting, T., Berrocal, V.J. and Johnson, N.A., 2006. The continuous ranked probability score 
    for circular variables and its application to mesoscale forecast ensemble verification. Quarterly Journal of 
    the Royal Meteorological Society, 132(621C), pp.2925-2942.
    
    Input
    ----------
        y_true: a numpy array with shape=(time, grids) and represents the (observed) truth 
        y_pred: a numpy array with shape=(time, ensemble_members, grids), represents the ensemble forecast
    
    Output
    ----------
        CRPS: the continuous ranked probability score, shape=(time, grids)
        MAE: mean absolute error
        SPREAD: pairwise absolute difference among ensemble members (not the spread)
        
    '''
    N_day, EN, N_grids = y_ens.shape
    M = 2*EN*EN
    
    # allocate outputs
    MAE = np.empty((N_day, N_grids),); MAE[...] = np.nan
    SPREAD = np.empty((N_day, N_grids),); SPREAD[...] = np.nan
    
    # loop over grid points
    for n in range(N_grids):
        # loop over days
        for day in range(N_day):
            # calc MAE
            MAE[day, n] = np.mean(np.abs(y_true[day, n]-y_ens[day, :, n]))
            # calc SPREAD
            spread_temp = 0
            for en1 in range(EN):
                for en2 in range(EN):
                    spread_temp += np.abs(y_ens[day, en1, n]-y_ens[day, en2, n])
            SPREAD[day, n] = spread_temp/M
            
    CRPS = MAE-SPREAD
    
    return CRPS, MAE, SPREAD

@nb.njit(fastmath=True)
def CRPS_2d(y_true, y_ens, land_mask='none'):
    
    '''
    Given two-dimensional ensemble forecast, compute its CRPS and corresponded two-term decomposition.
    
    CRPS, MAE, pairwise_abs_diff = CRPS_2d(y_true, y_ens, land_mask='none')
    
    Grimit, E.P., Gneiting, T., Berrocal, V.J. and Johnson, N.A., 2006. The continuous ranked probability score 
    for circular variables and its application to mesoscale forecast ensemble verification. Quarterly Journal of 
    the Royal Meteorological Society, 132(621C), pp.2925-2942.
    
    Input
    ----------
        y_true: a numpy array with shape=(time, gridx, gridy) and represents the (observed) truth.
        y_pred: a numpy array with shape=(time, ensemble_members, gridx, gridy), represents the ensemble forecast.
        land_mask: a numpy array with shape=(gridx, gridy). 
                   True elements indicate where CRPS will be computed.
                   Positions of False elements will be filled with np.nan
                   *if land_mask='none', all grid points will participate.
    
    Output
    ----------
        CRPS: the continuous ranked probability score, shape=(time, grids)
        MAE: mean absolute error
        SPREAD: pairwise absolute difference among ensemble members (not the spread)
        
    '''
    
    N_day, EN, Nx, Ny = y_ens.shape
    M = 2*EN*EN
    
    if land_mask == 'none':
        land_mask = np.ones((Nx, Ny)) > 0
    
    # allocate outputs
    MAE = np.empty((N_day, Nx, Ny),); MAE[...] = np.nan
    SPREAD = np.empty((N_day, Nx, Ny),); SPREAD[...] = np.nan
    
    # loop over grid points
    for i in range(Nx):
        for j in range(Ny):
            if land_mask[i, j]:
                # loop over days
                for day in range(N_day):
                    # calc MAE
                    MAE[day, i, j] = np.mean(np.abs(y_true[day, i, j]-y_ens[day, :, i, j]))
                    # calc SPREAD
                    spread_temp = 0
                    for en1 in range(EN):
                        for en2 in range(EN):
                            spread_temp += np.abs(y_ens[day, en1, i, j]-y_ens[day, en2, i, j])
                    SPREAD[day, i, j] = spread_temp/M
    CRPS = MAE-SPREAD

    return CRPS, MAE, SPREAD

@nb.njit(fastmath=True)
def BS_binary_1d(y_true, y_ens):
    '''
    
    '''
    
    N_days, EN, N_grids = y_ens.shape
    
    # allocation
    BS = np.empty((N_days, N_grids))

    # loop over initialization days
    for day in range(N_days):

        ens_vector = y_ens[day, ...]
        obs_vector = y_true[day, :]

        for n in range(N_grids):
            BS[day, n] = (obs_vector[n] - np.sum(ens_vector[:, n])/EN)**2

    return BS

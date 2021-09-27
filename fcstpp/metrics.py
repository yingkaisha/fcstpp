
import numba as nb
import numpy as np
from sklearn.metrics import confusion_matrix

from fcstpp.utils import *


def ETS(TRUE, PRED):
    '''
    Computing Equitable Threat Score (ETS) from binary input and target.
    '''
    TN, FP, FN, TP = confusion_matrix(TRUE, PRED).ravel()
    TP_rnd = (TP+FN)*(TP+FP)/(TN+FP+FN+TP)
    return (TP-TP_rnd)/(TP+FN+FP-TP_rnd)

def freq_bias(TRUE, PRED):
    '''
    Computing frequency bias from binary input and target.
    '''
    TN, FP, FN, TP = confusion_matrix(TRUE, PRED).ravel()
    return (TP+FP)/(TP+FN)

def PIT_nan(fcst, obs, q_bins):
    '''
    Probability Integral Transform (PIT) of observations based on forecast
    '''
    obs = obs[~np.isnan(obs)]
    
    # CDF_fcst
    cdf_fcst = np.quantile(fcst, q_bins)
    
    # transforming obs to CDF_fcst 
    n_obs = np.searchsorted(cdf_fcst, obs)
    # an uniform distributed random variale
    p_obs = n_obs/len(q_bins)
    # estimate CDF_fcst(obs)
    p_bins = np.quantile(p_obs, q_bins)
    
    return p_bins

#@nb.njit()
def CRPS_1d_from_quantiles(q_bins, CDFs, y_true):
    '''
    (experimental)
    Given quantile bins and values, computing CRPS from determinstic obs.
    
    Input
    ----------
        q_bins: quantile bins. `shape=(num_bins,)`.
        CDFs: quantile values corresponded to the `q_bins`.
              `shape=(num_bins, grid_points)`.
        y_true: determinstic true values. `shape=(obs_time, grid_points)`.
        
    Output
    ----------
        CRPS
        
    * `y_true` is 2-d. That said, one CDF paired for multiple obs.
      This is commonly applied for climatology CDFs vs. real-time obs. 
    
    '''
    
    L = len(q_bins)-1
    H_func = np.zeros((L+1,))
    d_bins = q_bins[1] - q_bins[0]
    
    N_days, N_grids = y_true.shape
    
    CRPS = np.empty((N_days, N_grids))
    
    for day in range(N_days):
        for n in range(N_grids):
            
            cdf = CDFs[:, n]
            obs = y_true[day, n]    
            step = np.searchsorted(cdf, obs)
            if step > L: step = L
                
            H_func[:] = 0.0
            H_func[step:] = 1.0
            
            CRPS[day, n] = np.trapz((q_bins-H_func)**2, x=cdf)
            #np.sum(np.diff(cdf)*(np.abs(q_bins-H_func)[:-1]))
    
    return CRPS

@nb.njit()
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

@nb.njit()
def CRPS_2d(y_true, y_ens, land_mask=None):
    
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
    
    if land_mask is None:
        land_mask_ = np.ones((Nx, Ny)) > 0
    else:
        land_mask_ = land_mask
    
    # allocate outputs
    MAE = np.empty((N_day, Nx, Ny),); MAE[...] = np.nan
    SPREAD = np.empty((N_day, Nx, Ny),); SPREAD[...] = np.nan
    
    # loop over grid points
    for i in range(Nx):
        for j in range(Ny):
            if land_mask_[i, j]:
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

@nb.njit()
def CRPS_1d_nan(y_true, y_ens):
    '''
    Given one-dimensional ensemble forecast, compute its CRPS and corresponded two-term decomposition.
    np.nan will not propagate.
    
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
            # if obs is nan, then mark result as nan
            if np.isnan(y_true[day, n]):
                MAE[day, n] = np.nan
                SPREAD[day, n] = np.nan
            else:
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



@nb.njit()
def BS_binary_1d(y_true, y_ens):
    '''
    Brier Score.
    
    BS_binary_1d(y_true, y_ens)
    
    ----------
    Hamill, T.M. and Juras, J., 2006. Measuring forecast skill: Is it real skill 
    or is it the varying climatology?. Quarterly Journal of the Royal Meteorological Society: 
    A journal of the atmospheric sciences, applied meteorology and physical oceanography, 132(621C), pp.2905-2923.
    
    Input
    ----------
        y_true: determinstic and binary true values. `shape=(obs_time, grid_points)`.
        y_ens: ensemble forecast. `shape=(time, ensemble_memeber, gird_points)`.
        
    Output
    ----------
        BS: Brier Score as described in Hamill and Juras (2006). 
        i.e., not scaled by `ensemble_memeber`, so can be applied for spatial-averaged analysis.
    
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

@nb.njit()
def BS_binary_1d_nan(y_true, y_ens):
    '''
    Brier Score. np.nan will not propagate.
    
    BS_binary_1d_nan(y_true, y_ens)
    
    ----------
    Hamill, T.M. and Juras, J., 2006. Measuring forecast skill: Is it real skill 
    or is it the varying climatology?. Quarterly Journal of the Royal Meteorological Society: 
    A journal of the atmospheric sciences, applied meteorology and physical oceanography, 132(621C), pp.2905-2923.
    
    Input
    ----------
        y_true: determinstic and binary true values. `shape=(obs_time, grid_points)`.
        y_ens: ensemble forecast. `shape=(time, ensemble_memeber, gird_points)`.
        
    Output
    ----------
        BS: Brier Score as described in Hamill and Juras (2006). 
        i.e., not scaled by `ensemble_memeber`, so can be applied for spatial-averaged analysis.
    
    '''
    
    N_days, EN, N_grids = y_ens.shape
    
    # allocation
    BS = np.empty((N_days, N_grids))

    # loop over initialization days
    for day in range(N_days):

        ens_vector = y_ens[day, ...]
        obs_vector = y_true[day, :]

        for n in range(N_grids):
            if np.isnan(obs_vector[n]):
                BS[day, n] = np.nan
            else:
                BS[day, n] = (obs_vector[n] - np.sum(ens_vector[:, n])/EN)**2

    return BS


import numba as nb
import numpy as np

@nb.njit()
def _simple_max(a, b):
    if a >= b:
        return a
    else:
        return b

@nb.njit()
def _simple_min(a, b):
    if a <= b:
        return a
    else:
        return b
    
    
@nb.njit()
def schaake_shuffle(fcst, traj):
    '''
    Schaake shuffle.
    
    Clark, M., Gangopadhyay, S., Hay, L., Rajagopalan, B. and Wilby, R., 2004. 
    The Schaake shuffle: A method for reconstructing space–time variability in forecasted 
    precipitation and temperature fields. Journal of Hydrometeorology, 5(1), pp.243-262.
    
    Input
    ----------
        fcst: ensemble forecasts. `shape=(ensemb_members, lead_time, grid_points)`.
        traj: trajectories. `shape=(history_time, lead_time, grid_points)`
              
              
    Output
    ----------
        output: shuffled ensemble forecasts. `shape=(ensemb_members, lead_time, grid_points)`.
        
    * number of trajectories and ensemble memebers must match: `history_time == ensemb_members`.
    * number of forecast lead time must match: `fcst.shape == traj.shape`.
    
    '''
    num_traj, N_lead, N_grids = traj.shape
    
    output = np.empty((N_lead, num_traj, N_grids))
    
    for l in range(N_lead):
        for n in range(N_grids):
            
            temp_traj = traj[:, l, n]
            temp_fcst = fcst[l, :, n]
            
            reverse_b_func = np.searchsorted(np.sort(temp_traj), temp_traj)
            
            output[l, :, n] = np.sort(temp_fcst)[reverse_b_func]
    return output
    
    
@nb.njit()
def quantile_mapping_stencil(pred, cdf_pred, cdf_true, land_mask, rad=1):
    '''
    (experimental)
    Quantile mapping with stencil grid points.
    
    quantile_mapping_stencil(pred, cdf_pred, cdf_true, land_mask, rad=1)
    
    ----------
    Scheuerer, M. and Hamill, T.M., 2015. Statistical postprocessing of 
    ensemble precipitation forecasts by fitting censored, shifted gamma distributions. 
    Monthly Weather Review, 143(11), pp.4578-4596.
    
    Hamill, T.M., Engle, E., Myrick, D., Peroutka, M., Finan, C. and Scheuerer, M., 2017. 
    The US National Blend of Models for statistical postprocessing of probability of precipitation 
    and deterministic precipitation amount. Monthly Weather Review, 145(9), pp.3441-3463.
    
    Input
    ----------
        pred: ensemble forecasts. `shape=(ensemb_members, gridx, gridy)`.
        cdf_pred: quantile values of the forecast. `shape=(quantile_bins, gridx, gridy)`
        cdf_true: the same as `cdf_pred` for the analyzed condition.
        land_mask: boolean arrays with True for focused grid points (i.e., True for land grid point).
                   `shape=(gridx, gridy)`.
        rad: grid point radius of the stencil. `rad=1` means 3-by-3 stencils.
        
    Output
    ----------
        out: quantile mapped and enlarged forecast. `shape=(ensemble_members, folds, gridx, gridy)`
             e.g., 3-by-3 stencil yields nine-fold more mapped outputs.
             
    * Quantile mapping is based on linear interpolations of the quantile values. No extrapolations.
    * If a stencil contains quantiles of out-of-mask grid points, then that quantile is not applied, the
      result is filled with np.nan. this happens at edeging and coastal grid points.
      
    '''
    
    EN, Nx, Ny = pred.shape
    N_fold = (2*rad+1)**2
    out = np.empty((EN, N_fold, Nx, Ny,))
    out[...] = np.nan
    
    for i in range(Nx):
        for j in range(Ny):
            # loop over grid points
            if land_mask[i, j]:
                min_x = _simple_max(i-rad, 0)
                max_x = _simple_min(i+rad, Nx-1)
                min_y = _simple_max(j-rad, 0)
                max_y = _simple_min(j+rad, Ny-1)

                count = 0
                for ix in range(min_x, max_x+1):
                    for iy in range(min_y, max_y+1):
                        if land_mask[ix, iy]:
                            for en in range(EN):
                                out[en, count, i, j] = np.interp(pred[en, i, j], cdf_pred[:, ix, iy], cdf_true[:, ix, iy])
                            count += 1
    return out

@nb.njit()
def bootstrap_fill(data, expand_dim, land_mask, fillval=np.nan):
    '''
    Fill values with bootstrapped aggregation.
    
    bootstrap_fill(data, expand_dim, land_mask, fillval=np.nan)
    
    Input
    ----------
        data: a four dimensional array. `shape=(time, ensemble_members, gridx, gridy)`.
        expand_dim: dimensions of `ensemble_members` that need to be filled. 
                    If `expand_dim` == `ensemble_members` then the bootstraping is not applied.
        land_mask: boolean arrays with True for focused grid points (i.e., True for land grid point).
                   `shape=(gridx, gridy)`.
        fillval: fill values of the out-of-mask grid points.
        
    Output
    ----------
        out: bootstrapped data. `shape=(time, expand_dim, gridx, gridy)`
        
    '''
    
    N_days, EN, Nx, Ny = data.shape
    out = np.empty((N_days, expand_dim, Nx, Ny))
    out[...] = np.nan
    
    for day in range(N_days):
        
        for ix in range(Nx):
            for iy in range(Ny):
                if land_mask[ix, iy]:
                    
                    data_sub = data[day, :, ix, iy]
                    flag_nonnan = np.logical_not(np.isnan(data_sub))
                    temp_ = data_sub[flag_nonnan]
                    L = len(temp_)
                    
                    if L == 0:
                        out[day, :, ix, iy] = fillval
                    elif L == expand_dim:
                        out[day, :, ix, iy] = temp_
                    else:
                        ind_bagging = np.random.choice(L, size=expand_dim, replace=True)
                        out[day, :, ix, iy] = temp_[ind_bagging]
    return out



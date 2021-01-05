
import numba as nb
import numpy as np

@nb.njit(fastmath=True)
def simple_max(a, b):
    if a >= b:
        return a
    else:
        return b

@nb.njit(fastmath=True)
def simple_min(a, b):
    if a <= b:
        return a
    else:
        return b
    
@nb.njit(fastmath=True)
def quantile_mapping_stencil(pred, cdf_pred, cdf_true, land_mask, rad=1):
    '''
    pred = (en, grids)
    cdf = (quantile, grids)
    '''
    EN, Nx, Ny = pred.shape
    N_fold = (2*rad+1)**2
    out = np.empty((EN, N_fold, Nx, Ny,))
    out[...] = np.nan
    
    for i in range(Nx):
        for j in range(Ny):
            if land_mask[i, j]:
                min_x = simple_max(i-rad, 0)
                max_x = simple_min(i+rad, Nx-1)
                min_y = simple_max(j-rad, 0)
                max_y = simple_min(j+rad, Ny-1)

                count = 0
                for ix in range(min_x, max_x+1):
                    for iy in range(min_y, max_y+1):
                        if land_mask[ix, iy]:
                            for en in range(EN):
                                out[en, count, i, j] = np.interp(pred[en, i, j], cdf_pred[:, ix, iy], cdf_true[:, ix, iy])
                            count += 1
    return out

@nb.njit()
def bootstrap_fill(data, expand_dim, land_mask, fillval=999):
    
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
                        
                    else:
                        ind_bagging = np.random.choice(L, size=expand_dim, replace=True)
                        out[day, :, ix, iy] = temp_[ind_bagging]

    return out



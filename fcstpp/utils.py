import numba as nb
import numpy as np

@nb.njit(fastmath=True)
def window_slider_cycled_1d(ind, ind_total_length, ind_radius):
    '''
    window_slider_cycled_1d(ind, ind_total_length, ind_radius)
    
    Given an index, search its cyclied indices within a sliding window.
    
    Input
    ----------
        ind: the centered index of a sliding window
        ind_total_length: the total length of available indices
        ind_radius: the radius of sliding window
        
    Output
    ----------
        ind_out: a 1-d array of bool types with length equals to `ind_total_length` 
                 and True represents indices within a sliding window that 
                 centered on `ind`.
                 
                 Total number of True is expected to be equal to `2*ind_radius + 1`.
    
    '''
    
    ind_out = np.zeros((ind_total_length,), dtype=np.bool_)
    
    ind_right = ind + ind_radius + 1
    ind_left = ind - ind_radius
    
    if ind_left >= 0 and ind_right <= ind_total_length: 
        ind_out[ind_left:ind_right] = True
        
    elif ind_left < 0:
        ind_out[0:ind_right] = True
        ind_out[ind_left:] = True
    else:
        ind_diff = ind_right - ind_total_length
        ind_out[ind_left:] = True
        ind_out[:ind_diff] = True
        
    return ind_out

@nb.njit(fastmath=True)
def climate_subdaily_prob(climate_tuple, day_window, period=24):
    '''
    Given a tuple of historical daily binary observations, 
    compute climatological probabilities of each day with
    a day window.
    
    Input
    ----------
        climate_tuple: a tuple of daily observations, each tuple element is a
                       numpy array that has two dimensions of `(time, grids)`.
                       For example: 
                           climate_tuple = (data_2000, data_2001, data_2002,)
                           data_2000.shape = (366, 180*360)
                       
        day_window: for the climatological probabilities of each time index, 
                    how many neghbouring days will be applied.
                    nonzero day_window indicates an augmentation of 
                    2*(day_window*24/period)+1 fold more data.
                    
        period: number of hours per time dimension index.
                For example, `period=24` means daily, `period=3` means every 3 hr
        
    Output
    ----------
        prob_out: a two- or three dimensional array that contains probabilities values.
                  The first dimension of the output is provided on a leap year basis.
                  For example, len(prob_out) = 366, if period=24.
                  
        N_sample: The number of samples that applied on the calculation of `prob_out` elements

    '''
    
    # number of tuple elements
    L_clim = len(climate_tuple)
    
    # the index of Feb 29th 0 hour
    feb_29 = int((31+28)*24/period)
    delta_day = int(24/period)
    
    # number of times per leap year
    full_time_size = int(366*24/period)
    
    # size of time window
    window_ = int(day_window*24/period)
    
    # number of available values per time per grid point
    L_aug = int(2*window_+1)
    
    # number of grid points
    grid_shape = climate_tuple[0].shape[1:]
    
    # allocations
    
    temp_storage = np.empty((full_time_size,)+grid_shape)
    temp_storage[...] = 0
    temp_aug_num = np.empty((full_time_size,)+grid_shape)
    temp_aug_num[...] = 0

    # loop over historical years
    for y in range(L_clim):
        
        # get the current tuple element
        temp_data = climate_tuple[y]
        days = len(temp_data)
        
        # if it is a leap year
        flag_leap_year = days == full_time_size

        # loop over days of each historical year
        for ind_d in range(days):

            # subset observations based on the time window
            flag_pick = window_slider_cycled_1d(ind_d, days, window_)
            temp_daily = temp_data[flag_pick, ...]

            # np.nan handling
            for n in range(grid_shape[0]):
                for i_aug in range(L_aug):
                    flag_nan = np.isnan(temp_daily[i_aug, n])

                    if not flag_nan:
                        
                        if flag_leap_year or ind_d < feb_29:
                            temp_storage[ind_d, n] += temp_daily[i_aug, n]
                            temp_aug_num[ind_d, n] += 1
                        else:
                            # if it's not in leap year, skip Feb 29th
                            temp_storage[ind_d+delta_day, n] += temp_daily[i_aug, n]
                            temp_aug_num[ind_d+delta_day, n] += 1
                            
    # converting counts to probabilities                          
    for ind_d in range(full_time_size):
        for n in range(grid_shape[0]):
            if temp_aug_num[ind_d, n] > 0:
                temp_storage[ind_d, n] = temp_storage[ind_d, n]/temp_aug_num[ind_d, n]
            else:
                temp_storage[ind_d, n] = np.nan

    prob_out = temp_storage
    N_sample = temp_aug_num
    return prob_out, N_sample

def facet(Z, rad=1):
    '''
    Compute terrain facet based on elevation.
    Returned values are: {0, 1, 2, 3, 4, 5, 6, 7, 8}.
    Corresponed facet groups are: {"N", "NE", "E", "SE", "S", "SW", "W", "NW", "flat"}.
    
    facet(Z, rad=1)
    
    ----------
    Daly, C., Gibson, W.P., Taylor, G.H., Johnson, G.L. and Pasteris, P., 2002. 
    A knowledge-based approach to the statistical mapping of climate. Climate research, 22(2), pp.99-113.
    
    Gibson, W., Daly, C. and Taylor, G., 1997. 7.1 DERIVATION OF FACET GRIDS FOR USE WITH THE PRISM MODEL.
    
    Input
    ----------
        Z: elevation. `shape=(grid x, gridy)`.
        rad: the "rad" in Gibson et al. (1997). Larger is smoother.
             (spatially more consistent facet groups).
    
    Output
    ---------
        out: facet groups.
    
    '''
    dZy, dZx = np.gradient(Z)
    dZy = -1*dZy
    dZx = -1*dZx
    Z_to_deg = np.arctan2(dZx, dZy)/np.pi*180
    Z_to_deg[Z_to_deg<0] += 360
    Z_ind = np.round(Z_to_deg/45.0)

    thres = np.sqrt(dZy**2+dZx**2) < 0.1
    Z_ind[thres] = 8
    Z_ind = Z_ind.astype(int)

    return facet_group(Z_ind, rad=rad)

def facet_group(compass, rad):
    '''
    The Gibson et al. (1997) facet grouping scheme.
    '''
    thres = rad*4
    grid_shape = compass.shape
    compass_pad = np.pad(compass, 2, constant_values=999)
    
    out = np.empty(grid_shape)
    
    for i in range(grid_shape[0]):
        for j in range(grid_shape[1]):
            group = compass_pad[i-rad:i+rad+1, j-rad:j+rad+1].ravel()
            flag_clean = ~(group==999)
            if np.sum(flag_clean)<thres:
                out[i, j] = np.nan
            else:
                group_clean = group[flag_clean]
                out[i, j] = Gibbs_rule(group_clean)
    return out
            
def adjacent(x1, x2):
    diffx = np.abs(x1 - x2)
    return np.min(np.array([diffx, np.abs(diffx+8), np.abs(diffx-8)]))

def sum_adjacent(counts, n0):
    n0_left = n0-1
    if n0_left < 0:
        n0_left += 8
    n0_right = n0+1
    if n0_right > 7:
        n0_right -= 8
    return np.max(np.array([counts[n0]+counts[n0_left], counts[n0]+counts[n0_right]])) 

def Gibbs_rule(compass_vec):
    '''
    The Gibbs et al. (1997) facet grouping decision tree.
    '''
    L = len(compass_vec)
    counts = np.bincount(compass_vec, minlength=9)
    count_sort = np.argsort(counts)[::-1]
    
    no0 = count_sort[0]
    no1 = count_sort[1]
    no2 = count_sort[2]
    no3 = count_sort[3]
    
    num_no0 = counts[no0]
    num_no1 = counts[no1]
    num_no2 = counts[no2]
    num_no3 = counts[no3]

    sum_no0 = sum_adjacent(counts, no0)
    sum_no1 = sum_adjacent(counts, no1)
    
    # 1 + 2 > 50%
    if num_no0 + num_no1 > 0.5*L:
        # 1-2 >= 20%, or 1, 2, 3 flat, or 1 adj to 2, 3
        if num_no0-num_no1 >= 0.2*L \
        or no0 == 8 or no1 == 8 or no2 == 8 \
        or adjacent(no0, no1) == 1 or adjacent(no0, no2) == 1:
            return no0
        else:
            # 1 not adj to 2 or 3, and 2 not adj to 3
            if adjacent(no0, no1) > 1 and adjacent(no0, no2) > 1 and adjacent(no1, no2) > 1:
                return no0
            else:
                # 1 adj to 4, 2 adj to 3
                if adjacent(no0, no3) == 1 and adjacent(no1, no2) == 1:
                    if num_no2-num_no3 <= 0.1*L:
                        if num_no0+num_no3 > num_no1+num_no2:
                            return no0
                        else:
                            return no1
                    else:
                        if num_no1 + num_no2 > num_no0:
                            return no1
                        else:
                            return no0
                else:
                    # 2 adj to 3
                    if adjacent(no1, no2) == 1:
                        if num_no1 + num_no2 > num_no0:
                            return no1
                        else:
                            return no0
                    else:
                        # impossible
                        return acdabbfatsh
    else:
        # 1 adj to 2, 1 not flat, 2 not flat
        if adjacent(no0, no1) == 1 and no0 != 8 and no1 != 8:
            return no0
        else:
            # 1 not adj to 2, 1 not flat, 2 not flat
            if no0 != 8 and no1 != 8 and adjacent(no0, no1) > 1:
                if sum_no0 > sum_no1:
                    return no0
                else:
                    return no1
            else:
                if no0 == 8 or no1 == 8:
                    # 1 is flat
                    if no0 == 8:
                        if sum_no1 > num_no0:
                            return no1
                        else:
                            return no0
                    else:
                        if num_no0 >= num_no1:
                            return no0
                        else:
                            return no1
                else:
                    # impossible
                    return afegdagt
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

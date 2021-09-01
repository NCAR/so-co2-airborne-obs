from scipy.optimize import curve_fit

import numpy as np
import xarray as xr
import dask
import dask.array as darray


# define the model
last_trend_parm = 4
nparm = 8
def poly_harm(t, mu, b1, b2, b3, a1, phi1, a2, phi2):
    """Linear trend plus harmonics."""
    return (mu + b1 * t + b2 * t**2 + b3 * t**3 +
            a1 * np.cos(1. * 2. * np.pi * t + phi1) +
            a2 * np.cos(2. * 2. * np.pi * t + phi2))


def poly_harm_nocycle(t, abcd):
    abcd_nocycle = abcd.copy()
    abcd_nocycle[last_trend_parm:] = 0.
    return poly_harm(t, *abcd_nocycle)


def poly_harm_justcycle(t, abcd):
    abcd_justcycle = abcd.copy()
    abcd_justcycle[:last_trend_parm] = 0.
    return poly_harm(t, *abcd_justcycle)


def curve_fit_nD(np_array, time_yrfrac):
    """Fit the `poly_harm` function to data."""
    shape = np_array.shape
    y = np_array.reshape([shape[0], -1])

    arr_abdc = np.empty((nparm,) + y.shape[1:])
    arr_abdc.fill(np.nan)

    for i in range(y.shape[1]):
        if any(np.isnan(y[:, i])):
            continue

        abcd, pcov = curve_fit(poly_harm, time_yrfrac, y[:, i])
        arr_abdc[:, i] = abcd

    return arr_abdc.reshape((nparm,) + shape[1:])


def fitted_mean(abcd_array, time_yrfrac):
    """A function to compute the long-term mean of the trend part of poly_harm."""
    shape = abcd_array.shape

    abcd_flat = abcd_array.reshape([shape[0], -1])

    arr_flat_mean = np.empty(abcd_flat.shape[1:])
    arr_flat_mean.fill(np.nan)

    for i in range(abcd_flat.shape[1]):
        if any(np.isnan(abcd_flat[:, i])):
            continue

        arr_flat_mean[i] = poly_harm_nocycle(time_yrfrac, abcd_flat[:, i]).mean()

    return arr_flat_mean.reshape(shape[1:])


def detrend(ds, map_blocks=False):
    """fit and apply detrending function to dataset, retain mean"""

    dso = ds.copy()
    print(dso)
    time_yrfrac = gen_year_fraction(ds)
    
    da_list = []
    for v in ds.data_vars:
        if 'time' in ds[v].dims and v not in ['time_bnds', 'time_bound']:            
            print(v)
            if map_blocks:
                dso[v] = detrend_da_mapblocks(ds[v], time_yrfrac)
            else:
                dso[v] = detrend_da(ds[v], time_yrfrac)

    #da_list = dask.compute(*da_list)
    
    return dso


def get_trend(ds):
    """return only the trend portion of the fit"""
    dso = ds.copy()
    time_yrfrac = gen_year_fraction(ds)

    da_list = []
    for v in ds.data_vars:
        if 'time' in ds[v].dims and v not in ['time_bnds', 'time_bound']:
            abcd = curve_fit_nD(ds[v].values, time_yrfrac)
            dso[v] = poly_harm_nocycle(time_yrfrac, abcd)
    return dso


def get_fitted(ds):
    """return only the trend portion of the fit"""
    dso = ds.copy()
    time_yrfrac = gen_year_fraction(ds)

    da_list = []
    for v in ds.data_vars:
        if 'time' in ds[v].dims and v not in ['time_bnds', 'time_bound']:
            abcd = curve_fit_nD(ds[v].values, time_yrfrac)
            dso[v] = poly_harm(time_yrfrac, *abcd)
    return dso

def detrend_da(da, time_yrfrac):
    """fit and apply detrending function, retain mean"""
    daout = da.copy()
    abcd = curve_fit_nD(da.values, time_yrfrac)
    daout.data = detrend_w_parms(abcd, da.values, time_yrfrac)
    return daout


def detrend_da_mapblocks(da, time_yrfrac):
    """fit and apply detrending function, retain mean"""
    daout = da.copy()
    nt = len(time_yrfrac)
    dda = da.data.rechunk((nt, 8, 8))      
    abcd = darray.map_blocks(curve_fit_nD, dda, time_yrfrac, 
                             chunks=(nparm, 8, 2), drop_axis=(0,), 
                             new_axis=(0,), dtype=dda.dtype).compute()   
    
    daout.data = detrend_w_parms(abcd, da.values, time_yrfrac)
    return daout


def detrend_w_parms(abcd_array, np_array, time_yrfrac):
    """a detrending function: removes the trend from data, but adds back in the long-term mean."""
    shape_parms = abcd_array.shape # i.e., nparm, nz, ny
    shape_arr = np_array.shape # i.e., nt, nz, ny

    abcd_flat = abcd_array.reshape([shape_parms[0], -1])
    arr_flat = np_array.reshape([shape_arr[0], -1])

    if arr_flat.shape[1:] != abcd_flat.shape[1:]:
        raise ValueError('dimension mismatch')

    tmean_flat = fitted_mean(abcd_array, time_yrfrac).reshape(-1)

    arr_detrended_flat = np.empty(arr_flat.shape)
    arr_detrended_flat.fill(np.nan)

    for i in range(abcd_flat.shape[1]):
        if any(np.isnan(abcd_flat[:, i])):
            continue

        trend = poly_harm_nocycle(time_yrfrac, abcd_flat[:, i])
        arr_detrended_flat[:, i] = arr_flat[:, i] - trend + tmean_flat[i]

    return arr_detrended_flat.reshape(shape_arr)


def seasonal_cycle_w_mean(abcd_array, time_yrfrac):
    """a function that returns just the seasonal cycle (including the mean)"""
    shape = abcd_array.shape

    abcd_flat = abcd_array.reshape([shape[0], -1])
    tmean_flat = fitted_mean(abcd_array, time_yrfrac).reshape(-1)

    arr_flat = np.empty((len(time_yrfrac),) + abcd_flat.shape[1:])
    arr_flat.fill(np.nan)

    for i in range(abcd_flat.shape[1]):
        if any(np.isnan(abcd_flat[:, i])):
            continue

        arr_flat[:, i] = poly_harm_justcycle(time_yrfrac / 365, abcd_flat[:, i]) + tmean_flat[i]

    return arr_flat.reshape((len(time_yrfrac),) + shape[1:])


def gen_year_fraction(ds):
    return (((ds.time - ds.time[0]) / np.timedelta64(1, 'D')).astype(np.float) / 365.25).compute()

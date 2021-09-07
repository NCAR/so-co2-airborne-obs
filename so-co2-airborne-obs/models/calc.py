from collections.abc import Iterable

import calendar
import cftime

import numpy as np
import xarray as xr


# units
mwCO2 = 44.
mwC = 12.01
mwAir = 28.966

mon_per_year = 12
g_per_Pg = 1e-15
g_per_kg = 1e3
d_per_yr = 365.25
s_per_d = 86400.

molC_to_PgC = g_per_Pg * mwC

kgCmon_to_PgCyr = mon_per_year * g_per_kg * g_per_Pg
kgCO2mon_to_PgCyr = mon_per_year * g_per_kg * g_per_Pg * mwC / mwCO2
kgCmon_to_molCyr = mon_per_year * g_per_kg / mwC
kgCO2mon_to_molCyr = mon_per_year * g_per_kg / mwCO2
molm2s_to_molCm2yr = d_per_yr * s_per_d
mols_to_PgCyr = mwC * g_per_Pg * d_per_yr * s_per_d
molCyr_to_PgCyr = mwC * g_per_Pg
kgCO2s_to_molCyr = d_per_yr * s_per_d * g_per_kg / mwCO2 

str_molCm2yr = 'mol C m$^{-2}$ yr$^{-1}$'
str_PgCyr = 'Pg C yr$^{-1}$'


def eomday(year, month):
    """end of month day"""
    if isinstance(year, Iterable):
        assert isinstance(month, Iterable)
        return np.array([calendar.monthrange(y, m)[-1] for y, m in zip(year, month)])
    else:
        return calendar.monthrange(year, month)[-1]

    
def yyyymmdd_to_year_mon_day(yyyymmdd):
    year = np.int(yyyymmdd * 1e-4)
    month = np.int((yyyymmdd - year * 1e4) * 1e-2)
    day = np.int(yyyymmdd - year * 1e4 - month * 1e2)
    return year, month, day


def nday_per_year(year):
    """number of days in a year"""
    if eomday(year, 2) == 29:
        return 366
    else:
        return 365

def to_datenum(y, m, d, time_units='days since 0001-01-01 00:00:00'):
    """convert year, month, day to number"""      
    return cftime.date2num(cftime.datetime(y, m, d), units=time_units)

       
def infer_lat_name(ds):
    lat_names = ['latitude', 'lat']
    for n in lat_names:
        if n in ds:
            return n
    raise ValueError('could not determine lat name')    


def infer_lon_name(ds):
    lon_names = ['longitude', 'lon']
    for n in lon_names:
        if n in ds:
            return n
    raise ValueError('could not determine lon name')    
    


def lat_weights_regular_grid(lat):
    """
    Generate latitude weights for equally spaced (regular) global grids.
    Weights are computed as sin(lat+dlat/2)-sin(lat-dlat/2) and sum to 2.0.
    """   
    dlat = np.abs(np.diff(lat))
    np.testing.assert_almost_equal(dlat, dlat[0])
    w = np.abs(np.sin(np.radians(lat + dlat[0] / 2.)) - np.sin(np.radians(lat - dlat[0] / 2.)))

    if np.abs(lat[0]) > 89.9999: 
        w[0] = np.abs(1. - np.sin(np.pi / 2. - np.radians(dlat[0] / 2.)))

    if np.abs(lat[-1]) > 89.9999:
        w[-1] = np.abs(1. - np.sin(np.pi / 2. - np.radians(dlat[0] / 2.)))

    return w


def compute_grid_area(ds, check_total=True):
    """Compute the area of grid cells."""
    
    radius_earth = 6.37122e6 # m, radius of Earth
    area_earth = 4.0 * np.pi * radius_earth**2 # area of earth [m^2]e
    
    lon_name = infer_lon_name(ds)       
    lat_name = infer_lat_name(ds)        
    
    weights = lat_weights_regular_grid(ds[lat_name])
    area = weights + 0.0 * ds[lon_name] # add 'lon' dimension
    area = (area_earth / area.sum(dim=(lat_name, lon_name))) * area
    
    if check_total:
        np.testing.assert_approx_equal(np.sum(area), area_earth)
        
    return xr.DataArray(area, dims=(lat_name, lon_name), attrs={'units': 'm^2', 'long_name': 'area'})  


def ensure_monthly(ds):
    np.testing.assert_approx_equal(
            actual=(ds.time.diff('time') / np.timedelta64(1, 'D')).mean(),
            desired=30.4,
            significant=1
        )
    
    
def ensure_daily(ds):
    np.testing.assert_approx_equal(
            actual=(ds.time.diff('time') / np.timedelta64(1, 'D')).mean(),
            desired=1,
            significant=7
        )
    

def normalize_freq(freq):
    if freq in ['mon', 'monthly', '1M', 'month_1']:
        return '1M'
    elif freq in ['day', 'daily', 'D']:
        return 'D'
    else:
        raise ValueError(f'unknown freq: {freq}')
        
    
def resample(ds, freq):
    """wrapper to xarray resample, avoiding putting time dim in data"""
    
    freq = normalize_freq(freq)
        
    # determine time vars
    time_vars = [v for v in ds.variables if 'time' in ds[v].dims]
    other_vars = set(ds.variables) - set(time_vars)

    # resample
    with xr.set_options(keep_attrs=True):
        dsr = ds[time_vars].resample(time=freq).mean()        

    # copy other vars
    for v in other_vars:
        dsr[v] = ds[v]
        
    return dsr
        
def regularize_monthly_time(ds):
    """Produce a time axis for the middle of the month"""

    # make sure it's monthly data
    ensure_monthly(ds)

    # construct time bounds
    year = ds.time.values.astype('datetime64[Y]').astype(int) + 1970
    month = ds.time.values.astype('datetime64[M]').astype(int) % 12 + 1
    lastday = eomday(year, month)

    oneday = np.timedelta64(1, 'D')
    time_bnds = np.vstack(
        ([
            np.datetime64(f'{y:04d}-{m:02d}-01') - oneday for y, m in zip(year, month)
        ],
        [
            np.datetime64(f'{y:04d}-{m:02d}-{d:02d}') 
            for y, m, d in zip(year, month, lastday)
        ])).T.astype('datetime64[ns]')

    # write time and time_bounds to dataset
    ds['time_bnds'] = xr.DataArray(
        time_bnds, 
        dims=('time', 'd2'), 
        name='time_bnds'
    )
    
    ds['time'] = xr.DataArray(
        time_bnds.astype('int64').mean(axis=1).astype('datetime64[ns]'), 
        dims=('time'), 
        name='time'
    )
    ds.time.attrs['bounds'] = 'time_bnds'

    return ds


def regularize_daily_time(ds):
    """Produce a time axis for the middle of the day"""

    # make sure it's daily data
    ensure_daily(ds)    
    
    # construct time bounds
    year = ds.time.values.astype('datetime64[Y]').astype(int) + 1970
    month = ds.time.values.astype('datetime64[M]').astype(int) % 12 + 1
    day = (ds.time.values.astype('datetime64[D]') - ds.time.values.astype('datetime64[M]') + 1).astype(int)

    oneday = np.timedelta64(1, 'D')
    time_bnds = np.vstack(
        ([
            np.datetime64(f'{y:04d}-{m:02d}-{d:02d}') for y, m, d in zip(year, month, day)
        ],
        [
            np.datetime64(f'{y:04d}-{m:02d}-{d:02d}') + oneday for y, m, d in zip(year, month, day)
        ]
        )).T.astype('datetime64[ns]')

    # write time and time_bounds to dataset
    ds['time_bnds'] = xr.DataArray(
        time_bnds, 
        dims=('time', 'd2'), 
        name='time_bnds'
    )
    
    ds['time'] = xr.DataArray(
        time_bnds.astype('int64').mean(axis=1).astype('datetime64[ns]'), 
        dims=('time'), 
        name='time'
    )
    ds.time.attrs['bounds'] = 'time_bnds'

    return ds


def compute_potential_temperature(pressure, temperature, p0=None):
    """compute potential temperate from pressure and temperature
        
        ptemp = temperature * (p0 / pressure)**0.286 
        
        p0 is determined based on units of pressure
    """
    if p0 is None:
        if 'units' in pressure.attrs:
            if pressure.attrs['units'].lower() in ['hpa', 'mb', 'millibar', 'millibars']:
                p0 = 1000.
            elif pressure.attrs['units'].lower() in ['pa']:
                p0 = 100000. # default units = Pa
        else:
            raise ValueError('cannot determine units of pressure')

    ptemp = temperature * (p0 / pressure)**0.286
    ptemp.name = 'theta'
    ptemp.attrs['long_name'] = 'Potential temperature'
    if 'units' in temperature.attrs:
        ptemp.attrs['units'] = temperature.attrs['units']

    return ptemp


def remap_vertical_coord(ds, da_new_levels, da_coord_field, levdim='lev', method='log',
                         include_coord_field=False):
    """Interpolate to new vertical coordinate.

    Parameters
    ----------
    ds :  xarray Dataset
        xarray dataset with data to be remapped
    da_new_levels : xarray DataArray
        The levels upon which to remap
    da_coord_field : xarray DataArray
        4d da_coord_field field
    levdim : str, optional
        The name of the "level" dimension    
    method : str
        log or linear
    include_coord_field : boolean
        Remap the coordinate field itself
    
    Returns
    -------
    dso : xr.Dataset
        Interpolated dataset
    """

    if method == 'linear':
        from metpy.interpolate import interpolate_1d
        interpolate = interpolate_1d
    elif method == 'log':
        from metpy.interpolate import log_interpolate_1d
        interpolate = log_interpolate_1d
    else:
        raise ValueError(f'unknown option for interpolation: {method}')
    
    if isinstance(da_coord_field, str):
        da_coord_field = ds[da_coord_field]
               
    # determine output dims
    dims_in = da_coord_field.dims
    interp_axis = dims_in.index(levdim)
    dims_out = dims_in[:interp_axis] + da_new_levels.dims + dims_in[interp_axis+1:]
    
    len_interp_dim = da_coord_field.shape[interp_axis] 
    assert len_interp_dim == len(da_new_levels), (
        f'new_levels must be the same length as the {levdim} ' 
        f'in input data (limitation of application of apply_ufunc)'
    )
    
    # loop over vars and interpolate
    dso = xr.Dataset()
    coords = {c: da_coord_field[c] for c in da_coord_field.coords if c != levdim}
    coords[da_new_levels.dims[0]] = da_new_levels
    
    def interp_func(da_coord_field, data_field):
        """Define interpolation function."""
        return interpolate(da_new_levels.values, da_coord_field, data_field, 
                           axis=interp_axis)
    
    for name in ds.variables:       
        da = ds[name]        
        if name == da_coord_field.name and not include_coord_field:
            continue
            
        if da.dims != dims_in: 
            dso[name] = da.copy()            
        else:
            data = xr.apply_ufunc(interp_func, da_coord_field, da,
                                  output_dtypes=[da.dtype],
                                  dask='parallelized')
            dso[name] = xr.DataArray(data.data, dims=dims_out,
                                     attrs=da.attrs, coords=coords)

    return dso





from collections.abc import Iterable

import calendar

import cftime

import numpy as np
import xarray as xr

import ESMF

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


def pressure_from_hybrid_levels(dsi, layer_center=True):
    """Calculate pressure at the hybrid levels.

    Parameters
    ----------

    dsi : xarray Dataset
       Dataset must contain P0,PS,and hybrid coefficients hya[m,i] hyb[m,i]
    layer_center : logical, optional
       compute pressure on cell centers, otherwise compute interface pressure
    """

    ds = dsi.copy()

    if layer_center:
        hya = 'hyam'
        hyb = 'hybm'
        name = 'Pm'
        long_name = 'Pressure (layer center)'
    else:
        hya = 'hyai'
        hyb = 'hybi'
        name = 'Pi'
        long_name = 'Pressure (interface)'
        
    require_variables(ds,['P0', 'PS', hya, hyb])
    
    # compute pressure
    P = (ds.P0 * ds[hya] + ds.PS * ds[hyb]) * 0.01 # kg m/m^2/s^2 = Pa,*0.01 convert to hPa
    P.name = name
    
    # put the time dimension first
    if 'time' in P.dims:
        newdims = ('time', ) + tuple(d for d in P.dims if d != 'time')
        P = P.transpose(*newdims)

    # set the attributes
    P.attrs['long_name'] = long_name
    P.attrs['units'] = 'hPa'

    return P


def require_variables(ds, req_var):
    missing_var_error = False
    for v in req_var:
        if v not in ds:
            print('ERROR: Missing required variable: %s'%v)
            missing_var_error = True
            
    if missing_var_error:
        raise ValueError('missing variables')
        
        
        
def compute_date_mid(ds):
    date = cftime.num2date(
        ds[ds.time.bounds].mean(dim=ds[ds.time.bounds].dims[-1]),
        units=ds.time.units,
        calendar=ds.time.calendar,
        only_use_cftime_datetimes=True
    )    
    attrs = ds.time.attrs
    encoding = ds.time.encoding
    encoding['units'] = 'days since 2000-01-01 00:00:00'
    encoding['calendar'] = attrs['calendar']
    del attrs['units']
    del attrs['calendar']
    
    ds['time'] = date
    ds.time.attrs = attrs
    ds.time.encoding = encoding        
    
    
    
def esmf_create_grid(xcoords, ycoords, xcorners=False, ycorners=False,
                corners=False, domask=False, doarea=False,
                ctk=ESMF.TypeKind.R8):
    """
    Create a 2 dimensional Grid using the bounds of the x and y coordiantes.
    :param xcoords: The 1st dimension or 'x' coordinates at cell centers, as a Python list or numpy Array
    :param ycoords: The 2nd dimension or 'y' coordinates at cell centers, as a Python list or numpy Array
    :param xcorners: The 1st dimension or 'x' coordinates at cell corners, as a Python list or numpy Array
    :param ycorners: The 2nd dimension or 'y' coordinates at cell corners, as a Python list or numpy Array
    :param domask: boolean to determine whether to set an arbitrary mask or not
    :param doarea: boolean to determine whether to set an arbitrary area values or not
    :param ctk: the coordinate typekind
    :return: grid
    """
    [x, y] = [0, 1]

    # create a grid given the number of grid cells in each dimension, the center stagger location is allocated, the
    # Cartesian coordinate system and type of the coordinates are specified
    max_index = np.array([len(xcoords), len(ycoords)])
    grid = ESMF.Grid(max_index,
                     staggerloc=[ESMF.StaggerLoc.CENTER],
                     coord_sys=ESMF.CoordSys.SPH_DEG,
                     num_peri_dims=1,
                     periodic_dim=x,
                     coord_typekind=ctk)

    # set the grid coordinates using pointer to numpy arrays, parallel case is handled using grid bounds
    gridXCenter = grid.get_coords(x)
    x_par = xcoords[grid.lower_bounds[ESMF.StaggerLoc.CENTER][x]:grid.upper_bounds[ESMF.StaggerLoc.CENTER][x]]

    gridXCenter[...] = x_par.reshape((x_par.size, 1))

    gridYCenter = grid.get_coords(y)
    y_par = ycoords[grid.lower_bounds[ESMF.StaggerLoc.CENTER][y]:grid.upper_bounds[ESMF.StaggerLoc.CENTER][y]]
    gridYCenter[...] = y_par.reshape((1, y_par.size))

    # create grid corners in a slightly different manner to account for the bounds format common in CF-like files
    if corners:
        raise ValueError('not tested')
        grid.add_coords([ESMF.StaggerLoc.CORNER])
        lbx = grid.lower_bounds[ESMF.StaggerLoc.CORNER][x]
        ubx = grid.upper_bounds[ESMF.StaggerLoc.CORNER][x]
        lby = grid.lower_bounds[ESMF.StaggerLoc.CORNER][y]
        uby = grid.upper_bounds[ESMF.StaggerLoc.CORNER][y]

        gridXCorner = grid.get_coords(x, staggerloc=ESMF.StaggerLoc.CORNER)
        for i0 in range(ubx - lbx - 1):
            gridXCorner[i0, :] = xcorners[i0+lbx, 0]
        gridXCorner[i0 + 1, :] = xcorners[i0+lbx, 1]

        gridYCorner = grid.get_coords(y, staggerloc=ESMF.StaggerLoc.CORNER)
        for i1 in range(uby - lby - 1):
            gridYCorner[:, i1] = ycorners[i1+lby, 0]
        gridYCorner[:, i1 + 1] = ycorners[i1+lby, 1]

    # add an arbitrary mask
    if domask:
        raise ValueError('not tested')
        mask = grid.add_item(ESMF.GridItem.MASK)
        mask[:] = 1
        mask[np.where((1.75 <= gridXCenter.any() < 2.25) &
                      (1.75 <= gridYCenter.any() < 2.25))] = 0

    # add arbitrary areas values
    if doarea:
        raise ValueError('not tested')
        area = grid.add_item(ESMF.GridItem.AREA)
        area[:] = 5.0

    return grid


def esmf_create_locstream_spherical(lon, lat, coord_sys=ESMF.CoordSys.SPH_DEG,
                                    mask=None):
    """
    :param coord_sys: the coordinate system of the LocStream
    :param domask: a boolean to tell whether or not to add a mask
    :return: LocStream
    """
    if ESMF.pet_count() is not 1:
        raise ValueError("processor count must be 1 to use this function")

    locstream = ESMF.LocStream(len(lon), coord_sys=coord_sys)

    locstream["ESMF:Lon"] = lon
    locstream["ESMF:Lat"] = lat
    if mask is not None:
        locstream["ESMF:Mask"] = mask.astype(np.int32)

    return locstream


def esmf_interp_points(ds_in, locs_lon, locs_lat, lon_field_name='lon',
                lat_field_name='lat'):
    """Use ESMF toolbox to interpolate grid at points."""

    # generate grid object
    grid = esmf_create_grid(ds_in[lon_field_name].values.astype(np.float),
                            ds_in[lat_field_name].values.astype(np.float))

    # generate location stream object
    locstream = esmf_create_locstream_spherical(locs_lon.values.astype(np.float),
                                                locs_lat.values.astype(np.float))

    # generate regridding object
    srcfield = ESMF.Field(grid, name='srcfield')
    dstfield = ESMF.Field(locstream, name='dstfield')

    regrid = ESMF.Regrid(srcfield, dstfield,
                         regrid_method=ESMF.RegridMethod.BILINEAR,
                         unmapped_action=ESMF.UnmappedAction.ERROR)

    # construct output dataset
    coords = {c: locs_lon[c] for c in locs_lon.coords}
    dims_loc = locs_lon.dims
    nlocs = len(locs_lon)
    ds_out = xr.Dataset(coords=coords, attrs=ds_in.attrs)

    for name, da_in in ds_in.data_vars.items():

        # get the dimensions of the input dataset; check if it's spatial
        dims_in = da_in.dims
        if lon_field_name not in dims_in or lat_field_name not in dims_in:
            continue

        # get the dimension/shape of output
        non_lateral_dims = dims_in[:-2]
        dims_out = non_lateral_dims + dims_loc
        shape_out = da_in.shape[:-2] + (nlocs,)

        # create output dataset
        da_out = xr.DataArray((np.ones(shape_out)*np.nan).astype(da_in.dtype),
                              name=name,
                              dims=dims_out,
                              attrs=da_in.attrs,
                              coords={c: da_in.coords[c] for c in da_in.coords
                                      if c in non_lateral_dims})
        dstfield.data[...] = np.nan

        if len(non_lateral_dims) > 0:
            da_in_stack = da_in.stack(non_lateral_dims=non_lateral_dims)
            da_out_stack = xr.full_like(da_out, fill_value=np.nan).stack(non_lateral_dims=non_lateral_dims)

            for i in range(da_in_stack.shape[-1]):
                srcfield.data[...] = da_in_stack.data[:, :, i].T
                dstfield = regrid(srcfield, dstfield, zero_region=ESMF.Region.SELECT)
                da_out_stack.data[:, i] = dstfield.data

            da_out.data = da_out_stack.unstack('non_lateral_dims').transpose(*dims_out).data

        else:
            srcfield.data[...] = da_in.data[:, :].T
            dstfield = regrid(srcfield,dstfield,zero_region=ESMF.Region.SELECT)
            da_out.data = dstfield.data

        ds_out[name] = da_out
        
    return ds_out    


def r2(ds_stl):
    """compute coefficient of determination"""
    sst = np.sum((ds_stl.observed - ds_stl.observed.mean())**2)
    ssr = np.sum(ds_stl.resid**2)
    return (1. - ssr/sst).values


def stl_ds(da, trend, seasonal, period):
    """
    Apply the STL model and return an Xarray Dataset.
            
    References
    ----------
    
    [1] https://www.statsmodels.org/devel/examples/notebooks/generated/stl_decomposition.html
    
    [2] R. B. Cleveland, W. S. Cleveland, J.E. McRae, and I. Terpenning
    (1990) STL: A Seasonal-Trend Decomposition Procedure Based on LOESS.
    Journal of Official Statistics, 6, 3-73.    
    """
    
    from statsmodels.tsa.seasonal import STL
    
    dso = xr.Dataset(
        {
            'observed': da.copy().reset_coords(
                [c for c in da.coords if c != 'time'], 
                drop=True,
            )
        }
    )    
    
    stl = STL(
        da, 
        period=period,
        trend=trend,
        seasonal=seasonal,
        robust=True,
    ).fit()                

    for attr in ['trend', 'seasonal', 'resid']:
        dso[attr] = xr.DataArray(
            getattr(stl, attr), 
            dims=('time'), 
            coords={'time': da.time},
        )
    dso['predicted'] = xr.DataArray(
        stl.trend + stl.seasonal,
        dims=('time'), 
        coords={'time': da.time},
    )
    dso.resid.data = dso.observed - dso.predicted            

    print(f'\tSTL fit: r^2 = {r2(dso):0.4f}')

    return dso


def apply_stl_decomp(co2_data, freq='monthly'):
    """
    (1) Apply the STL fit with `trend_window=121`;
    (2) Fit the residuals from (1) with `trend_window=25`;
    (3) Add (1) and (2) to get the final fit.

    """
    co2_data = co2_data.copy()
    
    if freq == 'monthly':
        windows = [121, 25]
        seasonal = 13 
        period = 12
        ensure_monthly(co2_data)
        
    elif freq == 'daily':
        windows = [3651, 731]
        seasonal = 367
        period = 365
        ensure_daily(co2_data)
    else:
        raise ValueError('unknown freq')
    
    spo_fits = []
    for trend_window in windows:
        print(f'{co2_data.name} applying STL:')
        print(f'\ttrend = {trend_window}')
        stl_fit =  stl_ds(
            co2_data, 
            trend=trend_window, 
            seasonal=seasonal,
            period = period,
        )
        spo_fits.append(stl_fit)
        co2_data.data = stl_fit.resid.data
    
    spo_fit = spo_fits[0]
    for i in range(1, len(spo_fits)):
        for v in ['trend', 'seasonal', 'predicted']:
            spo_fit[v].data = spo_fit[v] + spo_fits[i][v]

    spo_fit.resid.data = spo_fit.observed - spo_fit.predicted
    print(f'Overall r^2 = {r2(spo_fit):0.4f}')

    return spo_fit


def get_stl_trend_ds(dset, freq='monthly'):
    dso = dset.copy()
    for v in ['CO2', 'CO2_OCN', 'CO2_LND', 'CO2_FFF']:
        dso[v].data = apply_stl_decomp(dset[v], freq=freq).trend.data
    return dso
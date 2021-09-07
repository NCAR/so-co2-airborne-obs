"""
Routines support CarbonTracker
"""

import os
import xarray as xr
import numpy as np

import dask

from . generic_assets import list_assets
from . import calc

this_model = 'CarbonTracker'

products = [
    'fluxes',
    'molefractions_surface',
    'molefractions_z',
]   

extended_domain_subset = dict(
    lat=slice(-90, -30)
)    
    
z3_coord_26lev = xr.DataArray(
    np.array([ 100.,   150.,   200.,   300.,   400.,   800.,  1200.,  1600.,
            2000.,  2400.,  2800.,  3200.,  3600.,  4000.,  4400.,  4800.,  5200.,
            5600.,  6400.,  7200.,  8000.,  8800.,  9600., 10400., 11200.])[::-1], # make array go top to bottom
    dims=('zlev'), 
    attrs={'long_name': 'Geopotential height (above sea level),', 'units': 'm'},
    name='Z3',
)
    
def open_dataset(product, freq=None):
    """open a dataset"""
        
    if product == 'fluxes':
        ds = _open_dataset_fluxes()
        ds_freq = calc.normalize_freq('monthly')
        
    elif product == 'molefractions_surface':
        
        def preprocess(ds):
            ds = ds.rename({'P': 'PS'})
            return ds.isel(lev=0, drop=True)
        
        ds = _open_dataset_molefractions(preprocess).drop('Z3')
        ds = calc.regularize_daily_time(ds)
        ds_freq = calc.normalize_freq('daily')

    elif product == 'molefractions':
        
        ds = _open_dataset_molefractions()
        ds = calc.regularize_daily_time(ds)        
        ds_freq = calc.normalize_freq('daily')
        
    elif product == 'molefractions_z':
        
        def preprocess(ds):
            """remap to vertical coord"""
            dsz = calc.remap_vertical_coord(
                ds,
                da_new_levels=z3_coord_26lev, 
                da_coord_field='Z3',
                levdim='lev',
                method='log',
                include_coord_field=True,
            ).drop(['lev'])
            
            return dsz
        
        ds = _open_dataset_molefractions(preprocess)
        ds = calc.regularize_daily_time(ds)        
        ds_freq = calc.normalize_freq('daily')

    elif product == 'molefractions_theta':
        raise NotImplemented('no')
        def preprocess(ds):
            """remap to vertical coord"""
            
            ds['theta'] = calc.compute_potential_temperature(
                ds['P'], 
                ds['T']
            )
            dsz = calc.remap_vertical_coord(
                ds,
                da_new_levels=theta_coord_26lev, 
                da_coord_field='theta',
                levdim='lev',
                method='log',
                include_coord_field=True,
            ).drop(['lev'])
            
            return dsz
        
        ds = _open_dataset_molefractions(preprocess)
        ds = calc.regularize_daily_time(ds)        
        ds_freq = calc.normalize_freq('daily')
                
    else:        
        raise ValueError(f'unknown product {product}')
    
       
    if freq is None:
        return ds 
    
    freq = calc.normalize_freq(freq)    
    if freq != ds_freq:
        ds = calc.resample(ds, freq=freq) 

    if freq == calc.normalize_freq('monthly'):
        ds = calc.regularize_monthly_time(ds)        

    elif freq == calc.normalize_freq('daily'):
        ds = calc.regularize_daily_time(ds)        
    return ds

        
def _open_dataset_molefractions(preprocess=lambda ds: ds):
    """open a mole fractions dataset"""

    datestr = lambda str_f: str_f.split('.')[-2].split('_')[-1]
    
    tmp = list_assets(this_model, 'molefractions')
    files_co2_tot = tmp[0]
    files_co2_com = tmp[1]
    assert len(files_co2_tot) == len(files_co2_com), f'CT files mismatch'
    
    datestr_match = [
        datestr(f_tot) == datestr(f_com) 
        for f_tot, f_com in zip(files_co2_tot, files_co2_com)
    ]
    assert all(datestr_match), f'CT file date mismatch'

    coord_rename = dict(
        latitude='lat',
        longitude='lon',
        level='lev',
    )
    
    @dask.delayed
    def process_one_file(file_tot, file_com):
        """process one file"""
        
        data_vars = ['CO2', 'Z3', 'P', 'T', 'theta']
        with xr.open_dataset(file_tot, use_cftime=True) as dst:    
            dst = dst.rename(coord_rename)
            dst = dst.rename({'co2': 'CO2'})
            dst = dst.sel(**extended_domain_subset)
            
            dst = dst.rename({'temperature': 'T'})
            
            tmp_c = xr.full_like(dst.CO2, fill_value=np.nan)
            tmp_c.name = 'Z3'
            tmp_c.attrs = dst.gph.attrs            
            tmp_c.data = (dst.gph.data[:, :-1, :, :] + dst.gph.data[:, 1:, :, :]) / 2.
            dst['Z3'] = tmp_c
            
            tmp_c = xr.full_like(dst.CO2, fill_value=np.nan)
            tmp_c.name = 'P'
            attrs = dst.pressure.attrs
            del attrs['comment']
            tmp_c.attrs = attrs
            tmp_c.data = (dst.pressure.data[:, :-1, :, :] + dst.pressure.data[:, 1:, :, :]) / 2.
            dst['P'] = tmp_c
            
            
            dst['theta'] = calc.compute_potential_temperature(
                dst['P'], 
                dst['T']
            )
        
            dst = dst[data_vars]

        data_vars = ['CO2_OCN', 'CO2_LND', 'CO2_FFF']
        with xr.open_dataset(file_com, use_cftime=True) as dsc:    
            dsc = dsc.rename(coord_rename)
            dsc = dsc.sel(**extended_domain_subset)
            
            dsc['CO2_LND'] = dsc.fires + dsc.bio
            dsc.CO2_LND.attrs = dsc.bio.attrs
            dsc.CO2_LND.attrs['comment'] = 'sum of "fires" and "bio"'
            dsc.CO2_LND.encoding = dsc.bio.encoding 

            if False:
                dsc['CO2_sum'] = dsc.fires + dsc.bio + dsc.ocean + dsc.ff + dsc.bg
                dsc.CO2_sum.attrs = dsc.bio.attrs
                dsc.CO2_sum.attrs['comment'] = 'total CO2 (check sum)'
                dsc.CO2_sum.encoding = dsc.bio.encoding 
            
            dsc = dsc.rename({'ocean': 'CO2_OCN', 'ff': 'CO2_FFF'})
            dsc = dsc[data_vars]
            
        # merge
        ds = xr.merge((dst, dsc))   

        # add area field
        ds['area'] = calc.compute_grid_area(ds)
        
        ds = preprocess(ds)
        ds = calc.resample(ds, freq='D')   
        
        return ds
            
    i = 0
    ds_list = []    
    for f_tot, f_com in zip(files_co2_tot, files_co2_com):
        ds_list.append(process_one_file(f_tot, f_com)) 
        i += 1
    ds_list = dask.compute(*ds_list)
       
    ds = xr.concat(ds_list, dim='time', data_vars='minimal')
    
    return ds
    

def _open_dataset_fluxes():
    """open monthly fluxes"""

    files = list_assets(this_model, 'fluxes').popitem()[1]
    
    rename = {'ocn_flux_opt': 'SFCO2_OCN', 
              'bio_flux_opt': 'SFCO2_LND', 
              'fossil_flux_imp': 'SFCO2_FFF',
              'latitude': 'lat',
              'longitude': 'lon',              
             }   
    
    open_kwargs = dict(combine='nested', concat_dim='time',  
                       drop_variables=['time_components', 'decimal_time'])
        
    ds = xr.open_mfdataset(files, **open_kwargs)
    ds = ds.rename({k: v for k, v in rename.items() if k in ds.variables})    
    data_vars = list(filter(lambda v: v in ['SFCO2_OCN', 'SFCO2_LND', 'SFCO2_FFF'], list(ds.variables)))
    
    attrs = {v: ds[v].attrs for v in data_vars}
    
    if 'SFCO2_LND' in ds.variables:
        ds['SFCO2_LND'] = ds.SFCO2_LND + ds.fire_flux_imp
        ds = ds.drop(['fire_flux_imp'])
        
    ds['area'] = calc.compute_grid_area(ds)

    # compute total flux
    if 'SFCO2_FFF' in ds.variables and 'SFCO2_LND' in ds.variables:
        ds['SFCO2'] = ds.SFCO2_OCN + ds.SFCO2_LND + ds.SFCO2_FFF
        attrs['SFCO2'] = {'long_name': 'total surface flux', 
                          'units': attrs['SFCO2_OCN']['units']}
        data_vars += ['SFCO2']
    
    for v in data_vars:
        if attrs[v]['units'] == 'mol m-2 s-1':
            ds[v] = ds[v] * calc.molm2s_to_molCm2yr
            attrs[v]['units'] = calc.str_molCm2yr
        else:
            raise ValueError(f'unknown units: {attrs[v]["units"]}')
        ds[v].attrs = attrs[v]

    return calc.regularize_monthly_time(ds)
        


"""
Routines support CarboScope
"""

import os
import xarray as xr
import numpy as np

from . generic_assets import list_assets
from . import calc

this_model = 'CarboScope'
       
def open_dataset(product, freq='mon'):
    """open a dataset"""

    if product == 'fluxes':
        return _open_dataset_fluxes(freq)
    
    
def _open_dataset_fluxes(freq='mon'):
    """CarboScope surface flux data (it's daily)"""
    
    flux_files = list_assets(this_model, 'fluxes').popitem()[1]

    rename = {'co2flux_ocean': 'SFCO2_OCN', 
              'co2flux_land': 'SFCO2_LND',
              'co2flux_subt': 'SFCO2_FFF',
              'mtime': 'time',
              'dxyp': 'area',              
             }   
    
    open_kwargs = dict(
        combine='nested', 
        concat_dim='mtime',
        data_vars='minimal',
        drop_variables=['lspec', 'itime', 'myear', 'rt', 'year',
                        'lrt', 'itime_bounds',
                        'lproc', 'proc', 'area', 'tmask', 'dt', 
                        'lat_bounds', 'lon_bounds', 'spec'],
    )
    
    data_vars = ['SFCO2_OCN', 'SFCO2_LND', 'SFCO2_FFF']
            
    ds = xr.open_mfdataset(flux_files, **open_kwargs)

    ds = ds.rename(rename)
    attrs = {v: ds[v].attrs for v in data_vars}
    
    ds['SFCO2'] = ds.SFCO2_OCN + ds.SFCO2_LND + ds.SFCO2_FFF + ds.co2flux_excl
    attrs['SFCO2'] = {'long_name': 'total surface flux', 
                      'units': attrs['SFCO2_OCN']['units']}
    ds = ds.drop('co2flux_excl')
    data_vars += ['SFCO2']
    
    for v in data_vars:
        if attrs[v]['units'] == 'PgC/yr':
            ds[v] = ds[v] / ds.area / calc.molC_to_PgC
            attrs[v]['units'] = calc.str_molCm2yr
        else:
            raise ValueError(f'unknown units: {attrs[v]["units"]}')
        ds[v].attrs = attrs[v]

    if freq in ['mon', 'monthly']:
        dsr = calc.resample(ds, freq='mon')                    
        return calc.regularize_monthly_time(dsr)

    return ds
    
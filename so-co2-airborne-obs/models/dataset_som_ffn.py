"""
Routines support load landschutzer fluxes
"""

import os
import xarray as xr
import numpy as np

from . import calc

cache_dir = '/glade/p/eol/stephens/longcoll/co2-hole-cache'
try:
    os.listdir(cache_dir)
except:
    cache_dir = os.environ['TMPDIR']
try:
    os.listdir(cache_dir)
except:
    raise ValueError(f'cache_dir: {cache_dir} not accessible')

this_model = 'SOM-FFN'
       
def open_dataset(product, freq='mon'):
    """open a dataset"""

    if product == 'fluxes':
        return _open_dataset_fluxes(freq)
    
    
def _open_dataset_fluxes(freq='mon'):
    """An observation-based global monthly 
       gridded sea surface pCO2 product from 1982 onward and its monthly climatology
    """
    
    data_dir = f'{cache_dir}/landschutzer/download'
    filename = 'spco2_MPI_SOM-FFN_v2018.nc'
    flux_file = f'{data_dir}/{filename}'

    rename = {'fgco2_raw': 'SFCO2_OCN',
              'fgco2_smoothed': 'SFCO2_OCN_smoothed',
             }   
    
    data_vars = ['SFCO2_OCN',]
            
    ds = xr.open_dataset(flux_file, drop_variables='date')
    ds['area'] = calc.compute_grid_area(ds)

    ds = ds.rename(rename)
    attrs = {v: ds[v].attrs for v in data_vars}
    
    for v in data_vars:
        if attrs[v]['units'] == 'mol m^{-2} yr^{-1}':
            attrs[v]['units'] = calc.str_molCm2yr
        else:
            raise ValueError(f'unknown units: {attrs[v]["units"]}')
        ds[v].attrs = attrs[v]

    return calc.regularize_monthly_time(ds)
    
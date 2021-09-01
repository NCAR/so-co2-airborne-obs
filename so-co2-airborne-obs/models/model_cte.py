"""
Routines support CTE
"""

import os
import xarray as xr
import numpy as np

from . import calc
from . generic_assets import list_assets

this_model = 'CTE'

def open_dataset(product, freq='mon'):
    """open a dataset"""
    
    if freq not in ['mon', 'monthly']:
        raise NotImplementedError(f'{this_model}: freq other than "mon" not implemented')    
    
    if product == 'fluxes':
        return _open_dataset_fluxes()
    
    
def _open_dataset_fluxes():

    flux_files = list_assets(this_model, 'fluxes').popitem()[1]

    rename = {'ocn_flux_opt': 'SFCO2_OCN', 
              'bio_flux_opt': 'SFCO2_LND', 
              'fossil_flux_imp': 'SFCO2_FFF',
              'bio_flux_prior': 'SFCO2_LND_prior',
              'date': 'time',
              'cell_area': 'area',
              'latitude': 'lat',
              'longitude': 'lon',              
             }   
    
    open_kwargs = dict(combine='by_coords', concat_dim='date',  
                       drop_variables=['time_components', 'decimal_time'])
        
    ds = xr.open_mfdataset(flux_files, **open_kwargs)
    ds = ds.rename({k: v for k, v in rename.items() if k in ds.variables})    
    data_vars = list(filter(
        lambda v: v in ['SFCO2_OCN', 'SFCO2_LND', 'SFCO2_FFF', 'SFCO2_LND_prior'], list(ds.variables)
    ))
    ds = ds.drop([
        v for v in ds.variables if ('prior' in v or 'ensemble' in v) and (v not in data_vars)
    ])
    
    attrs = {v: ds[v].attrs for v in data_vars}
    
    ds['SFCO2_LND'] = ds.SFCO2_LND + ds.fire_flux_imp
    ds = ds.drop(['fire_flux_imp'])
        
    # compute total flux
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
        
    
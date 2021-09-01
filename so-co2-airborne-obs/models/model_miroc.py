"""
Routines support MIROC
"""

import os
import xarray as xr
import numpy as np

from . import calc
from . generic_assets import list_assets

this_model = 'MIROC'

def open_dataset(product, use_prior=False, freq='mon'):
    """open a dataset"""
    
    if freq not in ['mon', 'monthly']:
        raise NotImplementedError(f'{this_model}: freq other than "mon" not implemented')
    
    if product == 'fluxes':
        return _open_dataset_fluxes(use_prior)


def _open_dataset_fluxes(use_prior):

    flux_files = list_assets(this_model, 'fluxes').popitem()[1]
    
    if use_prior:
        rename = dict(
            flux_apri_ocean='SFCO2_OCN',                          
            flux_apri_land='SFCO2_LND',
            flux_apri_fossil='SFCO2_FFF',
            latitude='lat',
            longitude='lon',              
        )
    else:
        rename = dict(
            flux_apos_ocean='SFCO2_OCN', 
            flux_apos_land='SFCO2_LND', 
            flux_apri_fossil='SFCO2_FFF',
            latitude='lat',
            longitude='lon',              
        )
    open_kwargs = dict(combine='nested', concat_dim='time')
    
    data_vars = ['SFCO2_OCN', 'SFCO2_LND', 'SFCO2_FFF',]
    
    ds = xr.open_mfdataset(flux_files, **open_kwargs)
    ds = ds.rename(rename)      
    attrs = {v: ds[v].attrs for v in data_vars}
           
    ds['area'] = calc.compute_grid_area(ds)

    # compute total flux
    ds['SFCO2'] = ds.SFCO2_OCN + ds.SFCO2_LND + ds.SFCO2_FFF
    attrs['SFCO2'] = {'long_name': 'total surface flux', 
                      'units': attrs['SFCO2_OCN']['units']}
    data_vars += ['SFCO2']
    
    for v in data_vars:
        if attrs[v]['units'] == 'kg-CO2/m2/month':
            ds[v] = ds[v] * calc.kgCO2mon_to_molCyr
            attrs[v]['units'] = calc.str_molCm2yr
        else:
            raise ValueError(f'unknown units: {attrs[v]["units"]}')
        if use_prior:
            attrs[v]['note'] = 'theses are PRIOR fluxes!'
        ds[v].attrs = attrs[v]
    return calc.regularize_monthly_time(ds)
        
    
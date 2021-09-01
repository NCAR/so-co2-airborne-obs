import os
import xarray as xr
import numpy as np

from . import calc
from . generic_assets import list_assets

this_model = 'CAMS'

def open_dataset(product, freq='mon'):
    """open a dataset"""
    
    if freq not in ['mon', 'monthly']:
        raise NotImplementedError(f'{this_model}: freq other than "mon" not implemented')
        
    if product == 'fluxes':
        return _open_dataset_fluxes()
    
def _open_dataset_fluxes():

    flux_files = list_assets(this_model, 'fluxes').popitem()[1]
    
    rename = {'flux_apos_ocean': 'SFCO2_OCN', 
              'flux_apos_bio': 'SFCO2_LND', 
              'flux_foss': 'SFCO2_FFF',
              'flux_apri_ocean': 'SFCO2_OCN_prior', 
              'flux_apri_bio': 'SFCO2_LND_prior', 
              'lsf': 'land_frac',
              'latitude': 'lat',
              'longitude': 'lon',
             }
    
    open_kwargs = dict(
        combine='nested', concat_dim='time', 
        data_vars=[
            'flux_apri_bio', 'flux_apri_ocean', 
            'flux_apos_bio', 'flux_apos_ocean', 
            'flux_foss'
        ]
    )

    ds = xr.open_mfdataset(flux_files, **open_kwargs)
    ds = ds.rename(rename)  
    
    data_vars = ['SFCO2_OCN', 'SFCO2_LND', 'SFCO2_FFF', 'SFCO2_LND_prior']

    attrs = {v: ds[v].attrs for v in data_vars}

    # compute total flux
    ds['SFCO2'] = ds.SFCO2_OCN + ds.SFCO2_LND + ds.SFCO2_FFF
    attrs['SFCO2'] = {'long_name': 'total surface flux', 
                      'units': attrs['SFCO2_OCN']['units']}
    data_vars += ['SFCO2']

    ds['time'] = _time_from_files(flux_files)
    
    for v in data_vars:
        if attrs[v]['units'] == 'kgC m-2 month-1':
            ds[v] = ds[v] * calc.kgCmon_to_molCyr
            attrs[v]['units'] = calc.str_molCm2yr
        else:
            raise ValueError(f'unknown units: {attrs[v]["units"]}')
        ds[v].attrs = attrs[v]

    ds = xr.decode_cf(ds, decode_times=True)
    return calc.regularize_monthly_time(ds)
        
        
def _time_from_files(files):
    time_units='days since 0001-01-01 00:00:00'
    
    if this_model in ['CAMSv18']:
        year = [int(os.path.basename(f).split('_')[-6][:4]) for f in files]    
        month = [int(os.path.basename(f).split('_')[-6][4:]) for f in files]    
    elif this_model in ['CAMSv19', 'CAMSv20r1']:
        year = [int(os.path.basename(f).split('_')[-1][:4]) for f in files]    
        month = [int(os.path.basename(f).split('_')[-1][4:6]) for f in files]   
    else:
        raise ValueError(f'{this_model} unknown')
        
    time_data = [calc.to_datenum(y, m, calc.eomday(y, m)/2, time_units=time_units) 
                 for y, m in zip(year, month)]

    return xr.DataArray(
        time_data, 
        dims=('time'), 
        name='time',
        attrs={'units': time_units}
    )

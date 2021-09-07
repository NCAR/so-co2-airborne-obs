"""
Routines support CarbonTracker
"""

import os
import xarray as xr
import numpy as np

import dask

from . generic_assets import list_assets
from . import calc

this_model = 'TM5Flux'

products = [
    'fluxes',
]


def open_dataset(product, freq=None):
    """open a dataset"""
        
    if product == 'fluxes':
        ds_freq = calc.normalize_freq(freq)
        ds = _open_dataset_fluxes(ds_freq)
        
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

        
def _open_dataset_fluxes(freq):
    """
    Open flux datasets
    Fluxes were provided daily
    """
    
    daily_files = list_assets(this_model, 'fluxes').popitem()[1]

    if freq == calc.normalize_freq('monthly'):
        fileparts = os.path.basename(daily_files[0]).split('.')
        fileparts[1] = 'monthly'
        monthly_file = os.path.join(
            os.path.dirname(daily_files[0]), 
            '.'.join(fileparts)
        )
    
        if os.path.exists(monthly_file):
            print(f'opening {monthly_file}')
            return xr.open_dataset(monthly_file)
        
        print(f'computing {monthly_file}')
        
    @dask.delayed
    def process_one_file(file):
        """make daily flux file conform to what I like"""
        
        rename = {'ocn_flux_opt': 'SFCO2_OCN', 
                  'bio_flux_opt': 'SFCO2_LND', 
                  'fossil_flux_imp': 'SFCO2_FFF',
                  'latitude': 'lat',
                  'longitude': 'lon',                      
                 }   

        open_kwargs = dict(drop_variables=['time_components', 'decimal_time'])

        with xr.open_dataset(file, **open_kwargs) as ds:            

            ds = ds.rename({k: v for k, v in rename.items() if k in ds.variables})    
            data_vars = list(filter(lambda v: v in ['SFCO2_OCN', 'SFCO2_LND', 'SFCO2_FFF'], list(ds.variables)))

            if 'SFCO2_LND' in ds.variables:
                ds.SFCO2_LND.values = ds.SFCO2_LND + ds.fire_flux_imp
                ds = ds.drop(['fire_flux_imp'])
                
            ds['area'] = calc.compute_grid_area(ds)

            attrs = {v: ds[v].attrs for v in data_vars}

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
        
        return ds
    
    ds_list = []
    keys = []
    for f in daily_files:
        yyyymmdd = int(os.path.basename(f).split('.')[1])
        y, m, d = calc.yyyymmdd_to_year_mon_day(yyyymmdd)
        key = f'{y:04d}'
        ds_list.append(process_one_file(f))
        keys.append(key)

    if freq == calc.normalize_freq('monthly'):    
        file_list = []
        for key in sorted(set(keys)):
            ds_list_mon = [ds for k, ds in zip(keys, ds_list) if k == key]
            
            ds_list_mon = dask.compute(*ds_list_mon)        
            time_vars = [v for v, da in ds_list_mon[0].data_vars.items() if 'time' in da.dims]

            ds = xr.concat(ds_list_mon, dim='time', data_vars='minimal')
            ds_mon = ds[time_vars].resample(time='1M').mean()
                        
            # copy non-time vars
            for v in ds.variables:
                if v not in ds_mon:
                    ds_mon[v] = ds[v]

            # copy attributes        
            for v in ds_mon.data_vars:
                ds_mon[v].attrs = ds[v].attrs
            yyyymm_file = monthly_file.replace('.nc', f'.{key}.nc')
            
            print(f'writing monthly file: {yyyymm_file}')
            ds_mon.to_netcdf(yyyymm_file)
            file_list.append(yyyymm_file)

        ds_mon = xr.open_mfdataset(file_list, combine='by_coords', concat_dim='time', data_vars='minimal')        
        ds_mon = calc.regularize_monthly_time(ds_mon)

        ds_mon.info()
        ds_mon.time.encoding['units'] = 'days since 0001-01-01 00:00:00'
        print(f'writing monthly file: {monthly_file}')        
        ds_mon.to_netcdf(monthly_file)
        
        for f in file_list:
            os.remove(f)
            
        return ds_mon
    
    else:
        raise NotImplementedError('no daily support')


def get_flux_files():
    return list_assets(this_model, 'fluxes').popitem()[1]


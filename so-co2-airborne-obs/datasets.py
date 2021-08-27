import os
import shutil
from functools import partial

import xarray as xr

import obs_aircraft

cache_dir = 'data/cache'
os.makedirs(cache_dir, exist_ok=True)


def aircraft_sections(source='obs-multi-sensor', 
                      clobber=False, 
                      vertical_coord='z'):
    """return a dataset with aircraft section"""
    file_name_cache = f'{cache_dir}/aircraft-section-{source}-{vertical_coord}.zarr'
    
    if clobber and os.path.exists(file_name_cache):
        shutil.rmtree(file_name_cache)    
    
    if os.path.exists(file_name_cache):
        return xr.open_zarr(file_name_cache).compute()

    model = 'obs' if source == 'obs-multi-sensor' else source
    df = obs_aircraft.open_aircraft_data(model=model)  
    
    campaign_info = obs_aircraft.get_campaign_info(
        lump_orcas=True, clobber=clobber, verbose=True,
    )
    campaigns = list(campaign_info.keys())
    
    multi_sensor = 'multi-sensor' in source
    
    make_section = partial(obs_aircraft.make_section,                            
                           df=df,
                           vertical_coord=vertical_coord, 
                           multi_sensor=multi_sensor)
    
    ds_list = [obs_aircraft.make_section(df, 
                                         vertical_coord=vertical_coord, 
                                         campaign=c, 
                                         multi_sensor=multi_sensor) 
               for c in campaigns]
    
    ds = xr.concat(
        [ds for ds in ds_list if ds is not None],
        dim='time',
    )
    ds.to_zarr(file_name_cache, mode='w', consolidated=True)
    return ds.compute()
    
    
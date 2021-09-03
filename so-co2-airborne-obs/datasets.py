import os
import shutil
from functools import partial

import numpy as np
import xarray as xr

import emergent_constraint as ec
import obs_aircraft
import obs_surface
import util

cache_dir = 'data/cache'
os.makedirs(cache_dir, exist_ok=True)


def aircraft_sections(
    source='obs-multi-sensor', 
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
    

def obs_surface_stn_v_lat(season, constituent='CO2', minus_spo=True):
    """return a dataset with seasonal-mean surface-station observations"""
    da = obs_surface.open_surface_data_obs(constituent)
    
    if minus_spo:
        reference_records = obs_surface.reference_records[constituent]
        reference = da.sel(record=reference_records).mean('record')
        da = da - reference
                
    return util.ann_mean(da.to_dataset(), season=season, time_bnds_varname=None)


def obs_surface_climatology(constituent='CO2', minus_spo=True):
    """return a dataset with seasonal-mean surface-station observations"""
    constituent = constituent.upper()
    da = obs_surface.open_surface_data_obs(constituent)
    
    if minus_spo:
        reference_records = obs_surface.reference_records[constituent]
        reference = da.sel(record=reference_records).mean('record')
        da = da - reference

    southern_ocean_stn_list = (
        obs_surface.southern_ocean_stn_list if constituent == 'CO2'
        else obs_surface.southern_ocean_stn_list_sf6
    )

    da_clm = (da.reset_coords('lat')
              #.sel(time=slice('2009', '2018'))
              .groupby('time.month').mean('time')
              .groupby('stncode').mean('record').sel(stncode=southern_ocean_stn_list)
             )

    da_std = (da.reset_coords('lat')
              #.sel(time=slice('2009', '2018'))
              .groupby('stncode').mean('record').sel(stncode=southern_ocean_stn_list)
              .mean('stncode')
              .groupby('time.month').std('time')
              .rename({constituent: f'{constituent}_std'}).drop('lat')
             )
    return xr.merge((da_clm, da_std))



def _groupby_season_agg(ds, groups):
    """get groupby dict by season or groups of seasons"""
    group_ndx = {k: [] for k in groups}   
    for n, (season, ndx) in enumerate(ds.groupby('time.season').groups.items()):
        for g in groups:
            if season in g:
                group_ndx[g] += ndx
                break
    return group_ndx   


def _groupby(ds, groups, dim):    
    """group profile data into `groups`"""
    
    ubin = ec.get_parameters('default')['ubin']
    udθ = ec.get_parameters('default')['udθ']        
    theta_max = 321.
    
    theta_ref_slice = slice(ubin - udθ / 2, ubin + udθ / 2)# slice(295, 305) #
       
    ref = ds.sel(theta=theta_ref_slice).groupby(ds.campaign).median(['theta', 'profile'])

    ds_list = []
    for group, ndx in groups.items():
        dsi = ds[['doy', 'year']].isel(profile=ndx).mean('profile')
        for v in ds.data_vars:
            da = (ds[v].groupby(ds.campaign) - ref[v]).isel(profile=ndx).sel(theta=slice(None, theta_max))
            if '_med' in v:
                dsi[v] = da.median('profile')
            else:
                dsi[v] = da.mean('profile')
            dsi[f'{v}_std'] = da.std('profile')
        ds_list.append(dsi)

    dsg = xr.concat(ds_list, dim=xr.DataArray(list(groups.keys()), dims=(dim), name=dim))
    
    if dim == 'campaign':
        time = xr.DataArray(
            [info['time'] for c, info in campaign_info.items()],
            dims='campaign',
            coords={'campaign': list(campaign_info.keys())}
        )
        dsg, time = xr.align(dsg, time)
        dsg['time'] = time
        dsg = dsg.set_coords('time')
        
    return dsg


def aircraft_profiles(source='obs', tracer='CO2', 
                      vertical_coord='theta', clobber=False):
    
    lat_lo, lat_hi = -90., -45.
    
    file_name_cache = f'{cache_dir}/aircraft-profile-{source}-{vertical_coord}-{tracer}.zarr'
    if os.path.exists(file_name_cache) and clobber:
        shutil.rmtree(file_name_cache)

    if os.path.exists(file_name_cache):
        return xr.open_zarr(file_name_cache).compute()
    
    ds = obs_aircraft.make_profile_ds(
        source, tracer,
        profile_spec=vertical_coord,
        lat_lo=lat_lo, 
        lat_hi=lat_hi,
    )    
    ds.to_zarr(file_name_cache, consolidated=True)
    return ds.compute()


def aircraft_profiles_seasonal(source='obs', tracer='CO2', 
                               vertical_coord='theta', 
                               seasonal_groups=['DJF', 'MAM', 'JJA', 'SON'],
                               clobber=False):
    """return dataset with profiles aggegated by seasonal groups
       seasonal_group can be ['DJF', 'MAM', ...]
       or ['DJF', 'MAMJJASON']
    """
    file_name_cache = f'{cache_dir}/aircraft-profile-seasonal-{source}-{vertical_coord}-{tracer}.zarr'
    if os.path.exists(file_name_cache) and clobber:
        shutil.rmtree(file_name_cache)

    if os.path.exists(file_name_cache):
        return xr.open_zarr(file_name_cache).compute()
    
    ds = aircraft_profiles(source, tracer, vertical_coord, clobber)
    ds = _groupby(ds, _groupby_season_agg(ds, seasonal_groups), dim='season')
    
    #if source == 'obs':
    #    ds = ds[['co2_med', 'co2_med_std']]
        
    ds.to_zarr(file_name_cache, consolidated=True)
    return ds.compute()


def flux_inversions_cumsum_lat(clobber=False):
    """return a dataset with the cumlative integral of flux with latitude"""
    ds_cum_cache_file = f'{cache_dir}/sfco2-inversions-cumsum-lat.nc'
    
    if os.path.exists(ds_cum_cache_file) and not clobber:
        ds_cum = xr.open_dataset(ds_cum_cache_file)

    else:
        dsets_cum = {}
        for model in sfco2_ocn_inversions:
            mobj = models.Model(model)
            ds_list = []
            for lat_range in flux_lat_ranges:
                ds = mobj.open_derived_dataset(
                        'flux_ts_monthly', lat_range=lat_range
                    ).sel(time=slice('1998', '2020'))

                with xr.set_options(keep_attrs=True):                
                    dss = ds.groupby('time.season').mean('time').compute()
                ds_list.append(dss)

            dsets_cum[model] = xr.concat(ds_list, 
                                        dim=xr.DataArray(
                                            [rng[-1] for rng in flux_lat_ranges], 
                                            dims=('lat'), 
                                            name='lat')
                                       )
        model = xr.DataArray(sfco2_ocn_inversions, dims=('model'), name='model')
        ds_cum = xr.concat([dsets_cum[m].SFCO2_OCN for m in sfco2_ocn_inversions], dim=model).to_dataset()
        ds_cum.to_netcdf(ds_cum_cache_file)
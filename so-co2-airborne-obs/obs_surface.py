import os
import sys
import yaml
import warnings

import cftime
import calendar 

import numpy as np
import pandas as pd
import xarray as xr

import matplotlib.pyplot as plt

import util

start_date = '1998-12-01'
end_date = '2020-03-01'

southern_ocean_stn_list = ['CRZ', 'MQA', 'DRP', 'PSA', 'SYO', 'CYA', 'MAA', 'HBA',]# 'BHD', 'CGO']
southern_ocean_stn_list_sf6 = ['HBA', 'SYO', 'PSA', 'DRP', 'CRZ', 'USH']

southern_ocean_records = [
    'HBA_NOAA_flask_CO2',
    'SYO_NOAA_flask_CO2',
    'SYO_TU_insitu_CO2',
    'MAA_CSIRO_flask_CO2',
    'CYA_CSIRO_flask_CO2',
    'PSA_NOAA_flask_CO2',
    'PSA_SIO_O2_flask_CO2',
    'DRP_NOAA_flask_CO2',
    'MQA_CSIRO_insitu_CO2',
    'CRZ_NOAA_flask_CO2',  
]
#     'BHD_NIWA_insitu_CO2',
#     'CGO_CSIRO_insitu_CO2'
for rec in southern_ocean_records:
    assert rec[:3] in southern_ocean_stn_list, 'SO records and SO stn list mismatch'
    

spo_records = [
    'SPO_NOAA_insitu_CO2', 
    'SPO_NOAA_flask_CO2', 
    'SPO_SIO_O2_flask_CO2', 
    'SPO_SIO_CDK_flask_CO2',
    'SPO_CSIRO_flask_CO2'
]

attrs = dict(
    CO2=dict(
        long_name='CO$_2$', 
        units='ppm'
    ),
    SF6=dict(
        long_name='SF$_6$', 
        units='ppt'
    ),    
)  

def data_files(c, model=None):
    """get data file for station data"""
    model = 'obs' if model is None else model        
    if model == 'obs':
        assert c in ['CO2', 'SF6']
        if c == 'SF6':
            file = 'data/britt-R-plotting/SF6/SO_SF6_monthly.txt'
        else:
            #file = 'data/britt-R-plotting/SO_CO2_monthly.txt'
            file = 'data/so-co2-station-data/SO_CO2_monthly.txt'
    else:
        with open('data/model-description.yaml', 'r') as fid:
            model_paths = yaml.safe_load(fid)[model]['obs_data_paths']        
        if c in model_paths:
            key = c
        else:
            print(c)
            print(model)
            print(model_paths)
            return 

        file = f'{model_paths[key]}/SO_CO2_monthly.txt'    
    assert os.path.exists(file), f'missing {file}'
    return file


def get_stn_info(constituent='CO2'):
    if constituent=='CO2':
        file_info = 'data/so-co2-station-data/SO_CO2_locations.yaml'
    elif constituent=='SF6':
        file_info = 'data/so-sf6-station-data/SO_SF6_locations.yaml'
    else:
        raise ValueError(f'unknown constituent: {constituent}')
        
    with open(file_info, 'r') as fid:
        df = pd.DataFrame(yaml.safe_load(fid)).transpose().drop('include', axis=1)

    df = df.rename(columns={'stncode': 'stn'})       
    df.index = [f'{s}_{constituent}' for s in df.index]
    df['constituent'] = constituent
    assert df.index.is_unique, (
        'non-unique index'
    )
    
    return df

    
def read_stndata(file):
    if '_CO2_' in file:
        constituent = 'CO2'
    elif '_SF6_' in file:
        constituent = 'SF6'
    else:
        raise ValueError('unknown constituent')
        
    df = pd.read_csv(file, header=0, sep='\s+', na_values=[9.96921e+42, 'NA'])

    day = lambda row_year_mon: util.eomday(row_year_mon[0], row_year_mon[1]) / 2
    df['day'] = df[['year', 'month']].apply(day, axis=1)

    df['date'] = pd.to_datetime(df[['year', 'month', 'day']])
    df['year_frac'] = util.year_frac(df.year.to_numpy(), df.month.to_numpy(), df.day.to_numpy())
    df['polar_year'] = df.year.where(df.month <= 6, df.year + 1)

    time_cols = ['date', 'year', 'month', 'day', 'year_frac', 'polar_year']
    stn_cols = list(set(df.columns) - set(time_cols))
    df = df[time_cols+stn_cols]
    
    df = df.loc[(start_date <= df.date) & (df.date <= end_date)]
    
    stn_cols = list(set(df.columns) - set(time_cols))
    df = df.rename({s: f'{s}_{constituent}' for s in stn_cols}, axis=1)
    
    return df.set_index('date')
    

def to_dataset(
    stninfo, stndata, constituent, 
    station_list=[], plot_coverage=True, dropna=True, unique_stn=True,
    gap_fill=False,
):
    """make an xarray dataset from dataframe"""
    
    if not station_list:
        station_list = list(stninfo.index[stninfo.constituent==constituent])
        station_list = list(filter(lambda s: s in stndata.columns, station_list))

    # sort by latitude
    lat = stninfo.loc[station_list].lat.to_numpy(dtype=np.float)
    I = np.argsort(lat)    
    station_list = np.array(station_list)[I]
    
    data = stndata[station_list]
    
    info = stninfo.loc[station_list]    
    if unique_stn:
        info = info.set_index('stn')
        record_or_stn = 'stn'
    else:
        record_or_stn = 'record'
        
    def visualize_coverage(data):
        """make a plot of data coverage"""
               
        plt.figure(figsize=(15, 6))
        plt.pcolormesh(
            data.index,
            np.arange(0, len(info)+1, 1),
            np.where(np.isnan(data.to_numpy()), np.nan, 1).T,
        )
        plt.yticks(np.arange(0, len(info), 1)+0.5);
        plt.gca().set_yticklabels(info.index);
        plt.grid(True)
        plt.grid(True, axis='x', which='minor')

    # apply gap filling procedure
    if gap_fill:
        data_filled = data.interpolate(method='linear', axis=0, limit=1, limit_direction='backward')
    else:
        data_filled = data
        
    if plot_coverage:
        visualize_coverage(data)
        plt.suptitle(f'{constituent} Station Coverage', fontsize=16, fontweight='bold');

        visualize_coverage(data_filled)
        plt.suptitle(f'{constituent} Station Coverage (gap filled)', fontsize=16, fontweight='bold');    
        plt.show()
    
    # drop NaNs
    if dropna:
        data_filled = data_filled.dropna(axis=0) 

    # assemble DataArray
    return xr.DataArray(
        data_filled.to_numpy(),
        dims=('time', record_or_stn),
        coords={
            'time': xr.DataArray(data_filled.index.values, dims=('time')),
            'year_frac': xr.DataArray(stndata.year_frac.to_numpy().astype(np.float), dims=('time')),
            record_or_stn: xr.DataArray(info.index, dims=(record_or_stn)),
            'institution': xr.DataArray(info.institution, dims=(record_or_stn)),
            'lat': xr.DataArray(info.lat.to_numpy().astype(np.float), dims=(record_or_stn)),
            'lon': xr.DataArray(info.lon.to_numpy().astype(np.float), dims=(record_or_stn)),
            'stncode': xr.DataArray(info.stn, dims=(record_or_stn)),
        },
        name=constituent,
        attrs=attrs[constituent]
    )


def open_surface_co2_data(model, tracer):
    """return a dataset of station data"""
    
    # TODO: clean up this mess!
    if 'TM5' in model and tracer == 'CO2':
        tracer = 'CO2_SUM'
    
    if '+' in tracer or tracer == 'CO2_SUM':
        tracers = ['CO2_OCN', 'CO2_LND', 'CO2_FFF', 'CO2_BKG'] if tracer == 'CO2_SUM' else tracer.split('+')
        file = data_files(tracers[0], model) 
        das_srf = to_dataset(
            get_stn_info('CO2'), 
            read_stndata(file), 
            'CO2', 
            plot_coverage=False, dropna=False, unique_stn=False, gap_fill=False,
        )        
        for subt in tracers[1:]:
            file = data_files(subt, model)
            das_srf += to_dataset(
                get_stn_info('CO2'), 
                read_stndata(file), 
                'CO2', 
                plot_coverage=False, dropna=False, unique_stn=False, gap_fill=False,
            )
    else:
        file = data_files(tracer, model)    
        das_srf = to_dataset(
            get_stn_info('CO2'), 
            read_stndata(file), 
            'CO2', 
            plot_coverage=False, dropna=False, unique_stn=False, gap_fill=False,
        )
        
    # swap MQA_CSIRO_flask_CO2 for MQA_CSIRO_insitu_CO2
    if model != 'obs':
        assert 'MQA_CSIRO_flask_CO2' in das_srf.record
        assert 'MQA_CSIRO_insitu_CO2' not in das_srf.record
        ndx = np.where(das_srf.record == 'MQA_CSIRO_flask_CO2')[0]
        record = das_srf.record.copy()
        record.values[ndx] = 'MQA_CSIRO_insitu_CO2'
        das_srf['record'] = record

    return das_srf
    

def fill_gaps_in_SPO(da_co2):
    da_co2_out = da_co2.copy()
    
    idx_record = np.where(da_co2.record == 'SPO_NOAA_insitu_CO2')[0]
    idx_time = np.append(
        np.where(da_co2.time == np.datetime64('2001-01-15'))[0],
        np.where(da_co2.time == np.datetime64('2001-02-14'))[0]
    )
    idx_time_edges = np.append(
        idx_time[0]-1,
        idx_time[-1]+1,
    )
    da_co2_out[idx_time[0], idx_record] = da_co2.sel(record='SPO_NOAA_flask_CO2').isel(time=idx_time[0])
    da_co2_out[idx_time[1], idx_record] = da_co2.sel(record='SPO_SIO_O2_flask_CO2').isel(time=idx_time[1])    
    return da_co2_out

    
def filter_outliers(da, verbose=False, return_index=False):
    da_mean = da.mean('time')
    da_std = da.std('time')
    
    keep = (da_mean - 3 * da_std <= da) & (da <= da_mean + 3 * da_std)
    
    if verbose:
        print('-'*80)
        print('filtering outliers: n points removed')
        for record in da.record.values:
            n = da.sel(record=record).notnull().sum().values
            n_removed = n - keep.sel(record=record).sum().values
            print(f'\t{record}: {n_removed}, ({100. * n_removed/n:0.2f}%)')
        print('-'*80)
        
    if return_index:
        return keep
    else:
        return da.where(keep)    


def seasonal_uncertainty(das_srf, season=None, verbose=False):
    # compute difference of SPO records from SPO mean
    das_spo_a = das_srf.sel(record=spo_records) - das_srf.sel(record=spo_records).median('record')
    
    seasons = ['DJF', 'MAM', 'JJA', 'SON']
    stn_errors = []
    for season in seasons:
        das_spo_a_djf = util.ann_mean(das_spo_a.to_dataset(), season=season, time_bnds_varname=None, n_req=2)
        stn_errors.append(np.float(das_spo_a_djf.CO2.mean('time').std('record', ddof=1).values))

    if season is None:
        stn_error = max(stn_errors)
    else:
        assert season in seasons
        ndx = seasons.index(season)
        stn_error = stn_errors[ndx]
        
    n_stn = len(southern_ocean_records)
    n_rep = 2 # there are two stations with co-located records
    
    obs_gradient_std = np.sqrt(
        stn_error**2 +
        (n_stn - n_rep) * stn_error**2 / n_stn**2 + 
        (2. * n_rep * stn_error**2) / ((2. * n_stn)**2)
    )
    
    if verbose:
        print('-'*60)
        print(f'stn_error = {stn_error:0.4f} ppm')
        print([f'{s}: {e:0.4f}' for s, e in zip(seasons, stn_errors)])

        print('-'*60)
        print(f'SO-SPO seasonal gradient error = {obs_gradient_std:0.4f} ppm')
        print('-'*60)    
    return obs_gradient_std


def compute_DCO2y(da_srf, season):
    """compute the gradient metric from monthly surface data"""
    
    warnings.filterwarnings(action='ignore', message='Mean of empty slice')
    
    if season in ['DJF', 'MAM', 'JJA', 'SON']:
        ds = util.ann_mean(da_srf.to_dataset(), season=season, time_bnds_varname=None)
        ds['time'] = ds.time + util.season_yearfrac[season]
    elif season in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]:
        months = (da_srf.time.values.astype('datetime64[M]').astype(int) % 12 + 1).astype(int)
        ndx = np.where(months == season)[0]
        ds = da_srf.isel(time=ndx).to_dataset()
        ds['time'] = util.year_frac(*util.datetime64_parts(ds.time))
    for rec in southern_ocean_records:
        assert rec in ds.record, f'missing {rec}'
    
    assert 'SPO_NOAA_insitu_CO2' in ds.record, "missing 'SPO_NOAA_insitu_CO2'"
    return (
        (ds.sel(record=southern_ocean_records).groupby('stncode').mean('record') - 
         ds.sel(record='SPO_NOAA_insitu_CO2')).mean('stncode')
    )    
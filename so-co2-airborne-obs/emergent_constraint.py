import os
import sys
import contextlib
import yaml

import pprint
import warnings

from itertools import product
from toolz import curry

import pickle

from scipy.optimize import curve_fit
import numpy as np
import pandas as pd
import xarray as xr
import dask

import matplotlib.pyplot as plt

import config
import figure_panels
import models
import obs_aircraft
import obs_surface

import regression_models
import util

project_tmpdir = f"{config.get('project_tmpdir')}/cache-emergent-constraint"

campaign_info = obs_aircraft.get_campaign_info(verbose=False)

cache_path_pickles = f'{project_tmpdir}/pickles'
os.makedirs(cache_path_pickles, exist_ok=True)


def load_data(model_tracer_list, profiles_only=True, clobber=False):
    """Load the data to support the emergent constraint calculation, applying some filters.
    This function caches it's own output, which is simply read back in if the cache exists.
    
    Parameters
    ----------
    model_tracer_list: list of tuples
      The models and their tracers to use, i.e. [(CT2017, "CO2_OCN"), (CT2019B, "CO2_OCN"), ...]
    
    profiles_only : boolean
      Use only data collected while "profiling."
      
    clobber : boolean
      Clear cache and recompute.
      
    Returns
    -------
    dfs_obs : dict
      Dictionary of observational data contained in pandas.DataFrame's.
      
    dfs_model: dict
      Dictionary of simulated observations contained in pandas.DataFrame's.    
    """

    cache_path = f'{project_tmpdir}/inputdata'
    os.makedirs(cache_path, exist_ok=True)

    lat_lo_aircraft = -90.
    lat_hi_aircraft = -15.
    read_cache = {}

    def _load_data_aircraft(m, tracer='CO2'):
        """load data for a model m"""
        cache_file=f'{cache_path}/aircraft_data.{m}-{tracer}.{lat_lo_aircraft}_{lat_hi_aircraft}.profile-only={profiles_only}.csv.gz'
        if os.path.exists(cache_file) and clobber:
            os.remove(cache_file)

        if os.path.exists(cache_file):
            df = pd.read_csv(cache_file)
            read_cache[m] = True

        else:
            df = obs_aircraft.open_aircraft_data(m, tracer)
            sel_loc = obs_aircraft.groups_select_profiles(
                df, lat_lo_aircraft, lat_hi_aircraft, profiles_only=profiles_only
            )
            df = df.loc[sel_loc]
            df.to_csv(cache_file, index=False)
            read_cache[m] = False

        return df

    dfs_obs = {m: _load_data_aircraft(m) for m in ['pfp', 'medusa', 'obs',]}        
    dfs_model = {
        f'{m}-{t}': _load_data_aircraft(m, t) 
        for m, t in ensure_components(model_tracer_list)
    }

    return dfs_obs, dfs_model


def load_data_surface(model_tracer_list):
    """Return a dictionary of `xarray.DataArray`'s with observations 
    and simulated observations.
    
    Parameters
    ----------
    model_tracer_list: list of tuples
      The models and their tracers to use, i.e. [(CT2017, "CO2_OCN"), (CT2019B, "CO2_OCN"), ...]
     
    Returns
    -------
    das_srf : dict
      Dictionary of `xarray.DataArray`'s with model and observational data. 
      Keys are constructed from each `model_tracer_list` element as "{model}-{tracer}".
    """
    return {
        f'{m}-{t}': obs_surface.open_surface_co2_data(m, t)
        for m, t in [('obs', 'CO2')] + ensure_components(model_tracer_list)
    }


def ensure_components(model_tracer_list):
    """If a CO2 flavor is detected for a model, ensure all flavors are present."""
    co2_components = {'CO2', 'CO2_OCN', 'CO2_LND', 'CO2_FFF'}
    models = set(m for m, t in model_tracer_list)
    new_list = []
    for model in models:
        m_tracers = set(t for m, t in model_tracer_list if m == model)
        if m_tracers.intersection({'CO2_LND+CO2_FFF'}):
            new_list.extend((model, t) for t in co2_components.union(m_tracers))
        else:
            new_list.extend([(model, t) for t in m_tracers])
    return new_list


def get_parameters(spec='default'):
    """Return the parameters for computing the emergent constraint from aircraft data.
    
    Parameters
    ----------
    spec : string
      Specification of the constraint option in EC-input.yaml.  
      Only one parameter set is suppored, so `spec='default'` is the only support option.
    
    Returns
    -------
    air_parms : dict
      Dictionary with the parameters for computing the aircraft constraint.
      Example:
      
      .. code-block:: python
      
         air_parms = {
            'ubin': 300.0,
            'lbin': 280.0,
            'udθ': 10.0,
            'lbin_as_upper_bound': True,
            'ubin_as_lower_bound': False,
            'gradient_lat_range': (-90.0, -45.0),
            'flux_lat_range': (-90.0, -45.0),
            'flux_memory': 90,
            'fit_groups': ('DJF', 'MAMJJASON'),
          }

    """
    
    assert spec == "default", "`spec='default'` is the only supported option"
    with open('EC-input.yaml', 'r') as fid:
        input_dict = yaml.safe_load(fid)['parameters']

    def get_value(key):
        if key in ['gradient_lat_range', 'flux_lat_range',]:
            return tuple([float(v) for v in input_dict[key]['value']])
        elif key in ['fit_groups']:
            return tuple([v for v in input_dict[key]['value']])        
        elif key == 'flux_memory':
            return int(input_dict[key]['value'])
        else:
            return input_dict[key]['value']
    
    return {k: get_value(k) for k, d in input_dict.items()}
    
    
def get_model_tracer_lists(spec):
    """Return the `model_tracer_list` specified in the emergent constraint
    input file: EC-input.yaml.
    
    Parameters
    ----------
    
    spec : string
      Specification of the constraint option in EC-input.yaml.  
      Options include: 'ocean_constraint', 'total_constraint'
      
    Returns
    -------
    
    model_tracer_list: list of tuples
      The models and their tracers to use, i.e. [(CT2017, "CO2_OCN"), (CT2019B, "CO2_OCN"), ...]
      
    """
    
    
    with open('EC-input.yaml', 'r') as fid:
        input_dict = yaml.safe_load(fid)['model_input_lists']

    assert spec in input_dict, f'unknown spec: {spec}'
    kwargs = input_dict[spec]
    for list_i in ['model_tracer_list', 'model_tracer_ext_list',]:
        if list_i in kwargs:
            kwargs[list_i] = [tuple(mt) for mt in kwargs[list_i]]
        else:
            kwargs[list_i] = []

    if 'model_list_sfco2_lnd' not in kwargs:
        kwargs['model_list_sfco2_lnd'] = []

    return kwargs


def get_model_families():
    with open('EC-input.yaml', 'r') as fid:
        return yaml.safe_load(fid)['model_families']
    
    
def get_model_groups():
    input_dict = get_model_families()
    groups = list(set(input_dict.values()))
    model_groups = {k: [] for k in groups}
    for model, group in input_dict.items():
        model_groups[group].append(model)
    return model_groups                


def get_monthly_time_weights(flux_time, campaign_time_last, flux_memory):
    """
    Get temporal-weighting function to compute weighted mean of monthly fluxes.
    """

    start = campaign_time_last - np.timedelta64(flux_memory-1, 'D')
    end = campaign_time_last

    time_range = xr.cftime_range(start.astype(str), end.astype(str), freq='D')

    flux_time_year, flux_time_month, _ = util.datetime64_parts(flux_time)

    month_weights = np.zeros(len(flux_time))
    for i, t in enumerate(flux_time.values):
        f_year = flux_time_year[i]
        f_mon =  flux_time_month[i]
        ndx = np.where((time_range.month == f_mon) & (time_range.year == f_year))[0]
        month_weights[i] = len(ndx)

    if month_weights.sum() == 0:
        return xr.DataArray(month_weights, dims=('time'))

    if month_weights.sum() < flux_memory:
        return xr.DataArray(np.zeros(len(flux_time)), dims=('time'))

    return xr.DataArray(month_weights / month_weights.sum(), dims=('time'))


def compute_time_weighted_flux(flux, time_weights=None):
    """compute time-weighted fluxes"""
    if time_weights is None:
        return flux.mean('time')
    else:
        np.testing.assert_approx_equal(time_weights.sum('time'), 1., significant=7)
        assert len(time_weights) == len(flux.time), len(flux.time)

        return (flux * time_weights).sum('time')


def get_model_flux(campaign, ds_flux, flux_memory,
                   campaign_time_point='center'):
    """
    Compute time-weighted model flux
    """
    assert campaign_time_point in ['center', 'end']

    # special handling since we split 'ORCAS' into 2, but sometimes want to see it as one
    if campaign == 'ORCAS':
        if campaign_time_point == 'center':
            # close enough...nothing really depends on this assessment
            campaign_time_last = campaign_info['ORCAS-J']['time_bound'][-1]

        elif campaign_time_point == 'end':
            campaign_time_last = campaign_info['ORCAS-F']['time_bound'][-1]

    else:
        if campaign_time_point == 'center':
            campaign_time_last = campaign_info[campaign]['time']

        elif campaign_time_point == 'end':
            campaign_time_last = campaign_info[campaign]['time_bound'][-1]

    time_weights = get_monthly_time_weights(ds_flux.time, campaign_time_last, flux_memory)

    if time_weights.sum() == 0:
        return np.nan

    return compute_time_weighted_flux(
            ds_flux.FLUX,
            time_weights)


def aircraft_flux_v_gradient(campaign_sel_list, theta_bins,
                             gradient_lat_range, flux_memory,
                             campaign_time_point,
                             bin_aggregation_method,
                             model_tracer_list, dfs_model, dsets_fluxes):
    """
    Return a DataFrame with the vertical gradient and fluxes from the models.
    """

    lines = []
    for c in campaign_sel_list:
        for m, t in model_tracer_list:
            df = obs_aircraft.campaign_gradients(
                dfs_model[f'{m}-{t}'], [c], theta_bins, gradient_lat_range,
                bin_aggregation_method=bin_aggregation_method
            )
            model_flux = np.float(get_model_flux(c, dsets_fluxes[f'{m}-{t}'], flux_memory))
            if np.isnan(model_flux):
                continue

            model_gradient = df.loc[c].gradient_mean
            lines.append(dict(
                campaign=c,
                model=m,
                field=t,
                gradient=model_gradient,
                flux=model_flux,
            ))

    return pd.DataFrame(lines)


def surface_flux_v_gradient(das_srf, dsets_fluxes_mmm, season,
                            time_slice, model_tracer_list):
    """
    Return a DataFrame with the SO-SPO gradient and fluxes from the models as well as
    the observed gradient and its error metric.
    """

    ds_grad = {k: obs_surface.compute_DCO2y(da, season)
               for k, da in das_srf.items() if k != 'obs-CO2'}

    lines = []
    for m, tracer in model_tracer_list:
        key = f'{m}-{tracer}'
        flux = dsets_fluxes_mmm[key].sel(time=time_slice).FLUX
        gradient = ds_grad[key].sel(time=time_slice).CO2
        if len(flux.time) < 3:
            continue

        got_data = ~(np.isnan(flux).all() & np.isnan(gradient).all())

        flux, gradient = xr.align(flux, gradient)
        if got_data and np.isnan(flux).all() & np.isnan(gradient).all():
            raise ValueError('coordinate alignment failed')

        lines.append(dict(
            model=m,
            field=tracer,
            gradient=np.float(gradient.mean('time').values),
            gradient_std=np.float(gradient.std('time').values),
            flux=np.float(flux.mean('time').values),
            flux_std=np.float(flux.std('time').values),
        ))

    return pd.DataFrame(lines)


def surface_obs_gradient(da_srf, season, time_slice):
    """Compute the horizontal gradient from surface observations."""

    obs_gradient_mean = obs_surface.compute_DCO2y(
        da_srf, season
    ).sel(time=time_slice).CO2.mean('time').values

    # I am not subsetting in time here; we don't feel that we have a
    # clear enough handle onanalytical uncertainty to compute it in a
    # time-varying sense, so we just compute an estimate for the whole record
    obs_gradient_std = obs_surface.seasonal_uncertainty(da_srf, season=season, verbose=False)

    return obs_gradient_mean, obs_gradient_std


def estimate_flux(fit_dict, obs_gradient_mean, obs_gradient_std, flux_correction=None, flux_correction_std=None):
    """
    Compute estimate of observed flux and
    its associated uncertainty using bootstrap method.
    """
    N = int(1e6)

    flux = fit_dict['beta'][0] * obs_gradient_mean + fit_dict['beta'][1]

    xp = np.random.normal(
        loc=obs_gradient_mean,
        scale=obs_gradient_std,
        size=N
    )
    beta = np.random.multivariate_normal(
        mean=fit_dict['beta'],
        cov=fit_dict['cov_beta'],
        size=N,
    )
    flux_error = (beta[:, 0] * xp + beta[:, 1]).std(ddof=1)

    # apply correction
    if flux_correction is not None:
        assert flux_correction_std is not None
        flux = flux - flux_correction
        flux_error = np.sqrt(flux_error**2 + flux_correction_std**2)

    return flux, flux_error


def add_scaled_correction(fit_dict, ch4_gradient_mean, ch4_gradient_std):
    """add correction accounting for uncertainty in slope and value"""
    N = int(1e6)
    xp = np.random.normal(
            loc=ch4_gradient_mean,
            scale=ch4_gradient_std,
            size=N
        )

    beta = np.random.multivariate_normal(
        mean=fit_dict['beta'],
        cov=fit_dict['cov_beta'],
        size=N,
    )

    correction_mean = fit_dict['beta'][0] * ch4_gradient_mean
    correction_std = (beta[:, 0] * xp).std(ddof=1)

    return correction_mean, correction_std

def compute_constraint_fit(df, weight_by_sd_iav=False, weight_by_model_family=False):
    """Use model gradient/flux DataFrame to fit ODR model"""
    if weight_by_sd_iav and weight_by_model_family:
        raise ValueError('incompatible options')
        
    x = df.gradient.values
    y = df.flux.values
    k = np.isnan(x) | np.isnan(y)

    if np.sum(~k) < 3:
        fit = regression_models.linreg_odr()
        fit_dict = {f'fit_{k}': v for k, v in fit.to_dict().items()}
        fit_dict.update({f'yxfit_{k}': v for k, v in fit.to_dict().items()})        

    else:
        xerr, yerr = None, None
        if weight_by_sd_iav and 'gradient_std' in df and 'flux_std' in df:
            xerr = 1. / df.gradient_std.values[~k]
            flux_std = df.flux_std.values[~k]
            flux_std[flux_std==0.] = flux_std.mean()
            yerr = 1. / flux_std
        
        elif weight_by_model_family:
            # count number of models in each model family  
            models = df.model.values
            model_families = get_model_families()
            family_counts = {k: 0 for k in set(model_families.values())}
            
            for m in models:
                family = model_families[m]
                family_counts[family] += 1
            
            wgts = []
            for m in models:
                family = model_families[m]
                wgts.append(1./family_counts[family])
            xerr = wgts
            yerr = wgts
            
        fit_xy = regression_models.linreg_odr(x[~k], y[~k], xerr=xerr, yerr=yerr)
        fit_yx = regression_models.linreg_odr(y[~k], x[~k], xerr=yerr, yerr=xerr)
        
        fit_dict = {f'fit_{k}': v for k, v in fit_xy.to_dict().items()}
        fit_dict.update({f'yxfit_{k}': v for k, v in fit_yx.to_dict().items()})
        
    return fit_dict

    
def get_fit_dict(series, prefix=''):
    """
    Pull out all columns starting with "fit_" from pd.Series.
    """
    assert isinstance(series, pd.core.series.Series), (
        'fit_dict: require series'
    )
    if prefix:
        return {
            k.split(f'{prefix}fit_')[-1]: v 
            for k, v in series.items() if f'{prefix}fit_' in k
        }
    else:
        return {
            k.split(f'fit_')[-1]: v 
            for k, v in series.items() if f'fit_' in k and 'yxfit' not in k
        }
        

class surface_constraint(object):
    """This object computes the emergent flux constraint based on surface data.
      
    Parameters
    ----------
    
    periods : list of tuples
      A list of tuples with year ranges over which to compute the constraint; e.g. 
      [(1999, 2020), (1999, 2009), (2009, 2020)].
    
    flux_lat_range: [float, float]
      The latitude range over which to integrate fluxes.
 
    weight_by_sd_iav : boolean
      Use a weighted regression to estimate constraint relationship, with the weights set by the 
      standard deviation of interannual variability.
    
    seasons : list
      The seasons over which to compute the constraint. E.g. ["DJF", "JJA"].
    
    fit_period: string
      The year-range in `periods` to use as the "calibrated" constraint. E.g., "1999-2020".
    
    das_srf: dict
      Dictionary of observational data contained in pandas.DataFrame's returned from `load_data_surface`.
         
    model_tracer_list: list of tuples
      The models and their tracers to use, i.e. [(CT2017, "CO2_OCN"), (CT2019B, "CO2_OCN"), ...]

    model_tracer_ext_list : list of tuples
      A list of tuples specifying the (model, tracer) pairs to use to estimate "external" 
      contributions to the observed gradient. I.e., these data are used to correct the 
      observed estimates of the gradient for land and fossil contributions.
      
    model_list_sfco2_lnd : list of tuples
      A list of (model, tracer) pairs used to correct the resulting flux estimate for in-region 
      land and fossil fuel fluxes. This is only used if the contraint is based on total CO2.
    """
    def __init__(self,
                 periods,
                 flux_lat_range,
                 weight_by_sd_iav,
                 seasons,
                 fit_period,
                 das_srf,
                 model_tracer_list,
                 model_list_sfco2_lnd=[],
                 model_tracer_ext_list=None,
                ):
        """set up the object"""

        # incoming parameters
        self.periods = periods
        self.flux_lat_range = flux_lat_range
        self.weight_by_sd_iav = weight_by_sd_iav
        self.seasons = seasons
        self.fit_period = fit_period
        self.model_tracer_list = model_tracer_list
        self.model_tracer_ext_list = model_tracer_ext_list
        self.model_list_sfco2_lnd = model_list_sfco2_lnd

        assert fit_period in [self.period_str(p) for p in periods], (
            '`fit_period` must be in periods'
        )

        # compute gradients
        self.hg_obs = self._compute_surface_gradient(das_srf['obs-CO2'])
        self._compute_monthly_gradients(das_srf)
                
        # get fluxes
        dsets_fluxes = get_dset_fluxes(self.model_tracer_list, self.flux_lat_range)
        dsets_fluxes_mmm = get_dset_fluxes_mmm(dsets_fluxes)

        df_data, df_fits = self._compute_surface_constraint(
            das_srf, dsets_fluxes_mmm,
        )
        self.df_data = df_data
        self.df_fits = df_fits

        self._surface_flux = None
        self._p_pcov = None

    def _compute_surface_gradient(self, das_srf_obs):
        lines = []
        for period in self.periods:
            for season in self.seasons:
                obs_gradient_mean, obs_gradient_std = surface_obs_gradient(
                            das_srf_obs, season, slice(period[0], period[1])
                        )
                lines.append(dict(
                    period=self.period_str(period),
                    season=season,
                    gradient_mean=obs_gradient_mean,
                    gradient_std=obs_gradient_std,
                )
                )
        return pd.DataFrame(lines).set_index(['period', 'season'])

    def _compute_monthly_gradients(self, das_srf):
        """compute all the compontents of the surface constraint"""
        lines = []

        model_tracer_list = [('obs', 'CO2')] + ensure_components(
            self.model_tracer_list + self.model_tracer_ext_list
        )        
        for month in range(1, 13):
            for model, tracer in model_tracer_list:
                key = f'{model}-{tracer}'
                if key not in das_srf:
                    print(f'missing {key}, cannot compute monthly gradient')
                    continue
                ds_grad = obs_surface.compute_DCO2y(das_srf[key], month)
                for period in self.periods:
                    gradient = ds_grad.sel(time=slice(period[0], period[1])).CO2                    
                    result = dict(
                        model=model,
                        tracer=tracer,
                        period=self.period_str(period),                        
                        month=month,
                        season=get_season(month),
                        time_bounds=period,                        
                        gradient=np.float(gradient.mean('time').values),
                        gradient_std=np.float(gradient.std('time').values),                        
                    )
                    lines.append(result)
        self.df_gradients_mon = pd.DataFrame(lines).set_index(
            ['model', 'tracer', 'period', 'month',]
        )
    
    def _compute_surface_constraint(self, das_srf, dsets_fluxes_mmm):
        """compute all the compontents of the surface constraint"""
        
        df_list = []
        lines = []
        for period in self.periods:
            for season in self.seasons:
                df = surface_flux_v_gradient(das_srf, dsets_fluxes_mmm[season],
                                             season, slice(period[0], period[1]),
                                             self.model_tracer_list,
                                            )
                df['time_bounds'] = [period]*len(df)
                df['period'] = [self.period_str(period)]*len(df)
                df['season'] = season
                df_list.append(df)

                result = dict(
                    season=season, period=self.period_str(period), time=np.mean(period)
                )
                result.update(
                    compute_constraint_fit(df, weight_by_sd_iav=self.weight_by_sd_iav)
                )
                lines.append(result)

        df_srf = pd.concat(df_list).set_index(['period', 'season', 'model', 'field'])
        df_fits_srf = pd.DataFrame(lines).set_index(['period', 'season'])
        return df_srf, df_fits_srf

    @property
    def surface_flux(self):
        """Return a pandas.DataFrame with the surface flux estimates for each time 
        range defined in `periods`.
        """
        if self._surface_flux is None:
            self._surface_flux = self._compute_surface_flux()
        return self._surface_flux

    def _compute_surface_flux(self):

        flux_land = None
        flux_land_std = None
        
        if self.model_tracer_ext_list:
            df_ext = util.pd_xs_list(self.df_gradients_mon, self.model_tracer_ext_list, 
                                 level=('model', 'tracer'))
            series_ext_grad = (df_ext
                           .groupby(['model', 'period', 'season'])
                           .mean()
                           .groupby(['period', 'season'])
                           .gradient.mean()
                          )
            series_ext_grad_std = (df_ext
                               .groupby(['model', 'period', 'season'])
                               .mean()
                               .groupby(['period', 'season'])
                               .gradient.std()
                              )        
        
        lines = []
        for season in self.seasons:

            if self.model_list_sfco2_lnd:
                ds_sfco2_lnd = get_dset_fluxes_land(
                    self.flux_lat_range, self.model_list_sfco2_lnd, season_avg=True)[season]

            for period in self.periods:
                
                obs_gradient_mean, obs_gradient_std = self.hg_obs.loc[(self.period_str(period), season)]
                
                ext_gradient_mean = 0.
                ext_gradient_std = 0.
                if self.model_tracer_ext_list:
                    ext_gradient_mean = series_ext_grad.loc[(self.period_str(period), season)]
                    ext_gradient_std = series_ext_grad_std.loc[(self.period_str(period), season)]
                
                corrected_gradient_mean = obs_gradient_mean - ext_gradient_mean
                corrected_gradient_std = np.sqrt(obs_gradient_std**2 + ext_gradient_std**2)
                
                if self.model_list_sfco2_lnd:
                    da = ds_sfco2_lnd.FLUX.sel(time=slice(period[0], period[1])).mean('time')
                    flux_land = np.float(da.mean('model').values)
                    flux_land_std = np.float(da.std('model').values)
    
                fit_dict = get_fit_dict(self.df_fits.loc[self.fit_period, season])

                flux_nocorr, flux_error_nocorr = estimate_flux(
                    fit_dict, obs_gradient_mean, obs_gradient_std, 
                )

                flux, flux_error = estimate_flux(
                    fit_dict, corrected_gradient_mean, corrected_gradient_std, 
                    flux_land, flux_land_std,
                )

                lines.append(dict(
                    period=self.period_str(period),
                    season=season,
                    time_bnds=period,
                    obs_grad=obs_gradient_mean,
                    obs_grad_err=obs_gradient_std,
                    corrected_grad=corrected_gradient_mean,
                    corrected_grad_err=corrected_gradient_std,
                    flux=flux,
                    flux_error=flux_error,
                    flux_land=flux_land,
                    flux_land_error=flux_land_std,
                    flux_uncorrected=flux_nocorr,
                    flux_error_uncorrected=flux_error_nocorr,                    
                ))
        return pd.DataFrame(lines).set_index(['period', 'season'])

    def period_str(self, y1_y2):
        return '-'.join([f'{y:04d}' for y in y1_y2])


class aircraft_constraint(object):
    """This object computes the aircraft constraint.

    This is what happens here:
    
    1. Initialize object with parameters defining the computation;
    
    2. Compute observed gradient and error estimate from campaigns;
    
    3. Group campaigns into `fit_groups` and compute associated DataFrame;
    
    4. Compute fluxes for each campaign based on the associated `fit_group`
    
    5. Fit a harmonic function to the campaign flux estimates, use this fit to generate an 
    annual mean estimate with associated uncertainty.
       
    Parameters
    ----------
    ubin : float
      The value of θ on which to center the upper bin.
    
    lbin : float
      The value of θ on which to center the lower bin.    
    
    udθ : float
      The width in θ units of the upper bin.
    
    ldθ : float
      The width in θ units of the lower bin.    
    
    gradient_lat_range : [float, float]
       The latitude range over which to compute the vertical gradient.
    
    flux_memory: float
      The time in days over which to average air-sea fluxes in the calculation.
    
    flux_lat_range: [float, float]
      The latitude range over which to integrate fluxes.
    
    campaign_time_point: string
      Acceptable values: ["center", "end"]; where to set the campaign's time-axis value.
      
    bin_aggregation_method: string
      Acceptable values: ["mean", "median"]; how to aggregate aircraft data in the θ bins.
      
    fit_groups: iterable
      The groups by which to aggegrate campaigns.
      
    model_tracer_list: list of tuples
      The models and their tracers to use, i.e. [(CT2017, "CO2_OCN"), (CT2019B, "CO2_OCN"), ...]
    
    dfs_obs: dict
      Dictionary of observational data contained in pandas.DataFrame's returned from `load_data`.
         
    dfs_model: dict
      Dictionary of simulated observations contained in pandas.DataFrame's returned from `load_data`.
      
    model_groups : dict
      A dictionary specifying a grouping of the models; the code returns a model-group-weighted fit.
    
    model_tracer_ext_list : list of tuples
      A list of tuples specifying the (model, tracer) pairs to use to estimate "external" 
      contributions to the observed gradient. I.e., these data are used to correct the 
      observed estimates of the gradient for land and fossil contributions.
      
    model_list_sfco2_lnd : list of tuples
      A list of (model, tracer) pairs used to correct the resulting flux estimate for in-region 
      land and fossil fuel fluxes. This is only used if the contraint is based on total CO2.
      
    methane_theta_lbound : float
      Specifies the θ value above which to relate CH4 and CO2. This is only relevant if 
      `use_methane_gradient_correction=True`, which is not the case by default. This option
      was something we explored, but did not feel it well justified, so did not implement in the 
      final computation.
      
    use_methane_gradient_correction: boolean
      Use vertical gradients of CH4 to correct observed gradient; this option is not scientifically 
      justified and should not be used.
      
    lbin_as_upper_bound : boolean
      Interpret the value of `lbin` as the upper bound for the lower bin, i.e., aggegrate 
      data where `θ < lbin`. If `lbin_as_upper_bound=True`, then `ldθ` is ignored.
      
    ubin_as_lower_bound : boolean
      Interpret the value of `ubin` as the lower bound for the upper bin, i.e., aggegrate 
      data where `θ > ubin`. If `ubin_as_lower_bound=True`, then `udθ` is ignored.
      
    restrict_groups : boolean
      Only produce the analysis for `fit_groups`, not all possible groups (as determined by 
      those groups defined in the code).

    
    """
    def __init__(self,
                 ubin, lbin, udθ, ldθ,
                 gradient_lat_range,
                 flux_memory,
                 flux_lat_range,
                 campaign_time_point,
                 bin_aggregation_method,
                 fit_groups,
                 model_tracer_list,
                 dfs_obs,
                 dfs_model,
                 model_groups={},
                 model_tracer_ext_list=[],
                 model_list_sfco2_lnd=[],
                 methane_theta_lbound=305.,
                 use_methane_gradient_correction=False,
                 lbin_as_upper_bound=False,
                 ubin_as_lower_bound=False,
                 restrict_groups=False,
                ):
        """Initialize the `aircraft_contraint`."""
        # incoming parameters
        self.theta_bins = obs_aircraft.make_theta_bins(
            lbin, ubin, udθ, ldθ, lbin_as_upper_bound, ubin_as_lower_bound
        )
        self.gradient_lat_range = gradient_lat_range
        self.flux_memory = flux_memory
        self.flux_lat_range = flux_lat_range
        self.campaign_time_point = campaign_time_point
        self.bin_aggregation_method = bin_aggregation_method
        self.fit_groups = fit_groups
        self.model_tracer_list = model_tracer_list
        self.model_groups = model_groups
        self.methane_theta_lbound = methane_theta_lbound
        self.use_methane_gradient_correction = use_methane_gradient_correction

        if model_tracer_ext_list and model_list_sfco2_lnd:
            raise ValueError(
                f'detected both "model_list_sfco2_lnd" and "model_tracer_ext_list"'
            )

        self.model_tracer_ext_list = model_tracer_ext_list
        self.model_list_sfco2_lnd = model_list_sfco2_lnd
        self.restrict_groups = restrict_groups

        self._set_campaign_groups()
        
        self.vg_obs = self.get_campaign_gradients(dfs_obs)
        self.flight_gradients = self.get_flight_gradients(dfs_obs)
        
        self._compute_methane_correction_slope(dfs_obs)
        self._compute_vg_ext(dfs_model)
        self.gradient_summary = self._compute_gradient_summary()
        
        dsets_fluxes = get_dset_fluxes(self.model_tracer_list, self.flux_lat_range)
        self._compute_aircraft_constraint(dfs_model, dsets_fluxes)

        # init properties
        self._campaign_flux = None
        self._p_pcov = None
        self._p_pcov_vg_obs = None
        self._p_pcov_vg_ext = None

    @curry
    def get_flight_gradients(self, dfs):
        """Return a DataFrame with the ∆θCO2 gradient estiamtes for each flight."""
        return obs_aircraft.flight_gradients(
            dfs,
            theta_bins=self.theta_bins,
            gradient_lat_range=self.gradient_lat_range,
            bin_aggregation_method=self.bin_aggregation_method
            )

    @curry
    def get_campaign_gradients(self, dfs):
        """Return a DataFrame with the ∆θCO2 gradient estiamtes for each campaign."""
        return obs_aircraft.campaign_gradients(
            dfs,
            campaign_info.keys(),
            self.theta_bins,
            self.gradient_lat_range,
            bin_aggregation_method=self.bin_aggregation_method,
            constituent='co2',
        )
        
    def _compute_vg_ext(self, dfs_model):
        model_tracer_list_extended = ensure_components(
            self.model_tracer_list + self.model_tracer_ext_list
        )        

        df_list = []
        for m, t in model_tracer_list_extended:
            df = obs_aircraft.campaign_gradients(
                    dfs_model[f'{m}-{t}'],
                    campaign_info.keys(),
                    self.theta_bins,
                    self.gradient_lat_range,
                    bin_aggregation_method=self.bin_aggregation_method,
                )
            df['model'] = m
            df['tracer'] = t
            df = df.reset_index()
            df_list.append(df)
        self.vg_ext = pd.concat(df_list).set_index(['model', 'tracer', 'campaign'])
        
    def _compute_methane_correction_slope(self, dfs_obs):
        """compute methane gradients and co2 fits"""

        self.vg_ch4 = obs_aircraft.campaign_gradients(
            dfs_obs,
            campaign_info.keys(),
            self.theta_bins,
            self.gradient_lat_range,
            bin_aggregation_method=self.bin_aggregation_method,
            constituent='ch4',
            )

        fit_lines = []
        df_list = []
        for c, clist in self.campaign_groups.items():
            co2, ch4 = obs_aircraft.get_property_property(
                df=dfs_obs['obs'],
                campaign_sel_list=clist,
                xname='co2',
                yname='ch4',
                lat_range=self.gradient_lat_range,
                theta_bin=(self.methane_theta_lbound, np.Inf),
                filter_strat=True
            )
            df_list.append(
                pd.DataFrame(dict(co2=co2, ch4=ch4, campaign_group=c))
            )
            fit = regression_models.linreg_odr(ch4, co2)
            result = {'campaign_group': c}
            result.update({f'fit_{k}': v for k, v in fit.to_dict().items()})
            fit_lines.append(result)

        self.df_co2_v_ch4_fits = pd.DataFrame(fit_lines).set_index('campaign_group')
        self.df_co2_v_ch4_data = pd.concat(df_list).set_index('campaign_group')

    def _compute_gradient_summary(self):
        df = (self.vg_obs[['gradient_mean', 'gradient_std']]
              .rename(
                  {'gradient_mean': 'vg_obs', 'gradient_std': 'vg_obs_std'},
                  axis=1,
              )
             )
        df['month'] = [campaign_info[c]['year'] for c in df.index]
        df['month'] = [campaign_info[c]['month'] for c in df.index]
        df['doy'] = [util.day_of_year(np.array([campaign_info[c]['time']]))[0] for c in df.index]
        keep_fields = ['month', 'doy', 'vg_obs', 'vg_obs_std',]
        
        if self.model_tracer_ext_list:
            df2 = util.pd_xs_list(
                self.vg_ext, self.model_tracer_ext_list, level=('model', 'tracer')
            )
            df['vg_ext'] = df2.groupby('campaign').median().gradient_mean.align(df)[0]
            df['vg_ext_std'] = df2.groupby('campaign').std(ddof=1).gradient_mean.align(df)[0]
            keep_fields += ['vg_ext', 'vg_ext_std']

        df['vg_ch4_ppb'] = self.vg_ch4.gradient_mean
        df['vg_ch4_ppb_std'] = self.vg_ch4.gradient_std
        keep_fields += ['vg_ch4_ppb', 'vg_ch4_ppb_std']

        return df[keep_fields]

    def _compute_aircraft_constraint(self, dfs_model, dsets_fluxes):
        lines = []
        lines_wgt = []
        df_list = []
        model_fits = {k: [] for k in self.campaign_groups.keys()}
        for c, clist in self.campaign_groups.items():

            df = aircraft_flux_v_gradient(
                clist, self.theta_bins, self.gradient_lat_range, self.flux_memory,
                self.campaign_time_point, self.bin_aggregation_method,
                self.model_tracer_list,
                dfs_model, dsets_fluxes
            )
            df['campaign_group'] = [c] * len(df)
            df_list.append(df)

            result = {'campaign_group': c}
            result.update(
                compute_constraint_fit(df)
            )
            lines.append(result)

            result = {'campaign_group': c}
            result.update(
                compute_constraint_fit(df, weight_by_model_family=True)
            )
            lines_wgt.append(result)            
            
            if self.model_groups:
                for group_name, model_list in self.model_groups.items():
                    dfg = df.loc[df.model.isin(model_list)]
                    result = dict(
                        model_group=group_name,
                    )

                    result.update(
                        compute_constraint_fit(dfg)
                    )
                    model_fits[c].append(result)

        df_air = pd.concat(df_list).set_index(
            ['campaign_group', 'campaign', 'model', 'field']
        )
        df_fits_air = pd.DataFrame(lines).set_index('campaign_group')
        df_fits_wgts = pd.DataFrame(lines_wgt).set_index('campaign_group')
        
        if self.model_groups:
            df_model_fits = {
                k: pd.DataFrame(model_fits[k]).set_index('model_group')
                for k in self.campaign_groups.keys()
            }
        else:
            df_model_fits = None

        self.df_data = df_air
        self.df_fits = df_fits_air
        self.df_fits_weighted_by_model = df_fits_wgts
        self.df_model_fits = df_model_fits            

    def _set_campaign_groups(self):
        d = dict(
            DJF=[c for c, nf in campaign_info.items() if nf['month'] in [1, 2, 12]],
            JJA=[c for c, nf in campaign_info.items() if nf['month'] in [6, 7, 8]],
            MAM=[c for c, nf in campaign_info.items() if nf['month'] in [3, 4, 5]],
            SON=[c for c, nf in campaign_info.items() if nf['month'] in [9, 10, 11]],
            MAMJJASON=[
                c for c, nf in campaign_info.items() if nf['month'] in [3, 4, 5, 6, 7, 8, 9, 10, 11]
            ],
        )
        d.update({
            mon: [c for c, nf in campaign_info.items() if nf['month'] == n+1] 
            for n, mon in enumerate([
                'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec',
            ])
        })
        if self.restrict_groups:
            self.campaign_groups = {k: v for k, v in d.items() 
                                    if v and k in self.fit_groups}
        else:
            self.campaign_groups = d

    @property
    def campaign_flux(self):
        """Return a DataFrame with the flux estimate for each campaign."""
        if self._campaign_flux is None:
            self._campaign_flux = self._compute_campaign_flux()
        return self._campaign_flux

    def pick_fit_group(self, campaign):
        for group_name, group_clist in self.campaign_groups.items():
            if campaign in group_clist and group_name in self.fit_groups:
                return group_name
        raise ValueError(f'did not find the fit group for {campaign}')

    def _compute_campaign_flux(self):
        """Compute a flux estimate with uncertainty for each campaign."""

        flux_land = None
        flux_land_std = None
        if self.model_list_sfco2_lnd:
            ds_sfco2_lnd = get_dset_fluxes_land(self.flux_lat_range, self.model_list_sfco2_lnd)


        if self.model_tracer_ext_list:
            df = util.pd_xs_list(self.vg_ext, self.model_tracer_ext_list, 
                                 level=('model', 'tracer'))
            df_ext_gradient_mean = df.groupby('campaign').median()['gradient_mean']
            df_ext_gradient_std = df.groupby('campaign').std(ddof=1)['gradient_mean']

        lines = []
        for c, nf in campaign_info.items():
            fit_index = self.pick_fit_group(c)

            fit_dict = get_fit_dict(self.df_fits.loc[fit_index])

            obs_gradient_mean = self.vg_obs.loc[c].gradient_mean
            obs_gradient_std = self.vg_obs.loc[c].gradient_std

            # adjust gradient estimate for contributions from other sources
            # first compute for all options
            ext_gradient_estimates = {}
            if self.use_methane_gradient_correction:
                co2_v_ch4_fit_dict = get_fit_dict(
                    self.df_co2_v_ch4_fits.loc[fit_index]
                )
                ch4_gradient_mean = self.vg_ch4.loc[c].gradient_mean
                ch4_gradient_std = self.vg_ch4.loc[c].gradient_std
                ext_ch4_gradient_mean, ext_ch4_gradient_std = add_scaled_correction(
                    co2_v_ch4_fit_dict, ch4_gradient_mean, ch4_gradient_std
                )
                ext_gradient_estimates['ext_ch4_gradient_mean'] = ext_ch4_gradient_mean
                ext_gradient_estimates['ext_ch4_gradient_std'] = ext_ch4_gradient_std
            
            if self.model_tracer_ext_list:
                ext_gradient_estimates['ext_model_gradient_mean'] = df_ext_gradient_mean.loc[c]
                ext_gradient_estimates['ext_model_gradient_std']= df_ext_gradient_std.loc[c]

            # now apply for option that takes priority
            ext_gradient_mean = 0.
            ext_gradient_std = 0.
            if self.use_methane_gradient_correction:
                ext_gradient_mean = ext_ch4_gradient_mean
                ext_gradient_std = ext_ch4_gradient_std
            elif self.model_tracer_ext_list:
                ext_gradient_mean = df_ext_gradient_mean.loc[c]
                ext_gradient_std = df_ext_gradient_std.loc[c]

            corrected_gradient_mean = obs_gradient_mean - ext_gradient_mean
            corrected_gradient_std = np.sqrt(obs_gradient_std**2 + ext_gradient_std**2)

            # adjust flux estimate for land fluxes within the region
            if self.model_list_sfco2_lnd:
                da = get_model_flux(c, ds_sfco2_lnd, self.flux_memory)
                flux_land = np.float(da.mean('model').values)
                flux_land_std = np.float(da.std('model').values)

            flux_nocorr, flux_error_nocorr = estimate_flux(
                fit_dict, obs_gradient_mean, obs_gradient_std, 
            )
                
            flux, flux_error = estimate_flux(
                fit_dict, corrected_gradient_mean, corrected_gradient_std, 
                flux_land, flux_land_std,
            )

            flux_window_date_bound = [
                campaign_info[c]['time_bound'][-1] - np.timedelta64(self.flux_memory, '1D'),
                campaign_info[c]['time_bound'][-1],
            ]

            doy = util.day_of_year(np.array([campaign_info[c]['time_bound'][-1]]))[0]
            doy_start = doy - self.flux_memory
            doy_mid = np.mean([doy_start, doy])

            if doy_start < 0:
                doy_start = 365 + doy_start
            if doy_mid < 0:
                doy_mid = 365 + doy_mid

            year = campaign_info[c]['year']
            month = campaign_info[c]['month']
            day = campaign_info[c]['day']

            if month == 12:
                year = year + 1

            line_info = dict(
                campaign=c,
                time=campaign_info[c]['time'],
                month=month,
                year=util.year_frac(year, month, day),
                season=get_season(month),
                flux_window_date_bound=flux_window_date_bound,
                doy_start=doy_start,
                doy_end=doy,
                doy_mid=doy_mid,
                obs_grad=obs_gradient_mean,
                obs_grad_err=obs_gradient_std,
                corrected_grad=corrected_gradient_mean,
                corrected_grad_err=corrected_gradient_std,
                flux=flux,
                flux_error=flux_error,
                flux_land=flux_land,
                flux_land_error=flux_land_std,
                flux_uncorrected=flux_nocorr,
                flux_error_uncorrected=flux_error_nocorr,
            )
            line_info.update(ext_gradient_estimates)
            lines.append(line_info)

        return pd.DataFrame(lines).set_index('campaign')

    @property
    def harmonic_fit(self):
        """Return a two-harmonic fit to the campaign fluxes."""
        if self._p_pcov is None:
            self._p_pcov = self._compute_harmonic_fit()
        return self._p_pcov

    @property
    def harmonic_fit_vg_obs(self):
        if self._p_pcov_vg_obs is None:
            self._p_pcov_vg_obs = self._compute_harmonic_fit_vg_obs()
        return self._p_pcov_vg_obs

    @property
    def harmonic_fit_vg_ext(self):
        if self._p_pcov_vg_ext is None:
            self._p_pcov_vg_ext = self._compute_harmonic_fit_vg_ext()
        return self._p_pcov_vg_ext

    def _compute_harmonic_fit_vg_obs(self):
        df = self.gradient_summary.sort_values(by='month')
        x, y = util.antyear_daily(df.doy, df.vg_obs)
        _, yerr = util.antyear_daily(df.doy, df.vg_obs_std)
        return self._fit_harmonic(x, y, yerr)

    def _compute_harmonic_fit_vg_ext(self):
        df = self.gradient_summary.sort_values(by='month')
        x, y = util.antyear_daily(df.doy, df.vg_ext)
        _, yerr = util.antyear_daily(df.doy, df.vg_ext_std)
        return self._fit_harmonic(x, y, yerr)

    def _compute_harmonic_fit(self):
        df = self.campaign_flux.sort_values(by='month')
        x, y = util.antyear_daily(df.doy_mid, df.flux)
        _, yerr = util.antyear_daily(df.doy_mid, df.flux_error)
        return self._fit_harmonic(x, y, yerr)

    def _fit_harmonic(self, x, y, yerr):
        x = np.array(x)/365.
        y = np.array(y)
        yerr = np.array(yerr)
        k = np.isnan(x) | np.isnan(y) | np.isnan(yerr)

        try:
            return curve_fit(harmonic,
                             xdata=x[~k],
                             ydata=y[~k],
                             sigma=yerr[~k],
                             absolute_sigma=True,
                            )
        except Exception as error:
            print('unable to fit harmonic:')
            print(error)
            return np.ones((5,))*np.nan, np.ones((5, 5)) * np.nan

    @property
    def estimate_ann_mean_flux(self):
        """
        Estimate the annual mean flux by fitting a harmonic function to campaign flux estimates.
        Return the mean and an estimate of the uncertainty in the mean.
        """
        p, pcov = self.harmonic_fit
        perr = np.sqrt(np.diag(pcov))
        return p[0], perr[0]

    def seasonal_flux_fit(self, n=365):
        abcd, pcov = self.harmonic_fit
        x = np.linspace(-30, 365+30, n)
        return x, harmonic(x/365.25, *abcd)

    def seasonal_fit_vg_obs(self, n=365):
        abcd, pcov = self.harmonic_fit_vg_obs
        x = np.linspace(-30, 365+30, n)
        return x, harmonic(x/365.25, *abcd)

    def seasonal_fit_vg_ext(self, n=365):
        abcd, pcov = self.harmonic_fit_vg_ext
        x = np.linspace(-30, 365+30, n)
        return x, harmonic(x/365.25, *abcd)
    

class whole_enchilada(object):
    """Encapsulate entire worflow for computing `aircraft_constraint`.
    
    This object loads the observations and simulated observations. 
    It provides a method `get_ac` to compute the `aircraft_constraint` on those data.
    
    Parameters
    ----------
    
    model_tracer_list: list of tuples
      The models and their tracers to use, i.e. [(CT2017, "CO2_OCN"), (CT2019B, "CO2_OCN"), ...]
       
    model_tracer_ext_list : list of tuples
      A list of tuples specifying the (model, tracer) pairs to use to estimate "external" 
      contributions to the observed gradient. I.e., these data are used to correct the 
      observed estimates of the gradient for land and fossil contributions.
      
    model_list_sfco2_lnd : list of tuples
      A list of (model, tracer) pairs used to correct the resulting flux estimate for in-region 
      land and fossil fuel fluxes. This is only used if the contraint is based on total CO2.
      
    profiles_only : boolean
      Use only data collected while "profiling."
      
    clobber : boolean
      If true, recompute rather than reading pre-computed result from cache.
    """
    def __init__(self,
                 model_tracer_list,
                 model_list_sfco2_lnd=[],
                 model_tracer_ext_list=[],
                 profiles_only=True,
                 clobber=False,
                ):
        """Initialize the `whole_enchilada`; load the input data, applying filters."""
        # set incoming parameters
        self.model_tracer_list = model_tracer_list
        self.model_list_sfco2_lnd = model_list_sfco2_lnd
        self.model_tracer_ext_list = model_tracer_ext_list
        self.profiles_only = profiles_only

        assert not (model_tracer_ext_list and model_list_sfco2_lnd)

        # read the data
        self.dfs_obs, self.dfs_model = load_data(
            model_tracer_list=model_tracer_list + model_tracer_ext_list,
            profiles_only=profiles_only,
            clobber=clobber,
        )

        # unique token for this object
        token = dask.base.tokenize(
            model_tracer_list, model_list_sfco2_lnd, model_tracer_ext_list, profiles_only,
        )
        self.cache_dir = f'{cache_path_pickles}/{token}'
        os.makedirs(self.cache_dir, exist_ok=True)

    def __repr__(self):
        input_vars = [
            'model_tracer_list',
            'model_list_sfco2_lnd',
            'model_tracer_ext_list',
            'profiles_only',
        ]
        return pprint.pformat(
            {k: vars(self)[k] for k in input_vars},
            indent=2,
        )

    def _obj_cache_file(self, *args):
        token = dask.base.tokenize(*args)
        return f'{self.cache_dir}/{token}.pkl'

    def get_ac(self,
               ubin,
               lbin,
               gradient_lat_range,
               flux_lat_range,
               fit_groups,
               flux_memory=90,
               udθ=10.,
               ldθ=None,
               lbin_as_upper_bound=True,
               ubin_as_lower_bound=False,
               methane_theta_lbound=305.,
               use_methane_gradient_correction=False,
               model_groups={},
               clobber=False
              ):
        """Compute the aircraft constraint: return an instance of `aircraft_constraint`.
           See the documentation for `aircraft_constraint`.
        """

        cache_file = self._obj_cache_file(
            ubin,
            lbin,
            gradient_lat_range,
            flux_lat_range,
            fit_groups,
            flux_memory,
            udθ,
            ldθ,
            lbin_as_upper_bound,
            ubin_as_lower_bound,
            methane_theta_lbound,
            use_methane_gradient_correction,
            model_groups,
        )

        if os.path.exists(cache_file) and clobber:
            os.remove(cache_file)

        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as fid:
                AC = pickle.load(fid)
        else:
            print('computing...')
            AC = aircraft_constraint(
                ubin=ubin,
                lbin=lbin,
                lbin_as_upper_bound=lbin_as_upper_bound,
                ubin_as_lower_bound=ubin_as_lower_bound,
                gradient_lat_range=gradient_lat_range,
                flux_memory=flux_memory,
                flux_lat_range=flux_lat_range,
                udθ=udθ,
                ldθ=ldθ,
                fit_groups=fit_groups,
                methane_theta_lbound=methane_theta_lbound,
                use_methane_gradient_correction=use_methane_gradient_correction,
                model_groups=model_groups,
                model_tracer_list=self.model_tracer_list,
                model_list_sfco2_lnd=self.model_list_sfco2_lnd,
                model_tracer_ext_list=self.model_tracer_ext_list,
                dfs_obs=self.dfs_obs,
                dfs_model=self.dfs_model,
                campaign_time_point='end',
                bin_aggregation_method='median',
                restrict_groups=True,
            )
            # force computation of fit
            abcd, pcov = AC.harmonic_fit

            print(f'done.\nwriting {cache_file}')
            with open(cache_file, 'wb') as fid:
                pickle.dump(AC, fid)

        return AC
    
    def sensitivity(self, test_name, test_values, clobber=False, use_dask=False, **kwargs):
        parameters = dict(**get_parameters())
        parameters.update(kwargs)
        parameters.pop(test_name)
        if use_dask:
            delayed_objs = [
                dask.delayed(self.get_ac)(**{test_name: value}, **parameters, clobber=clobber) 
                for value in test_values
            ]            
            return dask.compute(*delayed_objs)
        else:
            return [self.get_ac(**{test_name: value}, **parameters, clobber=clobber) for value in test_values]

    
class whole_enchilada_srf(object):
    def __init__(self,
                 model_tracer_list,
                 model_list_sfco2_lnd=[],
                 model_tracer_ext_list=[],
                ):
        """Encapsulate entire worflow for computing `surface_constraint`.

        This object loads the observations and simulated observations. 
        It provides a method `get_sc` to compute the `surface_constraint` on those data.

        Parameters
        ----------

        model_tracer_list: list of tuples
          The models and their tracers to use, i.e. [(CT2017, "CO2_OCN"), (CT2019B, "CO2_OCN"), ...]

        model_list_sfco2_lnd : list of tuples
          A list of (model, tracer) pairs used to correct the resulting flux estimate for in-region 
          land and fossil fuel fluxes. This is only used if the contraint is based on total CO2.
          
        model_tracer_ext_list : list of tuples
          A list of tuples specifying the (model, tracer) pairs to use to estimate "external" 
          contributions to the observed gradient. I.e., these data are used to correct the 
          observed estimates of the gradient for land and fossil contributions.
        """        
        # set incoming parameters
        self.model_tracer_list = model_tracer_list
        self.model_list_sfco2_lnd = model_list_sfco2_lnd
        self.model_tracer_ext_list = model_tracer_ext_list

        assert not (model_tracer_ext_list and model_list_sfco2_lnd)

        # read the data
        self.das_srf = load_data_surface(
            model_tracer_list=self.model_tracer_list + self.model_tracer_ext_list,
        )   

        # unique token for this object
        token = dask.base.tokenize(
            self.model_tracer_list, self.model_list_sfco2_lnd, self.model_tracer_ext_list,
        )
        self.cache_dir = f'{cache_path_pickles}/{token}'
        os.makedirs(self.cache_dir, exist_ok=True)

    def __repr__(self):
        input_vars = [
            'model_tracer_list',
            'model_list_sfco2_lnd',
            'model_tracer_ext_list',
        ]
        return pprint.pformat(
            {k: vars(self)[k] for k in input_vars},
            indent=2,
        )

    def _obj_cache_file(self, *args):
        token = dask.base.tokenize(*args)
        return f'{self.cache_dir}/{token}.pkl'

    def get_sc(self,
               flux_lat_range=(-90., -45.), 
               weight_by_sd_iav=True,
               seasons=['DJF', 'JJA'],
               fit_period='1999-2020',
               periods=[
                   (2000, 2004),
                   (2005, 2009),
                   (2010, 2014),
                   (2015, 2019),
               ] + [(1999, 2020), (1999, 2009), (2009, 2020)],               
               clobber=False
              ):
        """Compute the emergent flux constraint based on surface observations: return 
        an instance of `surface_constraint`.
        See the documentation for `surface_constraint`.
        """

        cache_file = self._obj_cache_file(
            flux_lat_range,
            weight_by_sd_iav,
            seasons,
            fit_period,
            periods,
        )

        if os.path.exists(cache_file) and clobber:
            os.remove(cache_file)

        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as fid:
                SC = pickle.load(fid)
        else:
            print('computing...')
            SC = surface_constraint(
                periods=periods,
                flux_lat_range=flux_lat_range,
                weight_by_sd_iav=weight_by_sd_iav,
                seasons=seasons,
                fit_period=fit_period,
                das_srf=self.das_srf,
                model_tracer_list=self.model_tracer_list,
                model_list_sfco2_lnd=self.model_list_sfco2_lnd,
                model_tracer_ext_list=self.model_tracer_ext_list,
            )
            # force computation 
            df = SC.surface_flux
            
            print(f'done.\nwriting {cache_file}')
            with open(cache_file, 'wb') as fid:
                pickle.dump(SC, fid)

        return SC    

    def sensitivity(self, test_name, test_values, clobber=False, use_dask=False, **kwargs):
        parameters = dict(
               flux_lat_range=(-90., -45.), 
               weight_by_sd_iav=True,
               seasons=['DJF', 'JJA'],
               fit_period='1999-2020',
               periods=[
                   (2000, 2004),
                   (2005, 2009),
                   (2010, 2014),
                   (2015, 2019),
               ] + [(1999, 2020), (1999, 2009), (2009, 2020)],                
        )
        parameters.update(kwargs)
        parameters.pop(test_name)
        if use_dask:
            delayed_objs = [
                dask.delayed(self.get_sc)(**{test_name: value}, **parameters, clobber=clobber) 
                for value in test_values
            ]
            return dask.compute(*delayed_objs)
            
        else:       
            return [self.get_sc(**{test_name: value}, **parameters, clobber=clobber) for value in test_values]

    
    
def harmonic(t, mu, a1, phi1, a2, phi2):
    """A harmonic"""
    return (mu + a1 * np.cos(1. * 2. * np.pi * t + phi1) +
            a2 * np.cos(2. * 2. * np.pi * t + phi2))


def get_season(month):
    if month in [12, 1, 2]:
        return 'DJF'
    elif month in [3, 4, 5]:
        return 'MAM'
    elif month in [6, 7, 8]:
        return 'JJA'
    elif month in [9, 10, 11]:
        return 'SON'
    else:
        raise ValueError(f'bad month value {month}')


class devnull(object):
    """write into the void"""
    def write(self, x):
        pass


@contextlib.contextmanager
def nostdout():
    """context manager to suppress stdout"""
    save_stdout = sys.stdout
    sys.stdout = devnull()
    yield
    sys.stdout = save_stdout


def get_dset_fluxes(model_tracer_list, flux_lat_range):
    """return datasets of monthly """
    #with nostdout():

    dsets_fluxes = {}
    for model, tracer in model_tracer_list:
        # get obj
        model_objs = models.Model(model)
        # get flux
        key = f'{model}-{tracer}'
        ds = (model_objs
              .open_derived_dataset(
                  'flux_ts_monthly',
                  lat_range=flux_lat_range
              ).sel(time=slice('1998', '2020'))
              .compute()
             )
        if '+' not in tracer and tracer != 'CO2_SUM':
            ds = ds.rename({f'SF{tracer}': 'FLUX'})
        elif tracer == 'CO2_SUM':
            ds = ds.rename({'SFCO2': 'FLUX'})        
        else:
            sf_list = [f'SF{t}' for t in tracer.split('+')]
            ds['FLUX'] = ds[sf_list[0]]
            for sf in sf_list[1:]:
                ds['FLUX'] += ds[sf]

        dsets_fluxes[key] = ds[['FLUX']]

    return dsets_fluxes


def get_dset_fluxes_mmm(dsets_fluxes, seasons=['DJF', 'JJA']):

    dsets_fluxes_mmm = {k: {} for k in seasons}

    for model, dset in dsets_fluxes.items():
        for k in dsets_fluxes_mmm.keys():
            dsets_fluxes_mmm[k][model] = util.ann_mean(
                dsets_fluxes[model], season=k, time_bnds_varname=None
            )
            dsets_fluxes_mmm[k][model]['time'] = (
                dsets_fluxes_mmm[k][model].time + util.season_yearfrac[k]
            )
    return dsets_fluxes_mmm


def get_dset_fluxes_land(flux_lat_range, model_list, season_avg=False):
    """return a dataset of monthly fluxes"""

    with nostdout():
        ds_list = []
        for model in model_list:

            # get obj
            model_objs = models.Model(model)

            # get flux
            ds_list.append(model_objs
                           .open_derived_dataset(
                               'flux_ts_monthly',
                               lat_range=flux_lat_range)[['SFCO2_LND']]
                           .sel(time=slice('1998', '2020'))
                           .compute()
                          )

    coord_model = xr.DataArray(
        model_list, dims=('model'), coords={'model': model_list}, name='model',
    )
    dso = xr.concat(ds_list, dim=coord_model, join='outer').rename({'SFCO2_LND': 'FLUX'})

    if not season_avg:
        return dso
    else:
        dso_mmm = {k: {} for k in ['DJF', 'JJA']}
        for k in dso_mmm.keys():
            dso_mmm[k] = util.ann_mean(dso, season=k, time_bnds_varname=None)
            dso_mmm[k]['time'] = (dso_mmm[k].time + util.season_yearfrac[k])
        return dso_mmm


def plot_constraint(ax, df_data, fit_dict={},
                    title='', title_left='', title_right='',
                    ylabel='', xlabel='',
                    highlight_campaign=None,
                    plotted_elements=[],
                    include_equation=True,
                    xhat=None, circle_one=[],
                    only_models=[],
                   ):
    """make a scatter plot of constraint with fit line"""

    txt_box_props = dict(facecolor='w', alpha=0.75, edgecolor='None', boxstyle='square,pad=0')
    marker_spec = figure_panels.marker_spec_models()
    legend_elements = []
    eq_text = None


    i_model = df_data.index.names.index('model')
    i_field = df_data.index.names.index('field')
    if 'campaign' in df_data.index.names:
        i_campaign = df_data.index.names.index('campaign')

    X = []; Y = []
    for ndx in df_data.index:
        model = ndx[i_model]
        field = ndx[i_field]
        df = df_data.loc[ndx]

        if only_models:
            if model not in only_models:
                continue

        x, y = df.gradient, df.flux
        if np.isnan(x) | np.isnan(y): continue
        X.append(x); Y.append(y)

        if 'gradient_std' in df and 'flux_std' in df:
            xerr, yerr = df.gradient_std, df.flux_std
            p = ax.errorbar(x, y, xerr=xerr, yerr=yerr, **marker_spec[model][field])
        else:
            if highlight_campaign is None:
                if model in ["CT2017", "CTE2108"]:
                    p = ax.plot(x, y, linestyle='None', **marker_spec[model][field], zorder=100)
                else:
                    p = ax.plot(x, y, linestyle='None', **marker_spec[model][field])
                if (model, field) not in plotted_elements:
                    legend_elements.append(p[0])
                    plotted_elements.append((model, field))
            else:
                campaign = ndx[i_campaign]
                marker_spec_here = {k: v for k, v in marker_spec[model][field].items()}
                if campaign not in highlight_campaign:
                    marker_spec_here['alpha'] = 0.5
                    marker_spec_here['markersize'] = 4
                    marker_spec_here['color'] = 'gray'
                    marker_spec_here['markerfacecolor'] = 'gray'
                    marker_spec_here['zorder'] = -100                    
                p = ax.plot(x, y, linestyle='None', **marker_spec_here)

                if campaign in highlight_campaign:
                    if (model, field) not in plotted_elements:
                        legend_elements.append(p[0])
                        plotted_elements.append((model, field))
        if model in circle_one:
            ax.plot(x, y, 'ro', markersize=12, markerfacecolor='none')

    ax.axhline(0, linewidth=0.5, linestyle='--', color='k', zorder=-10)
    ax.axvline(0, linewidth=0.5, linestyle='--', color='k', zorder=-10)

    if fit_dict and fit_dict['pval'] <= 0.05:
        if xhat is None:
            xhat = np.array([np.nanmin(X), np.nanmax(X)])
        else:
            xhat = np.array(xhat)

        ax.plot(xhat, fit_dict['beta'][0] * xhat + fit_dict['beta'][1],
                '-',
                color='k',
                linewidth=1,
               )

        if include_equation:
            xoff = np.diff(ax.get_xlim()) * 0.025
            yoff = -np.diff(ax.get_ylim()) * 0.3

            slope, stderr_slope = fit_dict['beta'][0], fit_dict['stderr_beta'][0]
            r2, rmse = fit_dict['r2'], fit_dict['rmse']
            str_text = (
                f'{slope:0.1f}±{stderr_slope:0.1f} Pg C yr$^{{-1}}$:ppm\n' +
                f'RMSE={rmse:0.3f} Pg C yr$^{{-1}}$\n' +
                f'r$^2$={r2:0.3f}'
            )

            eq_text = ax.text(
                ax.get_xlim()[0]+xoff, ax.get_ylim()[1]+yoff,
                str_text,
                fontsize=12, fontweight='bold',
                bbox=txt_box_props,
            )

    if title:
        ax.set_title(title)

    if title_left:
        ax.set_title(title_left, loc='left')

    if title_right:
        ax.set_title(title_right, loc='right')

    if ylabel:
        ax.set_ylabel(ylabel)

    if xlabel:
        ax.set_xlabel(xlabel)

    return legend_elements, plotted_elements, eq_text


def add_obs_constraint(ax, df, indexes, 
                       labels=[], 
                       label_dx={}, 
                       label_dy={},
                       rotation={},
                       marker_spec={}):
    """add constraint lines"""
    ylm = ax.get_ylim()
    xlm = ax.get_xlim()
    dx = -np.diff(xlm) * 0.015
    dy = np.diff(ylm) * 0.03


    if 'gradient_mean' in df:
        obs_grad = 'gradient_mean'
        obs_grad_err = 'gradient_std' if 'gradient_std' in df else 'corrected_grad_err'
    elif 'corrected_grad' in df:
        obs_grad = 'corrected_grad'
        obs_grad_err = 'corrected_grad_err'
    elif 'obs_grad' in df:
        obs_grad = 'obs_grad'
        obs_grad_err = 'obs_grad_err'
    else:
        raise ValueError('cannot determine gradient field')

    for i, ndx in enumerate(indexes):
        row = df.loc[ndx]

        if marker_spec:
            color = marker_spec[ndx]['color']
        else:
            color = figure_panels.palette_colors[0]

        ax.axvspan(
             row[obs_grad] - row[obs_grad_err],
             row[obs_grad] + row[obs_grad_err],
             color=color, alpha=0.1
         )
        ax.axvline(row[obs_grad], color=color, linewidth=2.)
        #ax.axvline(row[obs_grad] - row[obs_grad_err],  color=color, linewidth=0.5, linestyle='--')
        #ax.axvline(row[obs_grad] + row[obs_grad_err],  color=color, linewidth=0.5, linestyle='--')
        if labels:
            use_dx, use_dy = np.float(dx), np.float(dy)
            use_rotation = 90.
            if labels[i] in label_dx:
                use_dx = label_dx[labels[i]]
            if labels[i] in label_dy:
                use_dy = label_dy[labels[i]]
            if labels[i] in rotation:
                use_rotation=rotation[labels[i]]
            ax.text(row[obs_grad]+use_dx, ylm[0]+use_dy,
                    labels[i],
                    rotation=use_rotation, ha='center', color=color, fontweight='bold')


def flux_contraint_seasonal_cycle(ax, ac, dsets_sfco2_mon=None, 
                                  obs_color='k', 
                                  include_fit=True,
                                  label_campaigns=True):

    if dsets_sfco2_mon is not None:
        marker_spec = figure_panels.marker_spec_models()
        sfco2_ocn_model_list = list(dsets_sfco2_mon.keys())

        for model in sfco2_ocn_model_list:
            x = util.doy_midmonth() #dsets_fluxes_mon[model].month - 0.5
            y = dsets_sfco2_mon[model].SFCO2_OCN
            x, y = util.antyear_daily(x, y)

            field = 'CO2' if 'CO2' in marker_spec[model] else 'CO2_OCN'
            marker_spec[model][field]['label'] = flux_label(marker_spec[model][field]['label'])
            x = np.concatenate(([-30.], x, [395]))
            y = np.concatenate(([y[-1]], y, [y[0]]))

            if model == 'SOM-FFN':
                ax.plot(x, y, linestyle='-', lw=2,
                  zorder=50,
                  **marker_spec[model][field])
            elif 'TM5-Flux' in model:
                ax.plot(x, y, linestyle='-', lw=2,
                  zorder=45,                
                  **marker_spec[model][field])                        
            else:
                ax.plot(x, y, linestyle='-', lw=1, **marker_spec[model][field], alpha=1)

    df = ac.campaign_flux.copy().reset_index().sort_values(by='month')

    x, y = util.antyear_daily(df.doy_mid, df.flux)
    _, yerr = util.antyear_daily(df.doy_mid, df.flux_error)
        
    _, clist = util.antyear_daily(df.doy_mid, df.campaign)

    h = ax.errorbar(x, y, yerr=yerr,
                    color=obs_color,
                    marker='.',
                    linestyle='none',
                    markersize=14,
                    label='Aircraft CO$_2$ observations',
                    capsize=0,
                    lw=3,
                    zorder=100)

    if label_campaigns:
        xoff = {
            'HIPPO-1': -2, #-5,
            'HIPPO-3': 0, #-5, #-6,
            'ORCAS-J': 2, #5,
            'ORCAS-F': 0, #+10,
            'ATom-2': 0, #+10
        }

        ylm = np.array(ax.get_ylim())
        offset = np.diff(ylm) * 0.18
        ylm[0] = ylm[0] - offset
        ax.set_ylim((ylm))

        for xi, c in zip(x, clist):
            dx = 0
            if c in xoff:
                dx = xoff[c]
            ax.text(xi+dx, ylm[0]+np.diff(ylm)*0.025, c,
                    rotation=90, ha='center', fontsize=8)

    if include_fit:
        xfit, yfit = ac.seasonal_flux_fit()
        ax.plot(xfit, yfit, '-', color=obs_color, lw=2, zorder=100)

    ax.set_xlim((-5, 370))
    ax.set_xticks(figure_panels.bomday)
    ax.set_xticklabels([f'        {m}' for m in figure_panels.monlabs_ant]+[''])

    ax.axhline(0, lw=1., c='k', zorder=-100)
    ax.set_ylabel('Air-sea flux [Pg C yr$^{-1}$]');

    return h


def flux_contraint_djf_timeseries(ax, dsets_sfco2_mmm, ac, df_surface_flux, sfco2_ocn_model_list=[]):

    marker_spec = figure_panels.marker_spec_models()

    if not sfco2_ocn_model_list:
        sfco2_ocn_model_list = list(dsets_sfco2_mmm['DJF'].keys())
        if 'TM5-Flux-mrf' in sfco2_ocn_model_list and 'SOM-FFN' in sfco2_ocn_model_list:
            sfco2_ocn_model_list.remove('TM5-Flux-mrf')

    for model in sfco2_ocn_model_list:
        x = dsets_sfco2_mmm['DJF'][model].time
        y = dsets_sfco2_mmm['DJF'][model].SFCO2_OCN

        field = 'CO2' if 'CO2' in marker_spec[model] else 'CO2_OCN'
        marker_spec[model][field]['label'] = flux_label(marker_spec[model][field]['label'])
        ax.plot(x, y, linestyle='-', #label=label,
                lw=1, **marker_spec[model][field],
               )


    df = ac.campaign_flux
    df = df.loc[df.season == 'DJF']

    y = df.flux.values
    yerr = df.flux_error.values
    x = df.year.values
    clist = df.index.values
    ax.errorbar(x, y, yerr=yerr,
                color='k', marker='.', markersize=14,
                label='Aircraft CO$_2$ observations',
                linestyle='none',
                lw=2, zorder=100)

    labeled = False
    for row in df_surface_flux.iterrows():
        s = row[1]
        rect = plt.Rectangle((s.time_bnds[0], s.flux-s.flux_error), np.diff(s.time_bnds), s.flux_error*2,
                             facecolor="black", alpha=0.1, zorder=-100)
        ax.add_patch(rect)
        if not labeled:
            ax.plot(s.time_bnds, [s.flux, s.flux], 'k--', label='Surface CO$_2$ observations')
            labeled = True
        else:
            ax.plot(s.time_bnds, [s.flux, s.flux], 'k--')

    ylm = ax.get_ylim()
    xoff = {
        'HIPPO-1': -0.5,
        'ATom-2': 0.6,
        'ORCAS-J': -0.5,
        'ORCAS-F': 0.6,
    }
    for xi, c in zip(x, clist):
        ax.text(xi+xoff[c], ylm[0]+np.diff(ylm)*0.05, c, rotation=90, ha='center')

    ax.axhline(0, lw=1., c='k', zorder=-100)
    ax.set_ylabel('Air-sea flux [Pg C yr$^{-1}$]');

    ax.set_xticks(np.arange(1998, 2022, 2));
    ax.set_xlim([1997, 2021]);


def flux_contraint_jja_timeseries(ax, dsets_sfco2_mmm, ac):

    marker_spec = figure_panels.marker_spec_models()

    sfco2_ocn_model_list = list(dsets_sfco2_mmm['JJA'].keys())
    if 'TM5-Flux-mrf' in sfco2_ocn_model_list and 'SOM-FFN' in sfco2_ocn_model_list:
        sfco2_ocn_model_list.remove('TM5-Flux-mrf')

    for model in sfco2_ocn_model_list:
        x = dsets_sfco2_mmm['JJA'][model].time
        y = dsets_sfco2_mmm['JJA'][model].SFCO2_OCN

        field = 'CO2' if 'CO2' in marker_spec[model] else 'CO2_OCN'
        marker_spec[model][field]['label'] = flux_label(marker_spec[model][field]['label'])
        ax.plot(x, y, linestyle='-', #label=label,
                lw=1, **marker_spec[model][field],
               )

    df = ac.campaign_flux
    df = df.loc[df.season == 'JJA']

    y = df.flux.values
    yerr = df.flux_error.values
    x = df.year
    clist = df.index.values

    ax.errorbar(x, y, yerr=yerr,
                color='k', marker='.', markersize=14,
                label='Aircraft CO$_2$ observations',
                linestyle='none',
                lw=2,
                zorder=100)


    xoff = {
        'HIPPO-5': -0.4,
        'ATom-1': -0.4,
    }
    ylm = ax.get_ylim()
    for xi, c in zip(x, clist):
        ax.text(xi+xoff[c], ylm[0]+np.diff(ylm)*0.05, c, rotation=90, ha='center')

    ax.axhline(0, lw=1., c='k', zorder=-100)
    ax.set_ylabel('Air-sea flux [Pg C yr$^{-1}$]');

    ax.set_xticks(np.arange(1998, 2022, 2));
    ax.set_xlim([1997, 2021]);


def flux_label(label, linebreaks=False):
    if linebreaks:
        replace_dict = {
            'TM5+SOCAT': 'SOCAT',
            'TM5+SOCAT+SOCCOM': 'SOCAT+\nSOCCOM',
            'TM5+SOCCOM(only)': 'SOCCOM\n(only)',
            'TM5+SOCCOM(only, –4µatm)': 'SOCCOM\n(only,–4µatm)',
        }
    else:
        replace_dict = {
            'TM5+SOCAT': 'SOCAT',
            'TM5+SOCAT+SOCCOM': 'SOCAT+SOCCOM',
            'TM5+SOCCOM(only)': 'SOCCOM(only)',
            'TM5+SOCCOM(only, –4µatm)': 'SOCCOM(only, –4µatm)',
        }

    if label in replace_dict:
        return label.replace(label, replace_dict[label])
    else:
        return label


def goodness_of_fit(ac):
    """return aggregated metric describing quality of fit"""
    sse = {}
    for fit_group, df in ac.df_model_fits.items():
        sse[fit_group] = (df.fit_sse_y / df.fit_n).sum()
    return sse



def sensitivity_suite(obj, test_name, test_values, clobber=False, **kwargs):
    """plot a variety of metrics describing aircraft constraint"""
    parameters = dict(
        ubin=300.,
        lbin=280.,
        udθ=10.,
        lbin_as_upper_bound=True,
        ubin_as_lower_bound=False,
        gradient_lat_range=(-90., -45.),
        flux_memory=90,
        flux_lat_range=(-90., -45.),
        fit_groups=('DJF', 'MAMJJASON'),
        clobber=clobber,
    )

    parameters.update(kwargs)
    parameters.pop(test_name)

    objs = []
    for value in test_values:
        objs.append(
            obj.get_ac(**{test_name: value}, **parameters)
        )

    ac_objs = {k: v for k, v in zip(test_values, objs)}

    fit_groups = parameters['fit_groups'] if 'fit_groups' in parameters else []

    pp = pprint.PrettyPrinter(indent=4)
    pp.pprint(parameters)
    fig, axs = util.canvas(2, 2)

    xbar = []
    ybar = []
    yerr = []
    ec_slope = {k: [] for k in fit_groups}
    ec_slope_err = {k: [] for k in fit_groups}
    for value, ac in ac_objs.items():
        # plot gradients
        xco2, yco2 = ac.seasonal_fit_vg_obs()
        axs[0, 0].plot(xco2, yco2, '-', label=value)

        # get seasonal cycle of flux and annual mean
        xfit, yfit = ac.seasonal_flux_fit()
        axs[1, 0].plot(xfit, yfit, '-', label=value)
        try:
            value / 2
            xbar.append(value)
        except:
            xbar.append(str(value))
        ybar.append(ac.estimate_ann_mean_flux[0])
        yerr.append(ac.estimate_ann_mean_flux[1])

        # plot slope and error for each fit group
        xco2, yco2 = ac.seasonal_fit_vg_ext()
        axs[0, 1].plot(xco2, yco2, '-', label=value)

    axs[0, 0].set_title('Fit to observed vertical gradient')
    axs[0, 1].set_title('Fit to simulated vertical LND+FFF gradient')

    axs[1, 0].set_title('Air-sea flux')
    axs[1, 0].set_ylabel('Air-sea flux [Pg C yr$^{-1}$]');
    axs[1, 0].legend(ncol=2);

    for ax in [axs[0, 0], axs[0, 1]]:
        ax.set_ylabel('$\Delta_{ θ}$CO$_2$ [ppm]')
        ax.axhline(0., lw=0.5, c='k')
        ax.legend(ncol=2);
        ax.set_xlim((-10, 375))
        ax.set_xticks(figure_panels.bomday)
        ax.set_xticklabels([f'        {m}' for m in figure_panels.monlabs_ant]+['']);


    axs[1, 1].bar(xbar, ybar, yerr=yerr)
    axs[1, 1].set_xticks(xbar)
    axs[1, 1].set_xticklabels(xbar)
    axs[1, 1].set_title('Annual mean flux')

    plt.suptitle(f'Sensitivity of flux to "{test_name}"');


def assemble_plot_constraint(ac, model_group=None, highlight_campaign=[], suppress_eqn=False):

    fit_groups = ac.fit_groups
    i_campaign = ac.df_data.xs((fit_groups[0]), level=('campaign_group')).index.names.index('campaign')

    fig, axs = util.canvas(len(fit_groups))
    nrow, ncol = axs.shape

    theta_str = figure_panels.theta_bin_def(ac.theta_bins)

    kwargs_dict = {}

    for n, fit_group in enumerate(fit_groups):
        kwargs_dict[fit_group] = dict(
            title=f'{fit_group}: 90-day flux vs. {theta_str} ∆CO$_2$',
            #highlight_campaign=['ORCAS-F'],
            df_data=ac.df_data.xs(fit_group, level='campaign_group'),
        )
        if not suppress_eqn:
            if model_group is not None:
                kwargs_dict[fit_group]['fit_dict'] = get_fit_dict(ac.df_model_fits[fit_group].loc[model_group])
            else:
                kwargs_dict[fit_group]['fit_dict'] = get_fit_dict(ac.df_fits.loc[fit_group])

        if highlight_campaign:
            kwargs_dict[fit_group]['highlight_campaign'] = [highlight_campaign[n]]

        if (n+1)%nrow == 0:
            kwargs_dict[fit_group]['xlabel'] = '$\Delta_{ θ}$CO$_2$ [ppm]'

    xlm = {}; ylm = {};  eq_text = {}
    ax_alias = {}
    plotted_elements = []; legend_elements = [];
    for n, k in enumerate(fit_groups):
        kwargs = {key: val for key, val in kwargs_dict[k].items()}
        i, j = np.unravel_index(n, (nrow, ncol))
        ax_alias[k] = axs[i, j]
        if j == 0:
            kwargs['ylabel'] = f'Surface flux south of {ac.flux_lat_range} [Pg C yr$^{{-1}}$]'

        le, pe, eq_text[k] = plot_constraint(axs[i, j], plotted_elements=plotted_elements, **kwargs)
        legend_elements.extend(le); plotted_elements.extend(pe)

    axs[-1, -1].legend(handles=legend_elements, ncol=4, loc=(-1.275, -0.75), frameon=False)

    if model_group is not None:
        plt.suptitle(model_group, fontweight='bold')
    
    if 'DJF' in fit_groups:
        add_obs_constraint(
            ax=ax_alias['DJF'],
            df=ac.campaign_flux,
            indexes=['ORCAS-F',],
            marker_spec=figure_panels.marker_spec_campaigns(),
            labels=['ORCAS-F',],
            label_dx={'ORCAS-F': -0.1},
        )

    if 'MAMJJASON' in fit_groups:
        add_obs_constraint(
            ax=ax_alias['MAMJJASON'],
            df=ac.campaign_flux,
            indexes=['ATom-1',],
            marker_spec=figure_panels.marker_spec_campaigns(),
            labels=['ATom-1',],
            label_dx={'ATom-1': 0.05},
        )

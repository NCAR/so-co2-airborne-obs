import os
import shutil

import yaml

import xarray as xr
import numpy as np
import xpersist as xp

from . import calc
from . detrend import detrend
from . generic_assets import list_assets

from . import model_cams
from . import model_carbontracker
from . import model_carboscope
from . import model_cte
from . import model_miroc
from . import model_tm5flux
from . import dataset_som_ffn

from .config import path_to_here, project_tmpdir, cache_rootdir_local

cache_rootdir_bigfiles = project_tmpdir


model_aliases = dict(
    s99oc45S_v2020='s99oc_v2020', # this is a run that Christian did with an
                                  # atm ocean tracer only responsive to region south of 45S
)


class Model(object):
    """
    High level object encapsulating model resources used in this project
    """
    dataset_methods = []

    def __init__(self, name):
        """initialize object"""
        
        if name in model_aliases:
            self.alias = name
            self.name = model_aliases[name]
        else:
            self.alias = None            
            self.name = name

        assert self.name in self.known_models, (
            f'unknown model: {self.name}'
        )
        self.use_prior = False
        if '-prior' in self.name:
            self.use_prior = True
            print(f'{self.name}: USING PRIORS')
        
    @property
    def known_models(self):
        return [
            'CAMSv18', 'CAMSv19', 'CAMSv20r1',
            's99oc_v4.3', 's99oc_ADJocI40S_v4.3', 
            'sEXTocNEET_SOCCOM_v2020', 'sEXTocNEET_v2020', 
            's99oc_SOCCOM_v2020', 's99oc_v2020', 's99oc_ADJocI40S_v2020',
            'CESM', 
            'CT2017', 'CT2019B', 
            'CTE2018', 'CTE2020',
            'MIROC', 'MIROC-prior', 'MIROC2019',
            'TM5-Flux-m0f', 'TM5-Flux-m0p', 'TM5-Flux-mmf', 'TM5-Flux-mmp',
            'TM5-Flux-mrf', 'TM5-Flux-mrp', 'TM5-Flux-mwf', 'TM5-Flux-mwp',
            'SOM-FFN',
        ]
        
    @property
    def _model_mod(self):

        if self.name in ['CAMSv18', 'CAMSv19', 'CAMSv20r1',]:
            mod = model_cams
        elif self.name in ['CarboScope', 'CarboScopeAdjSO', 
                           'sEXTocNEET_SOCCOM_v2020', 'sEXTocNEET_v2020', 
                           's99oc_SOCCOM_v2020', 's99oc_v2020', 's99oc_ADJocI40S_v2020']:
            mod = model_carboscope
        elif self.name in ['CESM']:
            mod = model_cesm
        elif self.name in ['CT2017', 'CT2019B']:
            mod = model_carbontracker
        elif self.name in ['CTE2018', 'CTE2020']:
            mod = model_cte
        elif self.name in ['MIROC', 'MIROC2019', 'MIROC-prior',]:
            mod = model_miroc
        elif self.name in ['TM5-Flux-m0f', 'TM5-Flux-m0p', 'TM5-Flux-mmf', 'TM5-Flux-mmp',
                           'TM5-Flux-mrf', 'TM5-Flux-mrp', 'TM5-Flux-mwf', 'TM5-Flux-mwp',]:
            mod = model_tm5flux            
        elif self.name in ['SOM-FFN']:
            mod = dataset_som_ffn
        else:
            raise ValueError(f'could not identify model module: {self.name}')
            
        mod.this_model = self.name            
        return mod
    
    @property
    def _dataset_methods(self):
        return dict(
            flux_ts_monthly=dict(
                function=self._compute_flux_ts_monthly,
                cache_rootdir=cache_rootdir_local,
                cache_format='nc',
            ),
            fluxes_za_ts_monthly=dict(
                function=self._compute_fluxes_za_ts_monthly,
                cache_rootdir=cache_rootdir_local,
                cache_format='nc',
            ),
            spo_ts_daily=dict(
                function=self._compute_spo_ts_daily,
                cache_rootdir=cache_rootdir_local,
                cache_format='nc',
            ),
            spo_ts_daily_detrend=dict(
                function=self._compute_spo_ts_daily_detrend,
                cache_rootdir=cache_rootdir_local,
                cache_format='zarr',
            ),
            spo_ts_daily_stl_trend=dict(
                function=self._compute_spo_ts_daily_stl_trend,
                cache_rootdir=cache_rootdir_local,
                cache_format='zarr',
            ),            
            spo_ts_monthly=dict(
                function=self._compute_spo_ts_monthly,
                cache_rootdir=cache_rootdir_local,
                cache_format='zarr',
            ),
            molefractions_surface_daily=dict(
                function=self._compute_molefractions_surface_daily,
                cache_rootdir=cache_rootdir_bigfiles,
                cache_format='zarr',
            ),
            molefractions_surface_daily_detrend=dict(
                function=self._compute_molefractions_surface_daily_detrend,
                cache_rootdir=cache_rootdir_bigfiles,
                cache_format='zarr',
            ),                  
            molefractions_surface_monthly=dict(
                function=self._compute_molefractions_surface_monthly,
                cache_rootdir=cache_rootdir_bigfiles,
                cache_format='zarr',
            ),            
            molefractions_z=dict(
                function=self._compute_molefractions_z,
                cache_rootdir=cache_rootdir_bigfiles,
                cache_format='zarr',
            ),
            molefractions_z_za=dict(
                function=self._compute_molefractions_z_za,
                cache_rootdir=cache_rootdir_bigfiles,
                cache_format='zarr',
            ),                 
            molefractions_z_za_detrend=dict(
                function=self._compute_molefractions_z_za_detrend,
                cache_rootdir=cache_rootdir_bigfiles,
                cache_format='zarr',
            ),            
            molefractions_theta=dict(
                function=self._compute_molefractions_theta,
                cache_rootdir=cache_rootdir_bigfiles,
                cache_format='zarr',
            ),     
            molefractions_theta_za=dict(
                function=self._compute_molefractions_theta_za,
                cache_rootdir=cache_rootdir_bigfiles,
                cache_format='zarr',
            ),
            molefractions_theta_za_detrend=dict(
                function=self._compute_molefractions_theta_za_detrend,
                cache_rootdir=cache_rootdir_bigfiles,
                cache_format='zarr',
            ),                   
            molefractions_theta_bins=dict(
                function=self._compute_molefractions_theta_bins,
                cache_rootdir=cache_rootdir_local,
                cache_format='nc',
            ),    
            molefractions_theta_bins_sectors=dict(
                function=self._compute_molefractions_theta_bins_sectors,
                cache_rootdir=cache_rootdir_local,
                cache_format='nc',
            ),              
        )
    
    def assets(self, product):
        return list_assets(self.name, product)
    
    @property
    def dataset_methods(self):
        return list(self._dataset_methods.keys())
    
    def open_dataset(self, product, **kwargs):
        """open a dataset"""
        if self.use_prior:
            return self._model_mod.open_dataset(product, use_prior=True, **kwargs)
        else:
            return self._model_mod.open_dataset(product, **kwargs)
            
    def open_derived_dataset(self, dataset_name, kwargs_name=None, persist=True,
                             clobber=False, return_none=False, **kwargs):
        """return a computed dataset"""
    
        assert dataset_name in self._dataset_methods, (
            f'unrecognized dataset: {dataset_name}' 
        )
        compute_func = self._dataset_methods[dataset_name]['function']
        
        if persist:
            # cache file
            cache_path = f'{self._dataset_methods[dataset_name]["cache_rootdir"]}/{self.name}'
            cache_format = self._dataset_methods[dataset_name]["cache_format"]
            
            if dataset_name == 'flux_ts_monthly':
                lat_range = [np.float(f) for f in kwargs['lat_range']]
                kwargs_name = f'lat_range={lat_range[0]:0.1f}_to_{lat_range[1]:0.1f}'.replace('.', '_')
            
            elif kwargs_name is None:
                kwargs_name = '.'.join([
                    f'{k}={v}'.replace('.', '_')
                    for k, v in kwargs.items()]
                )
            
            
            if kwargs_name:
                cache_name = '.'.join([dataset_name, kwargs_name])
            else:                
                cache_name = dataset_name
            cache_file = f'{cache_path}/{cache_name}.{cache_format}'                

            # I think there is a bug in xpersist            
            if clobber and os.path.exists(cache_file):
                if cache_format == 'zarr':
                    shutil.rmtree(cache_file)
                else:
                    os.remove(cache_file)
            
            if return_none and os.path.exists(cache_file):
                return
                
            # generate xpersist partial            
            persist_ds = xp.persist_ds(
                name=cache_name, 
                path=cache_path, 
                trust_cache=True, 
                format=cache_format, 
            )

            return persist_ds(compute_func)(**kwargs)

        else:
            return compute_func(**kwargs)
    
    def _compute_molefractions_surface_monthly(self):
        """get surface mole fractions at monthly frequency"""
        return self.open_dataset(
            product='molefractions_surface',
            freq='monthly',
        )

    def _compute_molefractions_surface_daily(self):
        """get surface mole fractions at daily frequency"""
        return self.open_dataset(
            product='molefractions_surface',
            freq='daily',
        )
    
    def _compute_molefractions_surface_daily_detrend(self):
        """get surface mole fractions at daily frequency, detrended"""
        return detrend(
            self.open_derived_dataset(
                'molefractions_surface_daily',
            ),
            map_blocks=True,
        )
    
    def _compute_spo_ts_daily(self):
        """get the spo timeseries"""

        ds = self.open_dataset(
            product='molefractions_surface',
            freq='daily',
        )
        
        lat_vars = [v for v in ds.variables if 'lat' in ds[v].dims]
        other_vars = set(ds.variables) - set(lat_vars)        
        
        with xr.set_options(keep_attrs=True):            
            dsa = ds.sel(
                lat=-90, 
                method='nearest', 
                drop=True,
            ).mean('lon').drop(['area']).compute()

        # copy other vars
        for v in other_vars:
            dsa[v] = ds[v]

        return dsa
    
    def _compute_spo_ts_daily_detrend(self):
        """get the spo timeseries, detrended"""        
        return detrend(
            self._compute_spo_ts_daily(),
        )
        
    def _compute_spo_ts_daily_stl_trend(self):
        return calc.get_stl_trend_ds(
            self.open_derived_dataset('spo_ts_daily'),
            freq='daily',
        )
        
    def _compute_spo_ts_monthly(self):
        """get the spo timeseries"""        

        ds = self.open_dataset(
            product='molefractions_surface',
            freq='monthly',
        )
        
        lat_vars = [v for v in ds.variables if 'lat' in ds[v].dims]
        other_vars = set(ds.variables) - set(lat_vars)        
        
        with xr.set_options(keep_attrs=True):            
            dsa = ds.sel(
                lat=-90, 
                method='nearest', 
                drop=True,
            ).mean('lon').drop(['area']).compute()

        # copy other vars
        for v in other_vars:
            dsa[v] = ds[v]

        return dsa
    
    def _compute_flux_ts_monthly(self, lat_range):
        """Compute the regional mean flux south of `lat_crit`"""
        
        ds = self.open_dataset('fluxes', freq='monthly')
        
        if self.name in ['SOM-FFN']:
            data_vars = ['SFCO2_OCN',]
        else:
            data_vars = ['SFCO2', 'SFCO2_OCN', 'SFCO2_LND', 'SFCO2_FFF']        
        
        # validate that that the dataset is how we like it
        check_vars_coords(
            ds, 
            expected_vars=data_vars, 
            expected_coords={'time', 'lat', 'lon'}
        )        
        calc.ensure_monthly(ds)
        
        # compute area normalization
        lat_crit = (lat_range[0] <= ds.lat) & (ds.lat <= lat_range[1])
        masked_area = ds.area.where(lat_crit).fillna(0.)
        np.testing.assert_approx_equal(
            (masked_area / masked_area.sum(dim=('lat' ,'lon'))).sum(dim=xr.ALL_DIMS), 1.,
            significant=6,
        )
        ds_so = (ds[data_vars] * masked_area).sum(dim=('lat' ,'lon'))
            
        # convert units
        for v in data_vars:
            attrs = ds[v].attrs
            units = ds[v].attrs['units']
            assert units == calc.str_molCm2yr, (
                f'unexpected units\n'
                f'expected: {calc.str_molCm2yr}\n'
                f'received: {units}'
            )
            ds_so[v] = ds_so[v] * calc.molCyr_to_PgCyr
            attrs['units'] = calc.str_PgCyr
            ds_so[v].attrs = attrs
        
        return ds_so
    
    def _compute_fluxes_za_ts_monthly(self):
        """Compute the zonal mean flux"""
        
        ds = self.open_dataset('fluxes', freq='monthly')
        
        if self.name in ['SOM-FFN']:
            data_vars = ['SFCO2_OCN',]
        else:
            data_vars = ['SFCO2', 'SFCO2_OCN', 'SFCO2_LND', 'SFCO2_FFF']
        
        # validate that that the dataset is how we like it
        check_vars_coords(
            ds, 
            expected_vars=data_vars, 
            expected_coords={'time', 'lat', 'lon'}
        )        
        calc.ensure_monthly(ds)
        
        # compute area normalization
        dlat = np.abs(np.diff(ds.lat))
        np.testing.assert_almost_equal(dlat, dlat[0], decimal=4)        
        #masked_area = ds.area.mean(dim='lon')
        ds_za = (ds[data_vars] * ds.area).sum(dim=('lon'))
        
        # convert units
        for v in data_vars:
            attrs = ds[v].attrs
            units = ds[v].attrs['units']
            assert units == calc.str_molCm2yr, (
                f'unexpected units\n'
                f'expected: {calc.str_molCm2yr}\n'
                f'received: {units}'
            )
            ds_za[v] = ds_za[v] * calc.molCyr_to_PgCyr / dlat[0]
            attrs['units'] = calc.str_PgCyr+' degree$^{-1}$'
            ds_za[v].attrs = attrs

        return ds_za    
    
    def _compute_molefractions_z(self):
        """Compute the molefraction fields in z coords"""        
        return self.open_dataset('molefractions_z')

    def _compute_molefractions_surface(self):
        """Compute the molefraction fields"""        
        return self.open_dataset('molefractions_surface')

    def _compute_molefractions_theta(self):
        """Compute the molefraction fields in theta coords"""        
        return self.open_dataset('molefractions_theta')
    
    def _compute_molefractions_theta_za(self):
        """Compute the molefraction fields in theta coords"""        
        return self.open_derived_dataset('molefractions_theta').mean('lon').compute()
    
    def _compute_molefractions_theta_za_detrend(self):
        """Compute the detrended zonal mean molefraction in theta coords"""        
        with xr.set_options(keep_attrs=True):
            ds = self.open_derived_dataset('molefractions_theta_za').compute()
            
        return detrend(ds)
    
    def _compute_molefractions_z_za(self):
        """Compute the zonal mean molefraction in z coords"""        
        with xr.set_options(keep_attrs=True):
            return self.open_derived_dataset('molefractions_z').mean('lon').compute()
    
    def _compute_molefractions_z_za_detrend(self):
        """Compute the detrended zonal mean molefraction in z coords"""        
        with xr.set_options(keep_attrs=True):
            ds = self.open_derived_dataset('molefractions_z').mean('lon').compute()
            
        return detrend(ds)

    def _compute_molefractions_theta_bins(self, lat_bounds, theta_bins, lon_bounds=None):
        """Compute theta-binned dataset"""
        ds = self.open_dataset('molefractions')
        
        # mask domain
        lon = ds.lon.compute()
        if any(lon < 0):
            lon.values[lon < 0] = lon.values[lon < 0] + 360.
            
        if lon_bounds is None:
            ds_r = ds.where((lat_bounds[0] <= ds.lat) & (ds.lat <= lat_bounds[-1]))
        else:
            ds_r = ds.where(
                (lat_bounds[0] <= ds.lat) & (ds.lat <= lat_bounds[-1]) &
                (lon_bounds[0] <= lon) & (lon <= lon_bounds[-1])
            )

        # compute theta-bins
        ds_list = []
        for tlo, thi in theta_bins:
            theta_mask = (tlo <= ds_r.theta) & (ds_r.theta < thi)
            ds_list.append(ds_r.where(theta_mask).mean(['lat', 'lon', 'lev']))
        
        # concatentate along bin-coordinate
        coord_theta_bins = xr.DataArray(
            [np.mean(tlo_thi) for tlo_thi in theta_bins], 
            dims=('theta_bins'), name='theta_bins'
        )
        coord_theta_bins_bounds = xr.DataArray(
            theta_bins,
            dims=('theta_bins', 'd2'), 
            coords=dict(theta_bins=coord_theta_bins),
            name='theta_bins_bounds'
        )

        dso = xr.concat(ds_list, dim=coord_theta_bins)
        dso['theta_bins_bounds'] = coord_theta_bins_bounds
        dso = dso.set_coords('theta_bins_bounds')
        
        # transpose dims
        for v in dso.data_vars:
            if 'time' in dso[v].dims:
                newdims = ['time']  + [d for d in dso[v].dims if d != 'time']
                dso[v] = dso[v].transpose(*newdims)

        return dso
            
    def _compute_molefractions_theta_bins_sectors(self, lat_bounds, theta_bins):
        """Compute theta-binned dataset"""
        x = np.linspace(0., 360., 7)
        sector_edges = [(x[i], x[i+1]) for i in range(len(x) - 1)]

        coord_lon_bins = xr.DataArray(
            [np.mean(xlo_xhi) for xlo_xhi in sector_edges], 
            dims=('lon_bins'), name='lon_bins'
        )
        coord_lon_bins_bounds = xr.DataArray(
            sector_edges,
            dims=('lon_bins', 'd2'), 
            coords=dict(theta_bins=coord_lon_bins),
            name='lon_bins_bounds'
        )
        
        ds_list_x = []
        for xlo_xhi in sector_edges:            
            ds_list_x.append(
                self._compute_molefractions_theta_bins(lat_bounds, theta_bins, lon_bounds=xlo_xhi).compute()
            )
        
        dso = xr.concat(ds_list_x, dim=coord_lon_bins)
        dso['coord_lon_bins_bounds'] = coord_lon_bins_bounds        

        # transpose dims
        for v in dso.data_vars:
            if 'time' in dso[v].dims:
                newdims = ['time']  + [d for d in dso[v].dims if d != 'time']
                dso[v] = dso[v].transpose(*newdims)
                
        return dso
    
def check_vars_coords(ds, expected_vars, expected_coords):
    missing_vars = set(expected_vars) - set(ds.variables)
    assert not missing_vars, (
        f'missing data vars\n'
        f'expected: {expected_vars}\n'
        f'received: {ds.variables}'
    )

    missing_coords = set(expected_coords) - set(ds.coords)
    assert not missing_coords, (
        f'missing coords\n'
        f'expected: {expected_coords}\n'
        f'received: {ds.variables}'
    )    
    
    
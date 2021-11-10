import os
import shutil
import yaml

import warnings

from functools import partial
from collections import OrderedDict
import pickle

import numpy as np
import pandas as pd

import cftime
import xarray as xr

import util

path_to_here = os.path.dirname(os.path.realpath(__file__))

split_orcas = True

if split_orcas:
    campaign_list = [
        'HIPPO-1',
        'HIPPO-2',
        'HIPPO-3',
        'HIPPO-5',
        'ORCAS-J',
        'ORCAS-F',
        'ATom-1',
        'ATom-2',
        'ATom-3',
        'ATom-4',
    ]
else:
    campaign_list = [
        'HIPPO-1',
        'HIPPO-2',
        'HIPPO-3',
        'HIPPO-5',
        'ORCAS',
        'ATom-1',
        'ATom-2',
        'ATom-3',
        'ATom-4',
    ]

instruments = dict(
    ORCAS=['AO2', 'QCLS', 'Medusa',],
    HIPPO=['AO2', 'OMS', 'Medusa', 'PFP'],
    ATom=['AO2', 'QCLS', 'Medusa', 'PFP'],
)

co2vars = {c: [f'co2{i.lower()}' for i in instr]
               for c, instr in instruments.items()}

co2mvars = {c: [f'co2m{i.lower()}' for i in instr]
               for c, instr in instruments.items()}

def expand_project_vars(project_vars):
    campaign_vars = {c: project_vars[c.split('-')[0]] for c in campaign_list}
    campaign_vars['ORCAS'] = campaign_vars['ORCAS-J']
    return campaign_vars

co2vars = expand_project_vars(co2vars)
co2mvars = expand_project_vars(co2mvars)

"""
best (only in situ) CH4 from HIPPO is CH4_QCLS
(PFP, UCATS, and PANTHER CH4 also available at lower time resolution)

best CH4 from ORCAS is CH4_NOAA
(CH4_QCLS not as good, but available)

best CH4 from ATom is CH4_NOAA
(CH4_QCLS not as good, but available, PFP, UCATS, and PANTHER as well)
"""
ch4vars_primary = dict(
    HIPPO='ch4qcls',
    ORCAS='ch4noaa',
    ATom='ch4noaa',
)

ch4vars = dict(
    HIPPO=['ch4pfp'], # 'ch4ucats', 'ch4panther',
    ORCAS=['ch4qcls'],
    ATom=['ch4qcls', 'ch4pfp'],  # 'ch4ucats', 'ch4panther',
)
ch4mvars = dict(
    HIPPO=['ch4mpfp'], # 'ch4mucats', 'ch4mpanther',
    ORCAS=['ch4mqcls'],
    ATom=['ch4mqcls', 'ch4mpfp'],  # , 'ch4mucats', 'ch4mpanther'
)

ch4vars = expand_project_vars(ch4vars)
ch4mvars = expand_project_vars(ch4mvars)

"""
best SF6 from HIPPO not as clear - see comparison at
https://www.nature.com/articles/nature13721/figures/10 (SM Fig. 6)
"ECD" = PANTHER, "NWAS" = PFP legend in panel m
also, no PANTHER or UCATS for HIPPO3
also, you can see from that figure that the analytical noise is not insignificant wrt within-hemisphere gradients
PANTHER and UCATS have a lot more data than PFP because they are in situ GCs that sample ~ every 2 min, vs 24 flasks per flight for PFPs, even though Prabir's figures don't show that (I think he must have binned the GC measurements, see links below for typical coverage)
I will provide all 3, variable names sf6pfp, sf6panther, sf6ucats, but I suggest you try using PFP for consistency - if that is not enough coverage, you can check the others

for ATom I am also providing both UCATS and PANTHER in the 10 sec merge file, PFP SF6 is not in the 10-sec merge, but only in the PFP merge, so I will put it there (ATOM_SO_mSPO_pfp.txt)
I have not looked closely at which is better, but PANTHER is much more precise at least on A1 SO:
https://archive.eol.ucar.edu/homes/stephens/ATOM1/RLS1809windiv/atom1_20160812_vlat_prof_SF6_PECD.png
https://archive.eol.ucar.edu/homes/stephens/ATOM1/RLS1809windiv/atom1_20160812_vlat_prof_SF6_UCATS.png
https://archive.eol.ucar.edu/homes/stephens/ATOM1/RLS1809windiv/PFP/atom1_20160812_vlat_prof_sf6_CCGG_PFP.png
(ATom PFPs were measured twice for SF6, once by CCGG and once by HATS, with better precision and likely better scale agreement with UCATS and PANTHER for HATS, so I will provide those - let me know if you want CCGG too)
https://archive.eol.ucar.edu/homes/stephens/ATOM1/RLS1809windiv/PFP/atom1_20160812_vlat_prof_sf6_HATS_PFP.png
https://archive.eol.ucar.edu/homes/stephens/ATOM1/RLS1809windiv/PFP/atom1_20160812_vlat_prof_sf6_HATS_PFP.png

Note for both HIPPO and ATom, SF6 PANTHER and UCATS also have error variables for every measurement. Let me know if you want to look at these.
"""
sf6vars_primary = dict(
    HIPPO='sf6panther',
    ORCAS=None,
    ATom='sf6panther',
)
sf6vars = dict(
    HIPPO=['sf6ucats', 'sf6pfp'],
    ORCAS=[],
    ATom=['sf6ucats', 'sf6pfp'],
)
sf6mvars = dict(
    HIPPO=['sf6mucats'],
    ORCAS=[],
    ATom=['sf6mucats'],
)

sf6vars = expand_project_vars(sf6vars)
sf6mvars = expand_project_vars(sf6mvars)

constituent_vars = dict(
    co2=co2vars,
    ch4=ch4vars,
    sf6=sf6vars,
)
constituent_mvars = dict(
    co2=co2mvars,
    ch4=ch4mvars,
    sf6=sf6mvars,
)


def assign_project(dfp, project):
    dfp['project'] = project
    if project == 'ORCAS':
        dfp['camp'] = 1


def fill_nans_on_missing_columns(dfs, columns):
    for df in dfs:
        for c in columns:
            if c not in df:
                df[c] = np.nan


def join_col(xlist):
    xout = []
    for xi in xlist:
        if isinstance(xi, int):
            xout.append(f'{xi:03d}')
        elif isinstance(xi, str):
            xout.append(xi)
        elif isinstance(xi, float) and ~np.isnan(xi) and np.int(xi) - xi == 0.:
            xout.append(f'{xi:03.0f}')
        elif np.isnan(xi):
            return np.nan
        else:
            print(xlist)
            raise ValueError(f'unknown type: {xi}')
    return '-'.join(xout)


def regroup(pcm):
    project, camp, month = pcm[0], pcm[1], pcm[2]

    if np.isnan(camp):
        camp = 'NA'
    else:
        camp = int(camp)

    if project in ['ORCAS']:
        assert month in [1, 2, 3], f'unexpected month: {month}'
        if project == 'ORCAS' and split_orcas:
            letter = {1: 'J', 2: 'F', 3: 'M'}
            return f'ORCAS-{letter[month]}'
        else:
            return project
    else:
        return f'{project}-{camp}'


def set_campaign_flight_profile_date(df):
    """ unique flight and profile id and add a date"""
    df['campaign_id'] = df[['project', 'camp', 'month']].apply(regroup, axis=1)
    df['flight_id'] = df[['project', 'camp', 'flt']].apply(join_col, axis=1)
    df['profile_id'] = df[['project', 'camp', 'flt', 'prof']].apply(join_col, axis=1)
    df['date'] = pd.to_datetime(df[['year', 'month', 'day']])


def open_aircraft_data(model='obs', tracer=None):

    if model in ['obs', 'medusa', 'pfp']:
        return _open_aircraft_data_obs(model)
    else:
        return _open_aircraft_data_model(model, tracer)


def _open_aircraft_data_model(model, tracer):

    files = dict(
        HIPPO='HIPPO_SO_mSPO.txt',
        ATom='ATOM_SO_mSPO.txt',
        ORCAS='ORCAS_SO_mSPO.txt',
    )

    with open(f'{path_to_here}/data/model-description.yaml', 'r') as fid:
        model_info = yaml.safe_load(fid)

    if tracer in model_info[model]['obs_data_paths']:
        obs_data_paths = {tracer: model_info[model]['obs_data_paths'][tracer]}
    elif tracer in ['CO2', 'CO2_SUM', 'CO2_LND+CO2_FFF']:
        obs_data_paths = model_info[model]['obs_data_paths']
    else:
        raise ValueError(f'unknown tracer requested: {tracer}')

    tracer_dfs = {}
    for tracer_i, path in obs_data_paths.items():
        dfs = []
        columns = set()
        for project, file in files.items():
            file_in = f'{path}/{file}'
            if not os.path.exists(file_in):
                print(f'{model}: {file_in} does not exist')
                continue
            dfp = pd.read_csv(file_in, delim_whitespace=True)
            assign_project(dfp, project)
            columns = columns.union(dfp.columns)

            # subsample as TM5 runs did not include all of ATOM data
            if 'TM5-Flux' in model and project == 'ATom' and tracer_i in ['CO2_LND', 'CO2_FFF', 'CO2_BKG']:
                dfp = dfp.loc[dfp.year != 2018]

            dfs.append(dfp)

        fill_nans_on_missing_columns(dfs, columns)
        df = pd.concat(dfs)
        set_campaign_flight_profile_date(df)

        tracer_dfs[tracer_i] = df.reset_index(drop=True)

    # add to generate CO2_SUM
    if tracer in ['CO2', 'CO2_SUM'] and tracer not in tracer_dfs.keys():
        assert all(v in tracer_dfs for v in ['CO2_OCN', 'CO2_LND', 'CO2_FFF', 'CO2_BKG'])
        df_sum = tracer_dfs['CO2_OCN'].copy()
        for v in ['CO2_LND', 'CO2_FFF', 'CO2_BKG']:
            # check to make sure these datasets really match
            sel_not_nan = ~np.isnan(tracer_dfs[v].prof)
            assert len(df_sum) == len(tracer_dfs[v])
            assert (tracer_dfs[v].prof.values[sel_not_nan] == df_sum.prof.values[sel_not_nan]).all()
            assert (tracer_dfs[v].hour.values[sel_not_nan] == df_sum.hour.values[sel_not_nan]).all()
            for f in ['theta', 'pressure']:
                np.testing.assert_allclose(tracer_dfs[v][f], df_sum[f], equal_nan=True,)

            df_sum['co2'] += tracer_dfs[v].co2
            df_sum['co2mspo'] += tracer_dfs[v].co2mspo
        tracer_dfs[tracer] = df_sum

    elif '+' in tracer:
        tracers = tracer.split('+')
        df_sum = tracer_dfs[tracers[0]]
        for subt in tracers[1:]:
            df_sum['co2'] += tracer_dfs[subt].co2
            df_sum['co2mspo'] += tracer_dfs[subt].co2mspo
        tracer_dfs[tracer] = df_sum

    return tracer_dfs[tracer]


def _open_aircraft_data_obs(model):

    if model == 'medusa':
        files = dict(
            HIPPO='HIPPO_SO_mSPO_medusa.txt',
            ATom='ATOM_SO_mSPO_medusa.txt',
            ORCAS='ORCAS_SO_mSPO_medusa.txt',
        )
    elif model == 'pfp':
        files = dict(
            HIPPO='HIPPO_SO_mSPO_pfp.txt',
            ATom='ATOM_SO_mSPO_pfp.txt',
        )

    else:
        files = dict(
            HIPPO='HIPPO_SO_mSPO.txt',
            ATom='ATOM_SO_mSPO.txt',
            ORCAS='ORCAS_SO_mSPO.txt',
        )

    get_file = lambda m, f: f'{path_to_here}/data/aircraft-obs/{f}'

    print(f'loading {model}')
    dfs = []
    columns = set()
    for project, file in files.items():
        file_in = get_file(model, file)

        if not os.path.exists(file_in):
            print(f'{model}: {file_in} does not exist')
            continue

        dfp = pd.read_csv(file_in, delim_whitespace=True)

        if model == 'obs':
            project_1 = 'ORCAS-J' if project == 'ORCAS' else f'{project}-1'
            if model not in ['medusa', 'pfp']:
                # add
                if project in ch4vars_primary and ch4vars_primary[project] in dfp:
                    dfp['ch4'] = dfp[ch4vars_primary[project]].copy()
                    # make mvars
                    for v in ch4vars[project_1]:
                        assert v != ch4vars_primary[project], 'primary var found in secondary list'
                        if v in dfp:
                            field = f'ch4m{v[3:]}'
                            if field not in dfp:
                                dfp[field] = dfp.ch4 - dfp[v]

                if project in sf6vars_primary and sf6vars_primary[project] in dfp:
                    dfp['sf6'] = dfp[sf6vars_primary[project]].copy()
                    # make mvars
                    for v in sf6vars[project_1]:
                        assert v != sf6vars_primary[project], 'primary var found in secondary list'
                        if v != sf6vars_primary[project] and v in dfp:
                            field = f'sf6m{v[3:]}'
                            if field not in dfp:
                                dfp[field] = dfp.sf6 - dfp[v]

        assign_project(dfp, project)
        columns = columns.union(dfp.columns)
        dfs.append(dfp)

    fill_nans_on_missing_columns(dfs, columns)

    df = pd.concat(dfs)#, ignore_index=True)

    # correct for type errors
    if model in ['medusa', 'pfp']:
        df = df.loc[~np.isnan(df.camp)]
        df['camp'] = [int(c) for c in df.camp.values]
        df['flt'] = [int(c) for c in df.flt.values]
        
        df.prof.values[np.isnan(df.prof.values)] = -1
        df['prof'] = [int(c) for c in df.prof.values]

    if "mon" in df.columns and "month" not in df.columns:
        df = df.rename(columns=dict(mon="month"))
        
    set_campaign_flight_profile_date(df)

    # retain 1 co2-minus field and reverse sign; drop other CO2-fields
    if model in ['medusa', 'pfp']:
        constituents = ['co2', 'ch4'] if model == 'pfp' else ['co2']
        these_fields = []
        for constituent in constituents:
            rev_dict = {
                f'{constituent}mnoaa': (
                    df.campaign_id.str.contains('ORCAS') | df.campaign_id.str.contains('ATom')
                ),
                f'{constituent}mqcls': (
                    df.campaign_id.str.contains('HIPPO') #& ~df.campaign_id.str.contains('HIPPO-1')
                ),
#                f'{constituent}mao2': (
#                    df.campaign_id.str.contains('HIPPO-1')
#                ),                
            }
            rev_field_name = f'{constituent}m{model}'
            this_field = f'{constituent}{model}'
            these_fields.extend([rev_field_name, this_field])

            # reverse the sign
            rev_field = np.nan * np.ones(len(df))
            for field, sel_loc in rev_dict.items():
                if field not in df:
                    print(f'Not reversing {field}')
                    continue                      
                rev_field[sel_loc] = -1.0 * df[field].loc[sel_loc]

            df[rev_field_name] = rev_field
            if constituent == 'co2':
                df[this_field] = df.co2
            assert this_field in df

        keep_fields = [
            v for v in df.columns if not any(c in v for c in constituents)
        ] + these_fields
        df = df[keep_fields]

#     else:
#         """
#         QCLS and OMS data were rejected for the entire 2nd half of HIPPO-1 
#         (starting with leaving CHC north) because of really large alt dependent biases. 
#         Something happened in Christchurch. We should reject them here and use AO2 as primary 
#         CO2 instrument.
#         """
#         df['co2'] = np.where(df.campaign_id == 'HIPPO-1', df.co2ao2, df.co2)
#         df['co2ao2'] = np.where(df.campaign_id == 'HIPPO-1', np.nan, df.co2ao2)
#         df['co2mao2'] = np.where(df.campaign_id == 'HIPPO-1', np.nan, df.co2mao2)
#         df['co2moms'] = np.where(df.campaign_id == 'HIPPO-1', df.co2 - df.co2oms, df.co2moms)
        
    return df.reset_index(drop=True)


def df_midtrop(df):
    mid_trop = (-90. <= df.lat) & (df.lat <= -45.)
    # mid_trop = mid_trop & (4000. <= df.alt) & (df.alt <= 6000.)
    mid_trop = mid_trop & (295. <= df.theta) & (df.theta <= 305.)
    return df.loc[mid_trop]


def get_campaign_info(clobber=False, verbose=True, lump_orcas=False):

    if lump_orcas:
        cache_file = f'{path_to_here}/data/cache/aircraft-campaign-info-lump-orcas.pkl'
    else:
        cache_file = f'{path_to_here}/data/cache/aircraft-campaign-info.pkl'

    if os.path.exists(cache_file) and clobber:
        os.remove(cache_file)

    if os.path.exists(cache_file):
        with open(cache_file, 'rb') as fid:
            campaign_info = pickle.load(fid)
    else:

        campaign_info = {
            model: group_flights(open_aircraft_data(model), lump_orcas=lump_orcas)
            for model in ['pfp', 'medusa', 'obs',]
        }

        co2_mt_data = {c: {} for c in campaign_info['obs'].keys()}
        for model in ['pfp', 'medusa', 'obs',]:
            for c, info in campaign_info[model].items():
                if 'co2_midtrop_each_sensor' in info:
                    co2_mt_data[c].update(info['co2_midtrop_each_sensor'])

        campaign_info = campaign_info['obs']
        for c in campaign_info.keys():
            campaign_info[c]['co2_midtrop_each_sensor'] = co2_mt_data[c]
            data = np.array([v for v in co2_mt_data[c].values()])
            campaign_info[c]['co2_midtrop_multi_sensor'] = np.nanmean(data)
            campaign_info[c]['co2_midtrop_multi_sensor_std'] = np.nanstd(data)

        with open(cache_file, 'wb') as fid:
            pickle.dump(campaign_info, fid)

    if verbose:
        for campaign_id, info in campaign_info.items():
            print(f'{campaign_id}:')
            for k, v in info.items():
                print(f'\t{k}: {v}')

    return campaign_info


def group_flights(df, verbose=False, lump_orcas=False):
    """Return a dictionary with sensible groupings of flights and some assoc attributes"""

    if lump_orcas:
        lumper = lambda c: 'ORCAS' if 'ORCAS' in c else c
        campaign_list_loc = util.list_set([lumper(c) for c in campaign_list])
    else:
        campaign_list_loc = campaign_list

    obs = {k: {} for k in campaign_list_loc}

    for campaign_id in campaign_list_loc:

        if lump_orcas and campaign_id == 'ORCAS':
            group_id = 'ORCAS'
            sel_c = (df.campaign_id == 'ORCAS-J') | (df.campaign_id == 'ORCAS-F')
            sel_loc = (df.lat <= -30) & sel_c
        else:
            group_id = campaign_id
            sel_loc = (df.campaign_id == campaign_id) & (df.lat <= -30)

        dfi = df.loc[sel_loc]

        if len(dfi) > 0:
            obs[group_id]['flights'] = sorted(list(dfi.flight_id.unique()))
            obs[group_id]['time'] = np.datetime64(dfi.date.mean(), 'D')
            obs[group_id]['year'] = obs[group_id]['time'].astype('datetime64[Y]').astype(int) + 1970
            obs[group_id]['month'] = (obs[group_id]['time'].astype('datetime64[M]').astype(int) % 12 +
                                         1).astype(int)
            obs[group_id]['day'] = (obs[group_id]['time'] -
                                       obs[group_id]['time'].astype('datetime64[M]') + 1).astype(int)
            obs[group_id]['time_bound'] = np.array([
                np.datetime64(dfi.date.min(), 'D'), np.datetime64(dfi.date.max(), 'D')])

            # compute some values
            dfi_mt = df_midtrop(dfi)
            for v in ['co2', 'ch4', 'sf6']:
                obs[group_id][f'{v}_midtrop'] = np.nan
                if v in dfi:
                    obs[group_id][f'{v}_midtrop'] = dfi_mt[v].mean()

            obs[group_id]['co2_midtrop_each_sensor'] = {}
            for v in ['co2']+co2vars[group_id]:
                if v in dfi:
                    obs[group_id]['co2_midtrop_each_sensor'][v] = dfi_mt[v].mean()

        else:
            obs[group_id]['flights'] = []
            for v in ['time', 'year', 'month', 'day', 'time_bound', 'co2_midtrop']:
                obs[group_id][v] = np.nan

    if verbose:
        for group_id, info in obs.items():
            print(f'{group_id}:')
            for k, v in info.items():
                print(f'\t{k}: {v}')

    return obs


def groups_get_dataframe(df, group_name, lump_orcas=False):
    info = group_flights(df, lump_orcas=lump_orcas)[group_name]

    if info['flights']:
        return df.loc[df.flight_id.isin(info['flights']) & (df.lat <= -30.)]


def groups_allflights(df):
    flights = []
    for info in group_flights(df).values():
        flights.extend(info['flights'])
    return flights


def groups_find_campaign(df, flight_id):
    for campaign, info in group_flights(df).items():
        if flight_id in info['flights']:
            return campaign


def groups_select_profiles(df, lat_lo, lat_hi, group_name=None, profiles_only=True):
    select_profile = (lat_lo <= df.lat) & (df.lat <= lat_hi)
    if profiles_only:
        select_profile = select_profile & (df.prof > 0)
    if group_name is None:
        target_flights = groups_allflights(df)
    else:
        target_flights = group_flights(df)[group_name]['flights']

    return select_profile & df.flight_id.isin(target_flights)


def make_section(df, campaign, vertical_coord='z', multi_sensor=False, common_reference=False):
    """compute cross-section binning/contouring"""

    assert vertical_coord in ['z', 'theta', 'pressure']

    # set up coordinate system
    dy_bin = 2.5
    ygrid_edges = np.arange(-80., -30.+dy_bin, dy_bin)

    if vertical_coord == 'z':
        dz_bin = 0.5
        zgrid_edges = np.arange(0, 11+dz_bin, dz_bin)

    elif vertical_coord == 'theta':
        dz_bin = 2.5
        zgrid_edges = np.arange(255., 400.+dz_bin, dz_bin)

    elif vertical_coord == 'theta_bins':
        dz_bin = 2.5
        zgrid_edges = np.arange(255., 400.+dz_bin, dz_bin)

    elif vertical_coord == 'pressure':
        dz_bin = 100.
        zgrid_edges = np.arange(100., 1000.+dz_bin, dz_bin)

    ygrid_center = np.vstack((ygrid_edges[:-1], ygrid_edges[1:])).mean(axis=0)
    zgrid_center = np.vstack((zgrid_edges[:-1], zgrid_edges[1:])).mean(axis=0)

    info = get_campaign_info(lump_orcas=True, clobber=False, verbose=False)[campaign]
    df_sub = groups_get_dataframe(df, campaign, lump_orcas=True)

    if df_sub is None:
        print(f'{campaign}: no data')
        return

    # get the obs
    y = df_sub.lat.values
    z = df_sub.alt.values * 1e-3
    theta = df_sub.theta.values
    pres = df_sub.pressure.values
    co2 = df_sub.co2.values
    stratosphere = df_sub.strat.values

    dco2 = co2 - info['co2_midtrop']

    if multi_sensor:
        nvars = len(co2vars[campaign])
        for sensor in ['obs', 'pfp', 'medusa']:
            if sensor == 'obs':
                df_sensor = df_sub
            else:
                df_sensor = groups_get_dataframe(open_aircraft_data(sensor), campaign, lump_orcas=True)
                if df_sensor is None:
                    continue

            for v in co2vars[campaign]:
                if v in df_sensor:
                    y = np.concatenate((y, df_sensor.lat.values))
                    z = np.concatenate((z, df_sensor.alt.values * 1e-3))
                    theta = np.concatenate((theta, df_sensor.theta.values))
                    pres = np.concatenate((pres, df_sensor.pressure.values))
                    co2 = np.concatenate((co2, df_sensor[v].values))
                    if not common_reference:
                        dco2 = np.concatenate(
                            (dco2, df_sensor[v].values - info['co2_midtrop_each_sensor'][v])
                        )

        if common_reference:
            dco2 = co2 - info['co2_midtrop_multi_sensor']

        data_vars = dict(
            CO2=co2,
            DCO2=dco2,
        )
    else:
        # fields to remap
        data_vars = dict(
            CO2=co2,
            DCO2=dco2,
            CH4=df_sub.ch4.values,
            DCH4=df_sub.ch4.values - info['ch4_midtrop'],
            SF6=df_sub.sf6.values,
            DSF6=df_sub.sf6.values - info['sf6_midtrop'],
            STRATOSPHERE=stratosphere,
        )

    if vertical_coord == 'z':
        zdata = z
    elif vertical_coord == 'theta':
        zdata = theta
    elif vertical_coord == 'pressure':
        zdata = pres

    # assemble Dataset
    dset_section = xr.Dataset()
    dims = (vertical_coord, 'y')
    DataArrays = {}
    if vertical_coord in ['z', 'pressure']:
        YI, ZI, THETA_binned, _ = util.griddata(
            y, zdata, theta, ygrid_edges, zgrid_edges, use_rbf=False, smooth=0.2
        )
        YI, ZI, THETA, _ = util.griddata(
            y, zdata, theta, ygrid_edges, zgrid_edges, use_rbf=True, smooth=0.2
        )
        DataArrays['THETA_binned'] = xr.DataArray(THETA_binned, dims=dims, name='THETA_binned')
        DataArrays['THETA'] = xr.DataArray(THETA, dims=dims, name='THETA')

    elif vertical_coord == 'theta':
        YI, ZI, ALT_binned, _ = util.griddata(
            y, zdata, z, ygrid_edges, zgrid_edges, use_rbf=False, smooth=0.2
        )
        YI, ZI, ALT, _ = util.griddata(
            y, zdata, z, ygrid_edges, zgrid_edges, use_rbf=True, smooth=0.2
        )
        DataArrays['ALT_binned'] = xr.DataArray(ALT_binned, dims=dims, name='ALT_binned')
        DataArrays['ALT'] = xr.DataArray(ALT, dims=dims, name='ALT')


    for key, values in data_vars.items():
        _, _, VALUES, N = util.griddata(
            y, zdata, values, ygrid_edges, zgrid_edges, use_rbf=False, smooth=0.2,
        )
        name = f'{key}_binned'
        DataArrays[name] = xr.DataArray(VALUES, dims=dims, name=name)
        name = f'N_{key}'
        DataArrays[name] = xr.DataArray(N, dims=dims, name=name)

    dset_section['month'] = xr.DataArray([info['month']], dims=('time'))
    dset_section['year'] = xr.DataArray([info['year']], dims=('time'))
    dset_section['campaigns'] = xr.DataArray([campaign], dims=('time'))

    time_coord = xr.DataArray([info['time']], dims=('time'), name='time')
    dset_section['time'] = time_coord
    for name, da in DataArrays.items():
        dset_section[name] = DataArrays[name].expand_dims(time=time_coord)

    dset_section['LAT'] = xr.DataArray(YI, dims=dims)
    dset_section['ALT'] = xr.DataArray(ZI, dims=dims)
    dset_section['ye'] = xr.DataArray(ygrid_edges[:-1], dims=('ye'))
    dset_section['ze'] = xr.DataArray(zgrid_edges, dims=('ze'))
    dset_section['y'] = xr.DataArray(ygrid_center, dims=(dims[-1]))
    dset_section[vertical_coord] = xr.DataArray(zgrid_center, dims=(dims[0]))

    return dset_section.set_coords(dims+('time', 'month', 'year', 'campaigns', 'ye', 'ze', 'LAT', 'ALT'))


def make_profile_ds(model, tracer, profile_spec, lat_lo, lat_hi):
    """bin data onto regular altitude levels"""

    fields = ['co2', 'strat']

    # generate vertical coordinate
    vertical_coord = profile_spec
    if profile_spec == 'z':
        dz_bin = 0.25
        zgrid_edges = np.arange(0, 11+dz_bin, dz_bin)
        fields += ['theta']
        zgrid_bounds = np.vstack((zgrid_edges[:-1], zgrid_edges[1:])).T
        zgrid_center = zgrid_bounds.mean(axis=1)

    elif profile_spec == 'theta':
        dz_bin = 5.
        zgrid_edges = np.arange(267.5, 400.+dz_bin, dz_bin)
        zgrid_bounds = np.vstack((zgrid_edges[:-1], zgrid_edges[1:])).T
        zgrid_center = zgrid_bounds.mean(axis=1)
        zgrid_edges[0] = -9999.
        
    elif profile_spec == 'pressure':
        dz_bin = 100.
        zgrid_edges = np.arange(100., 1000.+dz_bin, dz_bin)
        zgrid_bounds = np.vstack((zgrid_edges[:-1], zgrid_edges[1:])).T
        zgrid_center = zgrid_bounds.mean(axis=1)


    # read dataset
    df = open_aircraft_data(model, tracer)

    fields = list(filter(lambda s: s in df.columns, fields))
    if model == 'obs':
        fields += list(filter(lambda s: any(c in s for c in ['co2', 'ch4', 'sf6']), df.columns))
        fields = list(set(fields))

    select_profile = groups_select_profiles(df, lat_lo, lat_hi)

    profiles = df.loc[select_profile].profile_id.unique().astype(str)

    # dimensions dataset
    dset_profile = xr.Dataset()
    dset_profile[vertical_coord] = xr.DataArray(
        zgrid_center,
        dims=(vertical_coord),
        name=vertical_coord,
    )
    dset_profile[vertical_coord+'_bounds'] = xr.DataArray(
        zgrid_bounds,
        dims=(vertical_coord, 'd2'),
        name=vertical_coord+'_bounds',
    )
    dset_profile['profile'] = xr.DataArray(
        profiles, #[i for i in range(len(profiles))],
        dims=('profile'),
        name='profile'
    )
#     dset_profile['profile_name'] = xr.DataArray(
#         profiles,
#         dims=('profile'),
#         name='profile_name'
#     )

    dset_profile['time'] = xr.DataArray(
        np.empty(len(profiles)).astype('datetime64[ns]'),
        dims=('profile'),
        name='time',
    )
    dset_profile['flight_id'] = xr.DataArray(
        np.empty(len(profiles)).astype(str),
        dims=('profile'),
        name='flight_id',
    )
    dset_profile['campaign'] = xr.DataArray(
        np.empty(len(profiles)).astype(str),
        dims=('profile'),
        name='campaign',
    )

    for field in ['doy', 'month', 'year', 'lat', 'lon']:
        dset_profile[field] = xr.DataArray(
            np.empty(len(profiles)),
            dims=('profile'),
            name=field,
        )
    dset_profile = dset_profile.set_coords([v for v in dset_profile.variables])

    # dimension data_var
    for field in fields:
        dset_profile[field] = xr.DataArray(
            np.ones((len(profiles), len(zgrid_center))) * np.nan,
            dims=('profile', vertical_coord),
            name=field,
        )
        dset_profile[f'{field}_std'] = xr.DataArray(
            np.ones((len(profiles), len(zgrid_center))) * np.nan,
            dims=('profile', vertical_coord),
            name=f'{field}_std',
        )
        dset_profile[f'{field}_med'] = xr.DataArray(
            np.ones((len(profiles), len(zgrid_center))) * np.nan,
            dims=('profile', vertical_coord),
            name=f'{field}_med',
        )

    allfields = fields + [f'{f}_std' for f in fields] + [f'{f}_med' for f in fields]

    for i, profile in enumerate(profiles):
        dfi = df.loc[df.profile_id == profile]

        # compute and add coordinate
        y, m, d = int(dfi.year.mean()), int(dfi.month.mean()), int(dfi.day.mean())
        dset_profile['time'].data[i] = np.datetime64(f'{y:04d}-{m:02d}-{d:02d}')
        dset_profile['month'].data[i] = dfi.month.mean()
        dset_profile['year'].data[i] = dfi.year.mean()
        dset_profile['lat'].data[i] = dfi.lat.mean()
        dset_profile['lon'].data[i] = dfi.lon.mean()
        dset_profile['flight_id'].data[i] = dfi.flight_id.values[0]
        dset_profile['campaign'].data[i] = groups_find_campaign(df, dfi.flight_id.values[0])

        # compute the binned profile
        profile_data = compute_profile_edge_array(dfi, vertical_coord, zgrid_edges, fields)

        # copy data
        for field in allfields:
            data = profile_data[field].data

            if dset_profile[field].dims == ('profile'):
                dset_profile[field].data[i] = data

            elif dset_profile[field].dims == ('profile', vertical_coord):
                dset_profile[field].data[i, :] = data

            else:
                raise ValueError('unknown dims')

    # compute the day of the year
    dset_profile.doy.data[:] = util.day_of_year_noleap(dset_profile.time.values)

    return dset_profile


def compute_profile_edge_array(dfi, vertical_coord, zgrid_edges, fields, compute_centers=False):
    """compute a profile dataset"""

    vertical_coord_field = 'alt' if vertical_coord == 'z' else vertical_coord
    zscale_factor = 1e-3 if vertical_coord_field == 'alt' else 1.


    zdata = dfi[vertical_coord_field] * zscale_factor
    xbinned = dfi[fields].groupby(pd.cut(zdata, zgrid_edges))

    xbinned_mean = dfi[fields].groupby(pd.cut(zdata, zgrid_edges)).mean()
    xbinned_med = dfi[fields].groupby(pd.cut(zdata, zgrid_edges)).median()
    xbinned_std = dfi[fields].groupby(pd.cut(zdata, zgrid_edges)).std()

    dset_profile = xr.Dataset()
    for field in fields:
        dset_profile[field] = xr.DataArray(xbinned_mean[field].values, dims=(vertical_coord))
        dset_profile[f'{field}_std'] = xr.DataArray(xbinned_std[field].values, dims=(vertical_coord))
        dset_profile[f'{field}_med'] = xr.DataArray(xbinned_med[field].values, dims=(vertical_coord))

    if compute_centers:
        dset_profile[vertical_coord] = xr.DataArray(
            np.vstack((zgrid_edges[:-1], zgrid_edges[1:])).mean(axis=0),
            dims=(vertical_coord),)

    return dset_profile


def compute_ds_binned(df, vertical_coord, theta_bins, fields):
    """compute binned dataset using list of tuples as bin boundaries,
       i.e., theta_bins=[(297.5, 302.5), (272.5, 277.5),]
    """
    vertical_coord_field = 'alt' if vertical_coord == 'z' else vertical_coord
    zscale_factor = 1e-3 if vertical_coord_field == 'alt' else 1.

    zdata = df[vertical_coord_field] * zscale_factor

    ds_list = []
    for zlo_zhi in theta_bins:
        xbinned_mean = df[fields].groupby(pd.cut(zdata, zlo_zhi)).mean()
        xbinned_std = df[fields].groupby(pd.cut(zdata, zlo_zhi)).std()
        xbinned_med = df[fields].groupby(pd.cut(zdata, zlo_zhi)).median()

        ds = xr.Dataset()
        for field in fields:
            ds[field] = xr.DataArray(xbinned_mean[field].values, dims=(vertical_coord+'_bins'))
            ds[f'{field}_std'] = xr.DataArray(xbinned_std[field].values, dims=(vertical_coord+'_bins'))
            ds[f'{field}_med'] = xr.DataArray(xbinned_med[field].values, dims=(vertical_coord+'_bins'))
        ds_list.append(ds)

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
    return dso.set_coords('theta_bins_bounds')

def vertical_gradient(ds):
    """compute the vertical gradient from profile datasets"""
    if 'theta' in ds.coords:
        slice_bins = OrderedDict([
            ('270-300 K (10 K)', (
                slice(
                    ds.theta.values[int(np.where(ds.theta_bounds[:, 0] == 265.)[0])],
                    ds.theta.values[int(np.where(ds.theta_bounds[:, 1] == 275.)[0])],
                ),
                slice(
                    ds.theta.values[int(np.where(ds.theta_bounds[:, 0] == 295.)[0])],
                    ds.theta.values[int(np.where(ds.theta_bounds[:, 1] == 305.)[0])],
                )
            )),
            ('275-300 K (10 K)', (
                slice(
                    ds.theta.values[int(np.where(ds.theta_bounds[:, 0] == 270.)[0])],
                    ds.theta.values[int(np.where(ds.theta_bounds[:, 1] == 280.)[0])],
                ),
                slice(
                    ds.theta.values[int(np.where(ds.theta_bounds[:, 0] == 295.)[0])],
                    ds.theta.values[int(np.where(ds.theta_bounds[:, 1] == 305.)[0])],
                )
            )),
        ])
        coord_dim = 'theta'

    elif 'z' in ds.coords:
        slice_bins = OrderedDict([
            ('0.5-2.5 km', (slice(0., 1.), slice(2., 3.))),
            ('0.5-6.5 km', (slice(0., 1.), slice(6., 7.))),
        ])
        coord_dim = 'z'

    else:
        raise ValueError('no theta or z')

    gradient_coord = xr.DataArray(list(slice_bins.keys()), dims='vg', name='vg')

    vg_list = []
    for k, sb in slice_bins.items():
        b1 = ds[coord_dim+'_bounds'].sel({coord_dim: sb[0]}).values
        b2 = ds[coord_dim+'_bounds'].sel({coord_dim: sb[1]}).values
        b1 = (b1[0, 0], b1[-1, 1])
        b2 = (b2[0, 0], b2[-1, 1])
        print(f'computing gradient: {k}')
        print(f'bin bounds: {b1} - {b2}')
        vg_list.append(
            ds.sel({coord_dim: sb[0]}).median(coord_dim) - ds.sel({coord_dim: sb[1]}).median(coord_dim)
        )

    return xr.concat(vg_list, dim=gradient_coord).reset_coords(['doy', 'month', 'year'])


def make_theta_bins(lbin=275., ubin=300., udθ=10., ldθ=10., lbin_as_upper_bound=False, ubin_as_lower_bound=False):
    """generate 'theta_bin' tuple"""

    if ubin_as_lower_bound:
        ubin_values = (ubin, np.Inf)
    else:
        ubin_values = (ubin - udθ / 2, ubin + udθ / 2)

    if lbin_as_upper_bound:
        lbin_values = (-np.Inf, lbin)
    else:
        lbin_values = (lbin - ldθ / 2, lbin + ldθ / 2)

    return (ubin_values, lbin_values)

def flight_gradients(dfs, theta_bins, gradient_lat_range, bin_aggregation_method='median', constituent='co2'):
    """Compute gradient metric on individual flights"""

    compute_gradients = partial(campaign_gradients,
        theta_bins=theta_bins,
        gradient_lat_range=gradient_lat_range,
        bin_aggregation_method=bin_aggregation_method,
        constituent=constituent)

    if 'obs' in dfs:
        mkey = 'obs'
    else:
        assert len(dfs.keys()) == 1
        mkey = list(dfs.keys())[0]

    df_list = []
    for c in campaign_list:
        dfc = dfs[mkey].loc[dfs[mkey].campaign_id == c].copy()

        profile_ids = dfc.profile_id.unique()
        for profile_id in profile_ids:
            dfs_c = {k: df.loc[df.profile_id == profile_id] for k, df in dfs.items()}
            y = int(dfs_c[mkey].year.mean())
            m = int(dfs_c[mkey].month.mean())
            d = int(dfs_c[mkey].day.mean())
            df = compute_gradients(dfs_c, [c])
            df['profile_id'] = profile_id
            df['flight_id'] = dfs_c[mkey].flight_id.iloc[0]
            df['doy'] = util.day_of_year_noleap(np.array([np.datetime64(f'{y:04d}-{m:02d}-{d:02d}')]))
            df_list.append(df.reset_index().rename({'campaign': 'campaign_id'}, axis=1).set_index(['campaign_id', 'profile_id']))

    df = pd.concat(df_list).reset_index()
    first_cols = ['campaign_id', 'flight_id', 'profile_id', 'doy']
    return df[first_cols+[c for c in df.columns if c not in first_cols]]


def campaign_gradients(dfs, campaign_sel_list, theta_bins, gradient_lat_range,
                       bin_aggregation_method='median', constituent='co2', filter_strat=True,
                      ):
    """
    compute campaign-mean gradients, lumping campaigns according to `campaign_selector`.
    """
    if isinstance(dfs, pd.DataFrame):
        dfs = {'dummy-key': dfs}

    constituent = constituent.lower()

    bin_aggregation = util.nanmedian if bin_aggregation_method == 'median' else np.nanmean

    lat_lo = gradient_lat_range[0]
    lat_hi = gradient_lat_range[1]

    lines = []
    for campaign in campaign_sel_list:

        gradients = dict(campaign=campaign)
        gradient_diff_list = []
        gradient_list = []

        if campaign == 'ORCAS':
            clist = ['ORCAS-J', 'ORCAS-F']
        else:
            clist = [campaign]

        for m, df in dfs.items():
            dfc = df.loc[df.campaign_id.isin(clist)]

            if len(dfc) == 0:
                continue

            if filter_strat:
                if 'strat' not in dfc:
                    pass
                    #print(f'Warning: cannot apply strat filter to {campaign}')
            
            vlist = [v for v in constituent_vars[constituent][campaign] if v in dfc]
            
            if m in ['medusa', 'pfp']:
                assert len(vlist) <= 1
            else:
                vlist = vlist+[constituent]
            
            vlist_diff = [v for v in constituent_mvars[constituent][campaign] if v in dfc]
    
            if m in ['medusa' 'pfp']:
                assert len(vlist_diff) <= 1

            if not vlist:
                continue

            df_list = []
            for lo, hi in theta_bins:
                loc_select = (lo <= dfc.theta) & (dfc.theta < hi)
                loc_select = loc_select & (lat_lo <= dfc.lat) & (dfc.lat <= lat_hi)
                if filter_strat:
                    if 'strat' in dfc:
                        not_strat = (dfc.strat == 0)
                        nstrat = np.sum(loc_select) - np.sum(loc_select & not_strat)
                        loc_select = loc_select & not_strat
                df_list.append(dfc.loc[loc_select][vlist+vlist_diff].apply(bin_aggregation))

            # compute the gradient
            for v in vlist:
                gradient = df_list[1][v] - df_list[0][v]
                gradients[v] = gradient
                if ~np.isnan(gradient):
                    gradient_list.append(gradient)

            # compute the gradient diff
            for v in vlist_diff:
                gradient = df_list[1][v] - df_list[0][v]
                gradients[v] = gradient
                if ~np.isnan(gradient):
                    gradient_diff_list.append(gradient)

        gradients['gradient_mean'] = np.nan
        if gradient_list:
            gradients['gradient_mean'] = np.mean(gradient_list)

        gradients['gradient_std'] = np.nan
        if gradient_diff_list:
            gradients['gradient_std'] = np.std(gradient_diff_list+[0.], ddof=0)

        lines.append(gradients)

    df = pd.DataFrame(lines)
    df = df.set_index('campaign')
    df = df[[c for c in df.columns if c not in ['gradient_mean', 'gradient_std']]+['gradient_mean', 'gradient_std']]

    return df


def campaign_datestr(tb):
    monlabs = np.array(["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"])

    y, m, d = util.datetime64_parts_arr(tb)
    assert y[0] == y[1]

    if m[0] == m[1]:
        return f'{d[0]}-{d[1]} {monlabs[m[0]-1]} {y[0]}'
    else:
        return f'{d[0]} {monlabs[m[0]-1]} - {d[1]} {monlabs[m[1]-1]} {y[0]}'


def get_property_property(df, campaign_sel_list,
                          xname, yname,
                          theta_bin=(-np.Inf, np.Inf),
                          lat_range=(-90., 90.),
                          filter_strat=True):
    """return paired arrays of x and y and some other variable"""

    X = []; Y = [];
    for campaign_id in campaign_sel_list:
        # select campaign
        if campaign_id == 'ORCAS':
            clist = ['ORCAS-J', 'ORCAS-F']
        else:
            clist = [campaign_id]
        dfi = df.loc[df.campaign_id.isin(clist)]

        # select midtrop
        dfi_mt = df_midtrop(dfi)

        # select theta/lat region
        sel = (theta_bin[0] <= dfi.theta) & (dfi.theta <= theta_bin[1])
        sel = sel & (lat_range[0] <= dfi.lat) & (dfi.lat <= lat_range[1])
        if filter_strat:
            sel = sel & (dfi.strat == 0)

        dfi = dfi.loc[sel]
        y = (dfi[yname] - dfi_mt[yname].mean()).values
        x = (dfi[xname] - dfi_mt[xname].mean()).values
        k = (np.isnan(y) | np.isnan(x))
        Y.extend(list(y[~k]))
        X.extend(list(x[~k]))

    return np.array(X), np.array(Y)



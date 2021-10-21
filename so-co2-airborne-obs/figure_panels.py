import numpy as np

import yaml
import pickle

import scipy.interpolate as interp

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import matplotlib.path as mpath
import matplotlib.colors as colors
import matplotlib.gridspec as gridspec

import seaborn as sns
import cmocean

import cartopy.crs as ccrs
from cartopy.util import add_cyclic_point

import util
import obs_surface
import obs_aircraft

co2_anom_bounds = (-1.6, 2.3)
co2_delta = 0.1
levels = np.arange(co2_anom_bounds[0], co2_anom_bounds[1], co2_delta)
divnorm = colors.TwoSlopeNorm(vmin=levels.min(), vcenter=0., vmax=levels.max())

cmap = cmocean.cm.curl
spo_mean_during_orcas = 398.65 # noaa in situ

levels_co2 = np.arange(
    spo_mean_during_orcas - co2_anom_bounds[1],
    spo_mean_during_orcas + co2_anom_bounds[1],
    co2_delta
)

divnorm_co2 = colors.TwoSlopeNorm(
    vmin=levels_co2.min(), vcenter=spo_mean_during_orcas,
    vmax=levels_co2.max()
)


alt_lim = (0, 10.5)

sns_palette = 'colorblind' #None

palette_colors = sns.color_palette(sns_palette).as_hex()

co2_colors = {'CO2_FFF': sns.color_palette(sns_palette).as_hex()[3],
              'CO2_LND': sns.color_palette(sns_palette).as_hex()[2],
              'CO2_OCN': sns.color_palette(sns_palette).as_hex()[0],
              'CO2_LND+CO2_FFF': sns.color_palette(sns_palette).as_hex()[1],
              'CO2': 'k', #'#4d4d4d',
             }

co2_names = {'CO2_FFF': 'Fossil',
             'CO2_LND': 'Land',
             'CO2_OCN': 'Ocean',
             'CO2_LND+CO2_FFF': 'Land+Fossil',
             'CO2': 'Total'}

monlabs = np.array(["J", "F", "M", "A", "M", "J", "J", "A", "S", "O", "N", "D"])
monlabs_ant = np.concatenate((monlabs[6:], monlabs[:6]))

bomday = [0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334, 365]
mncntr = [15.5, 45., 74.5, 105., 135.5, 166., 196.5, 227.5, 258., 288.5, 319., 349.5]

marker_order = [
    ".", "o", "v", "^", "<", ">", "8", "s", "p", "P",
    "*", "X", "D", "d", "h", "H", 4, 5, 6, 7, 8, 9, 10, 11
]
marker_order += [f'${a}$' for a in [chr(i).upper() for i in range(97,97+26)]]


def marker_colororder(marker_spec, palette=None):
    """replace colors in marker spec"""
    
    import seaborn as sns
    
    current_palette = sns.color_palette(palette, len(marker_spec.keys()))
    colors = current_palette.as_hex()

    for label, spec in marker_spec.items():
        color = colors.pop(0)
        for attr, value in spec.items():
            if 'color' in attr and '#' in value:
                marker_spec[label][attr] = color
                

def marker_spec_co2_inst():
    with open('data/marker_spec_co2_institutions.yaml', 'r') as fid:
        marker_spec = yaml.safe_load(fid)
    marker_colororder(marker_spec, sns_palette)
    return marker_spec


def marker_spec_models():
    with open('data/model-description.yaml', 'r') as fid:
        model_info = yaml.safe_load(fid)
    models = model_info.keys()
    model_code = util.list_set([info['info']['group'] for m, info in model_info.items()])
    colors = sns.color_palette(sns_palette, 10).as_hex()
    # reorder
    colors = [colors[i] for i in [0, 1, 2, 3, 4, 5, 9, 8, 6, 7]]
    code_color = {k: colors.pop(0) for k in model_code}# if k != 'pCO2'}
    code_marker_order = {
        'CT-NOAA': ['.', 'o'],
        'CTE': ['v', '^'],
        'MIROC': ['<', '>', 'X', 'D'],
        'CAMS': ['D', 's', 'p'],
        'CarboScope': ['P', 'd', 'H', 'X', '$O$', '$N$', '$Z$'],
        'CESM': ['D', '*'],
        'TM5': ['p', '.', 'o', 's',]*2,
        'pCO2': ['*'],
    }

    marker_spec = {}
    for model, info in model_info.items():
        #color = 'indigo' if model == 'SOM-FFN' else code_color[info['info']['group']]
        color = code_color[info['info']['group']]
        label = info['info']['label']
        tracers = ['CO2', 'CO2_SUM', 'CO2_OCN', 'CO2_LND', 'CO2_FFF', 'CO2_LND+CO2_FFF']

        marker_spec[model] = {k: {'color': color, 'label': label} for k in tracers}

        marker_spec[model]['CO2_FFF']['marker'] = 'x'
        marker_spec[model]['CO2_FFF']['label'] = f'{label} (fossil)'

        marker_spec[model]['CO2_LND']['marker'] = '+'
        marker_spec[model]['CO2_LND']['label'] = f'{label} (land)'

        if code_marker_order[info['info']['group']]:
            marker = code_marker_order[info['info']['group']].pop(0)
        else:
            print(f'out of markers for {model}')
            marker = '$n$'
        marker_spec[model]['CO2']['marker'] = marker
        marker_spec[model]['CO2_SUM']['marker'] = marker
        marker_spec[model]['CO2_OCN']['marker'] = marker
        marker_spec[model]['CO2_LND+CO2_FFF']['marker'] = marker        

    for model, tracer_dict in marker_spec.items():
        for tracer, spec in tracer_dict.items():
            if 'TM5-Flux' in model:
                marker_spec[model][tracer]['markeredgecolor'] = 'k'               
            
            marker = spec['marker']
            if marker == '*':
                marker_spec[model][tracer]['markersize'] = 10
            elif marker in ['H', 'p', 'P']:
                marker_spec[model][tracer]['markersize'] = 8
            elif marker in ['D']:
                marker_spec[model][tracer]['markersize'] = 5
            else:
                marker_spec[model][tracer]['markersize'] = 6

    return marker_spec


def marker_spec_campaigns(lump_orcas=False):
    campaign_info = obs_aircraft.get_campaign_info(verbose=False, lump_orcas=lump_orcas)        
    campaigns = list(campaign_info.keys())
    current_palette = sns.color_palette(sns_palette, len(campaigns)).as_hex()
    markers = list(Line2D.filled_markers)

    marker_spec = {c: dict(
        color=current_palette.pop(0), marker=markers.pop(0)
    ) for c in campaigns} # if c != 'ORCAS-F'}
    #marker_spec['ORCAS-F'] = marker_spec['ORCAS-J']
    #marker_spec['ORCAS'] = marker_spec['ORCAS-J']

    return marker_spec


def marker_spec_surface_stations():
    southern_ocean_stn_list = [s for s in obs_surface.southern_ocean_stn_list]
    southern_ocean_stn_list += [s for s in obs_surface.southern_ocean_stn_list_sf6
                               if s not in southern_ocean_stn_list]
    markers = list(Line2D.filled_markers)
    current_palette = sns.color_palette(sns_palette, len(southern_ocean_stn_list))
    color_list = list(current_palette.as_hex())
    return {
        stn: dict(label=stn, marker=markers.pop(0), color=color_list.pop(0),)
        for stn in southern_ocean_stn_list
    }


def stn_v_lat(dset, ax, constituent='CO2', include_LMG=False):
    marker_spec = marker_spec_co2_inst()

    if constituent == 'CO2':
        y_text = -0.58 if not include_LMG else -0.8
    else:
        y_text = -0.07

    # loop over stations and plot
    text_added = []
    for record in dset.record.values:

        if 'Multi' in record:
            continue

        if not include_LMG and 'LMG' in record:
            continue

        # pull data
        da = dset.sel(record=record)
        inst = str(da.institution.values)
        stncode = str(da.stncode.values) #record[:3]
        xi = np.float(da.lat.values)
        yi = np.float(da.mean().values)
        yerri = np.float(da.std().values)

        spec = marker_spec[inst].copy()
        if 'LMG' in stncode:
            spec['markeredgecolor'] = 'k'
        ax.errorbar(xi, yi, yerr=yerri, label=inst, **spec)

        # add station text
        yoffset = 0
        if stncode == 'CRZ':
            offset = -1.6
        elif stncode == 'SYO':
            offset = -1.2
        elif stncode == 'MAA':
            offset = -1.
        elif stncode == 'CYA':
            offset = -0.7
        elif stncode == 'BHD':
            offset = -1.2
        elif stncode == 'CGO':
            offset = 0.2
        elif stncode == 'AMS':
            offset = 0
        elif 'LMG' in stncode:
            yoffset = 1.3
        else:
            offset = -0.5

        if stncode not in text_added:
            ax.text(xi + offset, y_text + yoffset, stncode,
                    rotation=90, color='k', fontsize=8)
            text_added.append(stncode)

    # add interpolation
    y = dset.groupby('lat').mean(['time', 'record'])
    x = y.lat

    spl = interp.UnivariateSpline(x, y)
    xs = np.linspace(x.min(), x.max(), 100)
    ax.plot(xs, spl(xs), 'k', lw=1)

    ax.axvspan(
         -80, -45,
         color=palette_colors[0], alpha=0.2
     )

    # plot tweaks
    ax.axhline(0., linewidth=1, color='k')
    ax.set_xlabel('Latitude [°N]')
    ylm = ax.get_ylim()

    if constituent == 'CO2':
        ax.set_ylabel('$\Delta$CO$_2$ [ppm]')
        ax.set_ylim([-0.63, 0.4]) #ylm[1]*1.03])
        ax.set_title('Surface obs: SO CO$_2$ minus SPO')
    else:
        ax.set_ylim([-0.095, ylm[1]])
        ax.set_ylabel('$\Delta_{y}$SF$_6$ [ppt]')
        ax.set_xticks(np.arange(-90., 0., 10.))
        ax.set_title('Surface obs: SF$_6$ minus SPO')
        axR = ax.twinx()
        axR.set_ylim(np.array(ylm)*14)
        #axR.set_yticks(ytick*14.)
        axR.set_ylabel('Estimated Fossil-Fuel $\Delta_{y}$CO$_2$ [ppm]')


    #ax.set_title(f'{season} Surface stations: Observed CO$_2$ minus SPO')

    # legend
#     legend_elements = [Line2D([0], [0], label=inst, linestyle=None, **spec)
#                        for inst, spec in marker_spec.items()]

#     ax.legend(handles=legend_elements, ncol=4, fontsize=8);


def obs_srf_trends_djf_jja(axs, ds_djf, ds_jja, constituent='CO2'):
    assert constituent in ['CO2', 'SF6']

    trend = {'djf': {}, 'jja': {}}

    def ammendments(ax):
        ax.legend(ncol=4) #bbox_to_anchor=(1.05, 1.05))
        ax.axhline(0, color='k', lw=1);
        ax.set_xticks(np.arange(1998, 2022, 2));
        ticklabels = np.arange(1998, 2022, 2).astype(str)
        ticklabels[::2] = ''
        ax.set_xticklabels(ticklabels)
        ax.set_xlim([1998, 2021])

        if constituent == 'CO2':
            ax.set_ylim([-0.73, 0.63]);
            ax.set_ylabel('$\Delta_{y}$CO$_2$ [ppm]')
        else:
            ylm = [-0.11, 0.11]
            ax.set_ylim(ylm);
            ax.set_ylabel('$\Delta_{y}$SF$_6$ [ppt]')
            axR = ax.twinx()
            axR.set_ylim(np.array(ylm)*util.CO2_SF6_emission_ratio)
            #axR.set_yticks(ytick*14.)
            axR.set_ylabel('Estimated Fossil-Fuel $\Delta_{y}$CO$_2$ [ppm]')

    marker_spec = marker_spec_surface_stations()

    #------------------------------
    # panel A
    dset = ds_djf
    ax = axs[0]

    x = dset.time + 0.04
    for i, stn_code in enumerate(dset.stncode.values):

        y = dset.sel(stncode=stn_code)
        ax.plot(x, y, linestyle='-', **marker_spec[stn_code])
    trend['djf']['x'] = x
    trend['djf']['y'] = dset.mean('stncode')


    y = dset.mean('stncode').rolling(time=3, center=True).mean()
    x = y.time + 0.04
    ax.plot(x, y, 'k', linewidth=3)
    #ax.plot(x, eof_a_so_djf['CO2'].mean('stncode'), 'r', lw=2)

    #model = sm.OLS(np.nanmean(Y, axis=1), sm.add_constant(x))
    #fitted = model.fit()
    #print(fitted.summary())
    ammendments(ax)
    ax.set_title('DJF, SPO subtracted')

    #------------------------------
    # panel B
    dset = ds_jja
    ax = axs[1]

    x = dset.time + 0.54
    for i, stn_code in enumerate(dset.stncode.values):

        y = dset.sel(stncode=stn_code)
        ax.plot(x, y, linestyle='-', **marker_spec[stn_code])

    trend['jja']['x'] = x
    trend['jja']['y'] = dset.mean('stncode')

    y = dset.mean('stncode').rolling(time=3, center=True).mean()
    x = y.time + 0.54
    ax.plot(x, y, 'k', lw=3)
    #model = sm.OLS(np.nanmean(Y, axis=1), sm.add_constant(x))
    #fitted = model.fit()
    #print(fitted.summary())

    ammendments(ax)
    ax.set_title('JJA, SPO subtracted')



def obs_srf_seasonal(ax, dset, constituent='CO2', just_the_mean=False):

    marker_spec = marker_spec_surface_stations()

    x = dset.month - 0.5
    for i, stn_code in enumerate(dset.stncode.values):
        y = dset[constituent].sel(stncode=stn_code)
        y = util.antyear_monthly(y)
        if not just_the_mean:
            ax.plot(x, y, linestyle='-', **marker_spec[stn_code])

    ax.plot(x, util.antyear_monthly(dset[constituent].mean('stncode')),
            '-', linewidth=3, color='k', label='$\Delta_{ y}$CO$_2$')

    ylm = [-0.63, 0.41]
    ax.axhline(0., linewidth=1., color='k')
    ax.set_xticks(np.arange(0, 13, 1))
    ax.set_xticklabels([f'        {m}' for m in util.antyear_monthly(monlabs)]+[''])

    if constituent == 'CO2':
        ax.set_ylabel('$\Delta$CO$_2$ [ppm]')
        ax.set_ylim(ylm)
        ax.set_title('Surface obs: SO CO$_2$ minus SPO')
    else:
        ax.set_ylabel('$\Delta$SF$_6$ [ppt]')
        ax.set_ylim(np.array(ylm)/util.CO2_SF6_emission_ratio)
        ax.axhline(0., linewidth=1., color='k')
        axR = ax.twinx()
        axR.set_ylim(np.array(ylm))
        axR.set_ylabel('Estimated Fossil-Fuel $\Delta_{y}$CO$_2$ [ppm]')
        ax.set_title('Surface obs: SF$_6$ minus SPO')
    ax.legend(ncol=2, columnspacing=0.8, fontsize=8, frameon=False)



def horizontal_gradient_seasonal(
    ds, ax,
    co2_var_list=['CO2', 'CO2_LND', 'CO2_FFF', 'CO2_OCN'],
    window=30,
    linewidth=3,
):
    labels = []
    handles = []
    for v in co2_var_list:
        v_spo = f'{v}_SPO'
        x, y = util.antyear_daily(ds.time, util.mavg_periodic(ds[v] - ds[v_spo], window))
        h = ax.plot(x, y, color=co2_colors[v], linestyle='-', linewidth=linewidth)

        handles.append(h[0])
        labels.append(co2_names[v])


    ax.axhline(0, color='k', linewidth=1.)
    ax.legend(handles, labels, loc='lower left')
    ax.set_xlim((0, 365))
    ax.set_xticks(bomday)
    ax.set_xticklabels([f'        {m}' for m in monlabs_ant]+[''])

    ax.set_ylim([-0.63, 0.41])
    ax.set_ylabel('$\Delta_{ y}$CO$_2$ [ppm]')
    return handles, labels


def models_fluxes_seasonal(dsets, ax):

    model_list = list(dsets.keys())

    current_palette = sns.color_palette('colorblind', len(model_list))
    colors = current_palette.as_hex()

    for model in model_list:
        x = dsets[model].month - 0.5
        y = dsets[model].SFCO2_OCN
        y = util.antyear_monthly(y)
        ax.plot(x, y, linestyle='-', label=model,
                lw=2.,
                color=colors.pop(0))

    ax.set_ylabel('$\Delta$CO$_2$ [ppm]')
    #ax.set_ylim([-0.63, 0.41])
    ax.axhline(0., linewidth=1., color='k')

    ax.set_xticks(np.arange(0, 13, 1))
    ax.set_xticklabels([f'        {m}' for m in util.antyear_monthly(monlabs)]+[''])

    #ax.set_title('Observed CO$_2$, SPO subtracted')

    ax.legend(ncol=2, columnspacing=0.8, loc='lower left')


def aircraft_CO2_xsection(XI, YI, CO2, THETA, ax, cax, remove_cf=False):

    levels_loc = levels #_co2
    divnorm_loc = divnorm #_co2

    cf = ax.contourf(XI, YI, CO2,
                      levels=levels_loc,
                      norm=divnorm_loc,
                      cmap=cmap, #'PuOr_r',
                      extend='both')

    cs = ax.contour(XI, YI, THETA,
                      levels=np.arange(255., 350., 10.),
                      linewidths=1,
                       colors='gray')
    fmt = '%d'
    if cax is not None:
        cb = plt.colorbar(cf, cax=cax)
        cb.ax.set_title('$\Delta$CO$_2$ [ppm]');

    if remove_cf:
        for c in cf.collections:
            c.remove()

        for c in cs.collections:
            c.remove()
    else:
        lb = plt.clabel(cs, fontsize=8,
                inline=True,
                fmt=fmt)
    ax.set_ylim((0, 10.5))
    ax.set_xlim(-91.25, -28.75)
    ax.set_xlabel('Latitude [°N]')
    ax.set_ylabel('Altitude [km]')


def vertical_profile(dset_sum, dset_win, ax):
    rgb_sum = '#029e73' #np.array([24, 127, 122])/255
    rgb_win = '#d55e00' #np.array([169, 61, 96])/255

    h_sum = ax.errorbar(
        dset_sum.co2mmidtrop_flt, dset_sum.z,
        xerr=dset_sum.co2mmidtrop_flt_std,
        fmt='.-',
        color=rgb_sum,
    )
    h_win = ax.errorbar(
        dset_win.co2mmidtrop_flt, dset_win.z,
        xerr=dset_win.co2mmidtrop_flt_std,
        fmt='.-',
        color=rgb_win,
    )

    ax.axvline(0., color='k', linewidth=1.)

    ax.set_ylabel('Altitude [km]')
    ax.set_xlabel('$\Delta$CO$_2$ [ppm]')
    ax.legend([h_sum, h_win], ['Summer', 'Winter'], loc='upper left')

    ax.set_ylim((-0.2, 10.2))
    ax.set_xlim((-2.1, 1.5))

def model_vertical_profile_season(dset, season, ax):
    for v in ['CO2', 'CO2_LND', 'CO2_FFF', 'CO2_OCN']:
        ax.plot(dset[v].sel(season=season), dset.zlev*1e-3,
                       color=co2_colors[v],
                       label=co2_names[v],
                       linewidth=2,
                      )
    ax.axvline(0., color='k', linewidth=1.)
    ax.set_ylabel('Altitude [km]')
    ax.set_xlabel('$\Delta$CO$_2$ [ppm]')
    ax.legend();

    ax.set_ylim((-0.2, 10.2))
    ax.set_xlim(co2_anom_bounds)
    ax.set_title(f'{season} minus deseasonalized SPO')

def model_CO2_xsection(lat, zlev, co2, theta, ax, cax=None, title=None,):

    co2[:, 0] = np.nan

    cf = ax.contourf(lat, zlev, co2,
                     levels=levels,
                     norm=divnorm,
                     cmap=cmap,
                     extend='both')

    cs = ax.contour(lat, zlev, theta,
                  levels=np.arange(255., 350., 10.),
                  linewidths=1,
                   colors='gray')
    lb = plt.clabel(cs, fontsize=8,
        inline=True,
        fmt='%d')

    ax.set_ylim(alt_lim)
    ax.set_xlim(-91.25, -28.75)
    if title is not None:
        ax.set_title(title)

    ax.set_xlabel('Latitude [°N]')
    ax.set_ylabel('Altitude [km]')

    if cax is not None:
        cb = plt.colorbar(cf, cax=cax)
        cb.ax.set_title('$\Delta$CO$_2$ [ppm]');

    return cf



def model_CO2_map(lon, lat, field, ax, cax=None, plot_stations=True, stninfo=None):

    ax.set_global()
    ax.set_extent([180, -180, -90,  -30], crs=ccrs.PlateCarree())
    # Compute a circle in axes coordinates, which we can use as a boundary
    # for the map. We can pan/zoom as much as we like - the boundary will be
    # permanently circular.
    theta = np.linspace(0, 2*np.pi, 100)
    center, radius = [0.5, 0.5], 0.5
    verts = np.vstack([np.sin(theta), np.cos(theta)]).T
    circle = mpath.Path(verts * radius + center)
    ax.set_boundary(circle, transform=ax.transAxes)

    cyclic_field, cyclic_lon = add_cyclic_point(field, coord=lon)

    cf = ax.contourf(cyclic_lon, lat, cyclic_field,
                     cmap=cmap,
                     extend='both',
                     levels=levels,
                     transform=ccrs.PlateCarree())


    ax.coastlines('50m')
    ax.gridlines()


    #cbax = fig.add_axes([0.87,0.1,0.03,0.75])
    #cb = plt.colorbar(cf,shrink=0.75,pad=0.01)#,cax=cbax)
    if cax is not None:
        cb = plt.colorbar(cf, cax=cax)
        cb.ax.set_title('$\Delta$CO$_2$ [ppm]');
    #cb.ax.tick_params(labelsize=15)

    if plot_stations:
        util.label_stations(ax, stninfo, fontsize=8) #, stninfo)label_stations(ax)

    return cf

def model_CO2_vertical_hovmoller(time, zlev, co2, ax, cax=None, title=None):

    jfmamj = time < 182.
    jasond = time >= 182.
    time = np.concatenate((time[jasond] - 181, time[jfmamj] + 184))
    co2 = np.concatenate((co2[jasond, :], co2[jfmamj, :]))


    cf = ax.contourf(time, zlev, co2.T,
                     levels=levels,
                     norm=divnorm,
                     cmap=cmap,
                     extend='both')


    ax.set_xticks(bomday)
    ax.set_xticklabels([f'        {m}' for m in monlabs_ant]+[''])
    #ax.set_xticklabels([f'         {m}' for m in monlabs_ant])


    if title is not None:
        ax.set_title(title)

    if cax is not None:
        cb = plt.colorbar(cf, cax=cax)
        cb.ax.set_title('$\Delta$CO$_2$ [ppm]');

    return cf


def four_xsection_canvas():
    fig = plt.figure(figsize=(12, 8))

    gs = gridspec.GridSpec(nrows=2, ncols=3, width_ratios=(1, 1, 0.02))
    gs.update(left=0.05, right=0.95, hspace=0.25, wspace=0.15)

    # total
    ax = {}
    ax['CO2'] = plt.subplot(gs[0, 0])
    ax['CO2_OCN'] = plt.subplot(gs[1, 0])
    ax['CO2_LND'] = plt.subplot(gs[0, 1])
    ax['CO2_FFF']= plt.subplot(gs[1, 1])

    cax = plt.subplot(gs[:, 2])

    return fig, ax, cax

def obs_aircraft_season_hovmoller(dset_seasonal, ax, cax=None):

    monlabs = np.array(["J", "F", "M", "A", "M", "J", "J", "A", "S", "O", "N", "D"])
    monlabs_ant = np.concatenate((monlabs[6:], monlabs[:6]))

    field = dset_seasonal.DCO2.values
    field = np.concatenate((field[:, 6:], field[:, :6]), axis=1)

    cf = ax.contourf(
        np.concatenate(([0.5], dset_seasonal.x, [12.5])),
        dset_seasonal.y,
        np.concatenate((field[:, 0:1], field, field[:, -1:]), axis=1),
        levels=levels,
        norm=divnorm,
        cmap=cmap,
        extend='both',
    )

    if cax is not None:
        cb = plt.colorbar(cf, cax=cax)
        cb.ax.set_title('$\Delta$CO$_2$ [ppm]');

    ax.set_xlim((0.5, 12.5))
    ax.set_xticks(np.arange(0.5, 13.5, 1));
    ax.set_xticklabels([f'        {m}' for m in monlabs_ant]+['']);

    ax.set_ylabel('Altitude [km]')
    ax.set_title('Observed CO$_2$ deseas.-SPO subtracted');


def vertical_gradient_seasonal_bin_ill(
    ds, ax,
    co2_var_list=['CO2', 'CO2_LND', 'CO2_FFF', 'CO2_OCN'],
    window=30,
    linewidth=3,
):

    labels = []
    handles = []
    for v in co2_var_list:
        x, y = util.antyear_daily(
            ds.time, util.mavg_periodic(util.upper_bin(ds[v]) - util.upper_bin(ds[v]).mean(), window)
        )
        h = ax.plot(x, y, color=co2_colors[v], linestyle='-', linewidth=linewidth)

        x, y = util.antyear_daily(
            ds.time, util.mavg_periodic(util.lower_bin(ds[v]) - util.lower_bin(ds[v]).mean(), window)
        )
        ax.plot(x, y, color=co2_colors[v], linestyle='--', linewidth=linewidth)

        handles.append(h[0])
        labels.append(co2_names[v])

    ax.set_xticks(bomday)
    ax.set_ylabel('CO$_2$ anomaly [ppm]')
    ax.axhline(0, color='k', linewidth=1.)

    custom_lines = [Line2D([0], [0], color='k', lw=2, linestyle='-'),
                    Line2D([0], [0], color='k', lw=2, linestyle='--'),]

    lg = ax.legend(handles, labels, loc='lower right')
    ax.add_artist(lg)

    ax.legend(
        custom_lines, [
            f'{util.zbins[0, 0]:0.1f}–{util.zbins[0, 1]:0.1f} km',
            f'{util.zbins[1, 0]:0.1f}–{util.zbins[1, 1]:0.1f} km',
        ],
        loc='lower left')


def vertical_gradient_seasonal(
    ds, ax,
    co2_var_list=['CO2', 'CO2_LND', 'CO2_FFF', 'CO2_OCN'],
    window=30,
    linewidth=3,
):
    labels = []
    handles = []
    for v in co2_var_list:
        x, y = util.antyear_daily(ds.time, ds[v])
        h = ax.plot(x, y, color=co2_colors[v], linestyle='-', linewidth=linewidth)

        handles.append(h[0])
        labels.append(co2_names[v])

    ax.axhline(0, color='k', linewidth=1.)
    ax.legend(handles, labels, loc='lower left')

    ax.set_xlim((0, 365))
    ax.set_xticks(bomday)
    ax.set_xticklabels([f'        {m}' for m in monlabs_ant]+[''])
    ax.set_ylabel('$\Delta_{ θ}$CO$_2$ [ppm]')
    return handles, labels

def obs_theta_gradient(df, ax,
                       theta_bins=None,
                       sensor_mean=True,
                       constituent='co2',
                       just_the_median=False,
                       median_color='k',
                       median_alpha=0.75,
                       median_size=8,
                      ):

    constituent = constituent.lower()
    field = 'gradient_mean' if sensor_mean else constituent

    from scipy.optimize import curve_fit
    def harm(t, mu, a1, phi1, a2, phi2):
        """A harmonic"""
        return (mu + a1 * np.cos(1. * 2. * np.pi * t + phi1) +
                a2 * np.cos(2. * 2. * np.pi * t + phi2))

    marker_spec = marker_spec_campaigns()
    x = []
    t = []
    #for n, ndx in vg.groupby(vg.flight_id).groups.items():

    for campaign_id in df.campaign_id.unique():
        dfc = df.loc[df.campaign_id == campaign_id]

        color = marker_spec[campaign_id]['color']
        marker = marker_spec[campaign_id]['marker']

        for flight_id in dfc.flight_id.unique():
            dfi = dfc.loc[dfc.flight_id == flight_id]

            doy, gradient = util.antyear_daily(
                dfi.doy,
                dfi[field].values,
            )
            if np.isnan(gradient).all():
                continue

            k = ~np.isnan(gradient)
            if not just_the_median:
                ax.plot(doy-0.5, gradient,
                        marker='.',
                        linestyle='None',
                        markerfacecolor=color,
                        color=color,
                        alpha=0.35,
                        markersize=6,
                       )

                ax.errorbar(np.mean(doy[k]), np.median(gradient[k]), yerr=np.std(gradient[k]),
                            color=color,
                            marker=marker,
                            markerfacecolor=color,
                            markersize=6,
                           )

                x.append(np.median(gradient[k]))
                t.append(np.mean(doy[k])/365.)
            else:
                ax.errorbar(np.mean(doy[k]), np.median(gradient[k]), yerr=np.std(gradient[k]),
                            color=median_color,
                            marker=marker,
                            markerfacecolor=median_color,
                            alpha=median_alpha,
                            markersize=median_size,
                           )
    if just_the_median:
        return [
            Line2D(
                [0], [0], label=c, linestyle='None',
                marker=marker_spec[c]['marker'], color='k',
            )
            for c in df.campaign_id.unique()
        ]

    else:
        legend_elements = [Line2D([0], [0], label=c,
                                  linestyle='None',
                                  marker=marker_spec[c]['marker'],
                                  color=marker_spec[c]['color'])
                            for c in df.campaign_id.unique()]

    abcd, pcov = curve_fit(harm, np.array(t), np.array(x))
    xhat, yhat = np.linspace(0, 365, 100), harm(np.linspace(0, 365, 100)/365.25, *abcd)
    ax.plot(
        xhat, yhat, '-',
        color='k',
    )

    ax.axhline(0, linewidth=0.5, color='k')
    ax.set_xlim((-10, 375))
    ax.set_xticks(bomday)
    ax.set_xticklabels([f'        {m}' for m in monlabs_ant]+[''])

    ax.legend(handles=legend_elements, ncol=2, fontsize=8, frameon=False);
    if constituent == 'co2':
        constituent_str = 'CO$_2$'
    elif constituent == 'ch4':
        constituent_str = 'CH$_4$'

    ax.set_ylabel(f'$\Delta_{{ θ}}${constituent_str} [ppb]')

    if theta_bins is not None:
        bin_def = theta_bin_def(theta_bins)
        ax.set_title(f'Aircraft obs: {bin_def} {constituent_str} diff')

    return xhat, yhat


def theta_bin_def(theta_bins):
    theta_str = []
    for tbin in theta_bins:
        if np.Inf in tbin:
            theta_str.append(f'(>{tbin[0]:0.0f}K)')
        elif -np.Inf in tbin:
            theta_str.append(f'(<{tbin[1]:0.0f}K)')
        else:
            tcenter = np.mean(tbin)
            theta_str.append(f'({tbin[0]:0.0f}-{tbin[1]:0.0f}K)')

    return ' – '.join(theta_str[::-1])

# Overview

This book documents the analysis presented in {cite:t}`Long-Stephens-etal-2021`.

## Abstract

The Southern Ocean plays an important role in determining atmospheric CO2, yet estimates of air-sea CO2 flux for the region diverge widely. We constrain Southern Ocean air-sea CO2 exchange by relating fluxes to horizontal and vertical CO2 gradients in atmospheric transport models and then apply atmospheric observations of these gradients to estimate fluxes. Aircraft-based measurements of the vertical atmospheric CO2 gradient provide robust flux constraints. We find an annual-mean flux of –0.55±0.23 Pg C yr$^{–1}$ (net uptake) south of 45°S during 2009–2018. This is consistent with the mean of atmospheric-inversion estimates and surface-ocean pCO2-based products, but our data indicate stronger annual-mean uptake than suggested by recent interpretations of profiling-float observations.


## CO<sub>2</sub> Observations

```{figure} figures/Fig-1-co2-aircraft-surface-obs.png
:figwidth: 800px
:name: aircraft-surface-obs

**Observed patterns in atmospheric CO2 over the Southern Ocean.** Upper panels: Cross-sections observed by aircraft during (A) ORCAS, Jan–Feb 2016 and (B) ATom-1, Aug 2016. Colors show the observed CO2 dry air mole fraction relative to the average observed within the 295–305 K potential temperature range south of 45°S on each campaign; contour lines show the observed potential temperature. See Figs S1–S2 for flight-tracks and cross-section plots for all campaigns and Figures S3–S4 for simulated fields. Lower panels: Compilation of mean CO2 observed at surface monitoring stations minus the NOAA in situ record at the South Pole Observatory (SPO) over 1999–2019 for (C) summer (DJF) and (D) winter (JJA). The black line is a spline fit provided simply as a visual guide. Blue shading denotes the latitude band in which we designate “Southern Ocean stations.” See Table S1 and Figure S5 for station locations and temporal coverage. Supplementary Material includes additional methodological details.
```

## Gradient metrics

```{figure} figures/Fig-2-metrics-seasonal-cycle.png
:figwidth: 800px
:name: metrics-seasonal-cycle

**Seasonal evolution of atmospheric CO2 over the Southern Ocean.** (A) Vertical profiles of CO2 observations collected by aircraft south of 45°S, binned on 5 K potential temperature (θ) bins and averaged by season (whiskers show standard deviation; Fig S6 shows model comparison). (B) The vertical gradient (∆θCO2) in CO2 measured from aircraft south of 45°S. Small points show ∆θCO2 for individual profiles; larger points show the median and standard deviation (whiskers) for each flight. The black line shows a two-harmonic fit to the flight-median points. (C) Monthly climatology (1999–2019) of the latitudinal gradient in CO2 measured by surface stations (Fig. 1); the black line shows the station mean metric (∆yCO2). Separate laboratory records at SYO and PSA have been averaged. The seasonal evolution of (D) ∆θCO2 and (E) ∆yCO2 simulated in a collection of atmospheric inverse models (Table S3). The points show the median across the models, whiskers show the standard deviation; the colors correspond to the “total” CO2 (black), and CO2 tracers responsive to only ocean (blue), land (green), and fossil (red) surface fluxes. Note that the y-axis bounds differ by panel.
```

## Emergent constraint

```{figure} figures/Fig-3-emergent-constraint-ocean_constraint.png
:figwidth: 800px
:name: emergent-constraint-ocean_constraint

**Emergent constraints on air-sea fluxes south of 45°S.** Upper panels: 90-day-mean air-sea fluxes south of 45°S versus ∆θCO2ocn from model simulations (see SM) replicating aircraft observations collected during (A) Dec-Feb and (B) Mar-Nov. Colored vertical lines show an observed value of ∆θCO2 (ORCAS during Jan in panel “A” and ATom-1 in panel “B”) corrected for land and fossil-fuel influence, with shading indicating both analytical uncertainty and model spread in the correction (see SM); colored points highlight the model samples from these particular campaigns, while gray points show data from other campaigns in the (A) Dec–Feb or (B) Mar–Nov timeframe. Figures S10 and S11 show similar plots for each individual aircraft campaign. Lower panels: Seasonal-mean surface fluxes versus ∆yCO2ocn computed from models for (C) summer (DJF) and (D) winter (JJA) over 1999–2019. Points correspond to individual models; whiskers denote the standard deviation of interannual variability. Light blue vertical lines show the observed ∆yCO2 corrected for land and fossil fuel influence; shading shows analytical uncertainty and model spread in the correction (see SM; Fig. S12A, B show ∆yCO2 time series). The sign convention for fluxes is positive upward. Diagonal lines, where significant, show the best-fit line to all data points shown: inset text shows an estimate of the slope with standard error (SM); goodness-of-fit statistics are also shown. Table S3 provides detailed information on the model products, defining the acronyms used in the legend. Note that the axis bounds differ by panel. See Figure S16 for a version of this plot based on total CO2.
```

## Flux etimates

```{figure} figures/Fig-4-fluxes-ocean_constraint.png
:figwidth: 800px
:name: fluxes-ocean_constraint

Observationally-based estimates of Southern Ocean air-sea fluxes. (A) The seasonal cycle of air-sea CO2 flux south of 45°S estimated from aircraft campaigns (black points, labels), plotted at the center of the 90-day window for which the emergent flux constraint was calibrated; whiskers show the standard deviation derived from propagating analytical and statistical uncertainties; the black line shows a two-harmonic fit used to estimate the annual mean flux. The colored lines give the seasonal cycle from atmospheric inversion systems as well as the neural-network extrapolation (22) of the Surface Ocean CO2 Atlas (“SOCAT”) pCO2 observations (31) and profiling-float observations from the Southern Ocean Carbon and Climate Observations and Modeling (SOCCOM) project (32). Fluxes are averaged over 2009–2018, except for the three neural-network based flux estimates (27) incorporating SOCCOM observations, which are averaged over 2015–2017. (B) Annual mean flux estimated in this study (leftmost bar) including uncertainty (whisker), along with the mean and standard deviation (whiskers) across the inversion systems shown in panel “A” as well as the surface-ocean pCO2-based methods; averaging time-periods are noted in the axis labels (both SOCAT flux estimates were derived using neural-network training over the full observational period). The uncertainty estimate on the SOCAT and SOCCOM fluxes is approximated from ref. 10, who report ±0.15 Pg C yr-1 as the “method uncertainty” associated with the neural-network-based flux estimates for the whole Southern Ocean (south of 44°S). Note that while s99oc_v2020 and s99c_SOCAT+SOCCOM_v2020 are global inversions, their ocean fluxes are prescribed, not optimized using atmospheric observations (see SM); similarly, the CAMS(v20r1) ocean fluxes remain close to its SOCAT pCO2-based prior. 
```

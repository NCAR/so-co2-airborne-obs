# Overview

This book documents the analysis presented in {cite:t}`Long-Stephens-etal-2021`.
The aim is to enable the entire calculation to be reproduced from data that is available via websites and public repositories.
We include [instructions for pre-staging](./_prestage-obs-instructions.md) the input data, which includes a combinations of observations and models.
We provide documentation of the analysis code and demonstrate how each figure was created.


## Summary of the science

The Southern Ocean plays an important role in determining atmospheric CO<sub>2</sub>, yet estimates of air-sea CO<sub>2</sub> flux for the region diverge widely.
Here we present an analysis in which we estimate Southern Ocean air-sea CO<sub>2</sub> fluxes using aircraft observations of atmospheric CO<sub>2</sub>.
Using a collection of transport models, we "calibrate" a relationship between the flux and the lower-troposphere vertical CO<sub>2</sub> gradient over the Southern Ocean.
Using data from aircraft campaigns conducted over 2009–2018, we find a Southern Ocean flux of –0.55±0.23 Pg C yr<sup>–1</sup> (net uptake) south of 45°S.
We also examine the use of surface-based atmospheric CO<sub>2</sub> measurements.
The horizontal gradients captured by these observations do not provide as robust constraints on the fluxes as the vertical gradients accessible from aircraft.
However, the inferences from surface observations are consistent with the aircraft-based flux estimates, and suggest a trend of increasing Southern Ocean uptake since 2005.

Most previous estimates of air-sea flux on seasonal to interannual timescales are based on either observations of surface-ocean pCO<sub>2</sub> or the CO<sub>2</sub> mixing ratio in the lower atmosphere.
Flux estimates obtained from surface-ocean pCO<sub>2</sub> observations require extrapolation of these data in space and time, as available observations are relatively sparse {cite:p}`Takahashi2009-mx,Landschutzer2016-wg`; fluxes are computed using a parameterization for gas exchange that includes an uncertain, wind-speed dependent piston velocity.
Atmospheric inversion models are used to estimate fluxes from CO<sub>2</sub> mixing ratio observations.
Inversion models are based on atmospheric transport simulations; they determine fluxes that yield simulated CO<sub>2</sub> concentrations that are maximally consistent with those observed [e.g., {cite:author}`Peters2005-ta`, {cite:year}`Peters2005-ta`].
Inversion models, however, have failed to converge on consistent Southern Ocean fluxes estimates owing to relatively sparse observations in the region, inaccuracies in the simulated transport, and requirements to meet tighter constraints elsewhere in the world, where signals are stronger and measurements less sparse {cite:p}`Peylin2013-bf,Peters2005-ta,Enting2000-ft,Denning1995-lc,Schuh2019-ds,Basu2018-mf`.
Furthermore, atmospheric inversion models rely on uncertain “prior” flux estimates---generally derived from surface-ocean pCO<sub>2</sub>.
This requirement for priors means the inversion models and surface ocean pCO<sub>2</sub> observations are not entirely independent.

Our approach to estimating fluxes is novel and does not directly depend on these previous methodologies.
We exploit different realizations of atmospheric transport across a collection of atmospheric inversion models, and are therefore not subject to the biases associated with any one particular model.
Our analysis depends on atmospheric CO<sub>2</sub> observations collected at by aircraft, and we also make use of CO<sub>2</sub> data from surface station.
We leverage a collection of atmospheric inversions models that simulate time-varying, three-dimensional CO<sub>2</sub> fields sampled to replicate these observations.
From the models, we generate "emergent constraints," relating fluxes to horizontal and vertical gradient in atmospheric CO<sub>2</sub>.
This constraint enables using observations of these gradient metrics to estimate fluxes.


## Outline

The calculation has the following components.

1. {ref}`obs`
1. {ref}`gradients`
1. {ref}`emergent-constraint`
1. {ref}`fluxes`

(obs)=
### Structure of variability in atmospheric carbon dioxide

We examine the structure of atmospheric CO<sub>2</sub> variability over the Southern Ocean, demonstrating surface drawdown in the lower troposphere during summer and relatively homogenous distributions during winter.
{numref}`aircraft-surface-obs` shows these patterns for two selected airborne transects and a collection of surface monitoring stations.
The code to produce this figure is [here](./obs-main.ipynb).

```{figure} figures/Fig-1-CO2-aircraft-surface-obs.png
:figwidth: 600px
:name: aircraft-surface-obs

**Observed patterns in atmospheric CO<sub>2</sub> over the Southern Ocean.** Observed patterns in atmospheric CO<sub>2</sub> over the Southern Ocean. Upper panels: Cross-sections observed by aircraft during (A) ORCAS, Jan–Feb 2016 and (B) ATom-1, Aug 2016. Colors show the observed CO<sub>2</sub> dry air mole fraction relative to the average observed within the 295–305 K potential temperature range south of 45°S on each campaign; contour lines show the observed potential temperature.
See Figures S1–S2 for flight-tracks and cross-section plots for all campaigns and Figures S3–S4 for simulated fields. Lower panels: Compilation of mean CO<sub>2</sub> observed at surface monitoring stations minus the NOAA in situ record at the South Pole Observatory (SPO) over 1999–2019 for (C) summer (DJF) and (D) winter (JJA). The black line is a spline fit provided simply as a visual guide. Blue shading denotes the latitude band in which we designate “Southern Ocean stations.” See Table S1 and Figure S5 for station locations and temporal coverage. Supplementary Material includes additional methodological details.
```

(gradients)=
###  Gradient metrics to isolate ocean influence

To isolate CO<sub>2</sub> gradients driven by Southern Ocean fluxes, we examine CO<sub>2</sub> anomalies relative to a local reference, using potential temperature ($\theta$) to delineate boundaries in the vertical.
We define metrics quantifying the vertical and horizontal CO<sub>2</sub> gradients
- $\Delta_{\theta}\ce{CO}_2$ is the difference between the median value of CO<sub>2</sub> observed south of 45°S where $\theta$ < 280 K and that in the mid- to upper-troposphere, $\pu{295 K} < \theta < \pu{305 K}$.
- $\Delta_{y}\ce{CO}_2$ is the difference between CO<sub>2</sub> averaged across stations in the core latitudes of summertime CO<sub>2</sub>-drawdown (shaded region, {numref}`aircraft-surface-obs`C, D) and that at the South Pole Observatory (SPO).

The atmospheric inversion models explicitly simulate CO<sub>2</sub> tracers responsive only to ocean $(\ce{CO}_2^{ocn})$, land $(\ce{CO}_2^{lnd})$ and fossil fuel $(\ce{CO}_2^{ff})$ fluxes and subject to identical transport fields.
We take advantage of these tracers to demonstrate that the gradient metrics ($\Delta_{\theta}\ce{CO}_2$, $\Delta_{y}\ce{CO}_2$) are primarily responsive to local ocean influence.
{numref}`metrics-seasonal-cycle` illustrates the seasonal variation of the gradient metrics in the observations and the models.
The code to produce this figure is [here](./gradients-main.ipynb).


```{figure} figures/Fig-2-metrics-seasonal-cycle.png
:figwidth: 600px
:name: metrics-seasonal-cycle

**Seasonal evolution of atmospheric CO<sub>2</sub> over the Southern Ocean.** (A) Vertical profiles of CO<sub>2</sub> observations collected by aircraft south of 45°S, binned on 5 K potential temperature (θ) bins and averaged by season (whiskers show standard deviation). (B) The vertical gradient ($\Delta_{\theta}\ce{CO}_2$) in CO<sub>2</sub> measured from aircraft south of 45°S. Small points show $\Delta_{\theta}\ce{CO}_2$ for individual profiles; larger points show the median and standard deviation (whiskers) for each flight. The black line shows a two-harmonic fit to the flight-median points. (C) Monthly climatology (1999–2019) of the latitudinal gradient in CO<sub>2</sub> measured by surface stations ({numref}`aircraft-surface-obs`); the black line shows the station mean metric ($\Delta_{y}\ce{CO}_2$). Separate laboratory records at SYO and PSA have been averaged. The seasonal evolution of (D) $\Delta_{\theta}\ce{CO}_2$ and (E) $\Delta_{y}\ce{CO}_2$ simulated in a collection of atmospheric inverse models (Table S3). The points show the median across the models, whiskers show the standard deviation; the colors correspond to the “total” CO<sub>2</sub> (black), and CO<sub>2</sub> tracers responsive to only ocean (blue), land (green), and fossil (red) surface fluxes. Note that the <i>y</i>-axis bounds differ by panel.
```

(emergent-constraint)=
### Construction of emergent constraints

To develop quantitative flux estimates, we relate simulated $\Delta_{\theta}\ce{CO}_2^{ocn}$ and $\Delta_{y}\ce{CO}_2^{ocn}$ to regionally-integrated, temporally-averaged air-sea flux in each modeling system.
{numref}`emergent-constraint-ocean_constraint` shows the relationships obtained for seasonal aggregations of the data.
The relationships are all significant, except for surface observations $(\Delta_{y}\ce{CO}_2^{ocn})$ during winter (JJA; {numref}`emergent-constraint-ocean_constraint`D).
The code to produce this figure is [here](./emergent-constraint.ipynb).

```{figure} figures/Fig-3-emergent-constraint-ocean_constraint.png
:figwidth: 600px
:name: emergent-constraint-ocean_constraint

**Emergent constraints on air-sea fluxes south of 45°S.** Upper panels: 90-day-mean air-sea fluxes south of 45°S versus ∆θCO<sub>2</sub>ocn from model simulations (see SM) replicating aircraft observations collected during (A) Dec-Feb and (B) Mar-Nov. Colored vertical lines show an observed value of $\Delta_{\theta}\ce{CO}_2$ (ORCAS during Jan in panel “A” and ATom-1 in panel “B”) corrected for land and fossil-fuel influence, with shading indicating both analytical uncertainty and model spread in the correction; colored points highlight the model samples from these particular campaigns, while gray points show data from other campaigns in the (A) Dec–Feb or (B) Mar–Nov timeframe. Figures S10 and S11 show similar plots for each individual aircraft campaign. Lower panels: Seasonal-mean surface fluxes versus ∆yCO<sub>2</sub>ocn computed from models for (C) summer (DJF) and (D) winter (JJA) over 1999–2019. Points correspond to individual models; whiskers denote the standard deviation of interannual variability. Light blue vertical lines show the observed ∆yCO<sub>2</sub> corrected for land and fossil fuel influence; shading shows analytical uncertainty and model spread in the correction (see SM; Fig. S12A, B show ∆yCO<sub>2</sub> time series). The sign convention for fluxes is positive upward. Diagonal lines, where significant, show the best-fit line to all data points shown: inset text shows an estimate of the slope with standard error (SM); goodness-of-fit statistics are also shown. Table S3 provides detailed information on the model products, defining the acronyms used in the legend. Note that the axis bounds differ by panel. See Figure S16 for a version of this plot based on total CO<sub>2</sub>.
```

(fluxes)=
### Seasonal flux estimates

Vertical lines on {numref}`emergent-constraint-ocean_constraint` show representative observations of each gradient metric corrected for land and fossil fuel contributions; the intersection of these lines with the flux-gradient fit provides a quantitative flux estimate.
Applying this emergent constraint for each aircraft campaign yields 10 flux estimates spread over 7 months of the year; these data suggest that the Southern Ocean is a strong sink for CO<sub>2</sub> in austral summer, with fluxes that are near-neutral during winter.
{numref}`fluxes-ocean_constraint` shows the flux estimates we obtain from the aircraft observations.
The code to produce this figure is [here](./fluxes.ipynb).

```{figure} figures/Fig-4-fluxes-ocean_constraint.png
:figwidth: 600px
:name: fluxes-ocean_constraint

Observationally-based estimates of Southern Ocean air-sea fluxes. (A) The seasonal cycle of air-sea CO<sub>2</sub> flux south of 45°S estimated from aircraft campaigns (black points, labels), plotted at the center of the 90-day window for which the emergent flux constraint was calibrated; whiskers show the standard deviation derived from propagating analytical and statistical uncertainties; the black line shows a two-harmonic fit used to estimate the annual mean flux. The colored lines give the seasonal cycle from atmospheric inversion systems as well as the neural-network extrapolation {cite:p}`Landschutzer2016-wg` of the Surface Ocean CO<sub>2</sub> Atlas (“SOCAT”) pCO<sub>2</sub> observations {cite:p}`Bakker2016-tt` and profiling-float observations from the Southern Ocean Carbon and Climate Observations and Modeling (SOCCOM) project {cite:p}`Johnson2017-al`. Fluxes are averaged over 2009–2018, except for the three neural-network based flux estimates {cite:p}`Landschutzer2019-mr` incorporating SOCCOM observations, which are averaged over 2015–2017. (B) Annual mean flux estimated in this study (leftmost bar) including uncertainty (whisker), along with the mean and standard deviation (whiskers) across the inversion systems shown in panel “A” as well as the surface-ocean pCO<sub>2</sub>-based methods; averaging time-periods are noted in the axis labels (both SOCAT flux estimates were derived using neural-network training over the full observational period). The uncertainty estimate on the SOCAT and SOCCOM fluxes is approximated from {cite:t}`Bushinsky2019-ha`, who report ±0.15 Pg C yr<sup>-1</sup> as the “method uncertainty” associated with the neural-network-based flux estimates for the whole Southern Ocean (south of 44°S). Note that while s99oc_v2020 and s99c_SOCAT+SOCCOM_v2020 are global inversions, their ocean fluxes are prescribed, not optimized using atmospheric observations; similarly, the CAMS(v20r1) ocean fluxes remain close to its SOCAT pCO<sub>2</sub>-based prior.
```

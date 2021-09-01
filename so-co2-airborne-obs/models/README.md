
# Southern Ocean Air-Sea Carbon Fluxes from Aircraft Observations: Modeling Datasets

This directory contains data from models used in the analysis presented by:
Long, M. C., B. B. Stephens, K. McKain, C. Sweeney, R. F. Keeling, Eric A. Kort, et al. Strong
Southern Ocean Carbon Uptake Evident in Airborne Observations, in review, Science.

The data are organized in directories for each model; each of these contains a subdirectory 
containing surface CO2 fluxes ("fluxes") and another subdirectory containing simulated 
observations data ("simulated-obs"). 

The "simulated-obs" files are either in NOAA ObsPack format or report values matched 
to a particular NOAA ObsPack, but in a different format.

For more information on ObsPack:

Masarie, K. A., Peters, W., Jacobson, A. R., and Tans, P. P.: 
ObsPack: a framework for the preparation, delivery, and attribution of atmospheric 
greenhouse gas measurements, Earth Syst. Sci. Data, 6, 375–384, 
https://doi.org/10.5194/essd-6-375-2014, 2014.

https://gml.noaa.gov/ccgg/obspack/

##  Model specific information:

### CAMS (CAMSv20r1)
Contains modified Copernicus Atmosphere Monitoring Service Information 2021. Neither the European Commission nor ECMWF is responsible for any use that may be made of the information it contains.

Original data gateway:
https://ads.atmosphere.copernicus.eu/cdsapp#!/dataset/cams-global-greenhouse-gas-inversion?tab=form

Simulated observations are in flat text files with ObsPack IDs for each value, corresponding to 
obspack_co2_1_GLOBALVIEWplus_v6.0_2020-09-11

### CarboScope (s99oc_v2020, s99oc_ADJocI40S_v2020, and  s99oc_SOCCOM_v2020)
Documentation: http://www.bgc-jena.mpg.de/CarboScope/s/tech_report6.pdf
File naming convention described with subsequent updates here:
http://www.bgc-jena.mpg.de/~christian.roedenbeck/INVERSION/HOWTO/StationCodes.html
Simulated observations are in flat text files corresponding to obspack_co2_1_GLOBALVIEWplus_v4.2.2_2019-06-05 and obspack_co2_1_ATom_v4.0_2020-04-06

### CarbonTracker (CT2017 and CT2019B)
https://gml.noaa.gov/ccgg/carbontracker/

The DOI for CT2017 is http://dx.doi.org/10.25925/V3K6-5168
The DOI for CT2019B is http://dx.doi.org/10.25925/20201008

Simulated observations in ObsPack format. For CT2017 these correspond to: obspack_co2_1_CARBONTRACKER_CT2017_2018-05-02, which uses obspack_co2_1_ORCAS_v2.0_2017-04-05, obspack_co2_1_GLOBALVIEWplus_v3.1_2017-10-18, and obspack_co2_1_NRT_v4.0_2017-09-08.

For CT2019B these correspond to: obspack_co2_1_CARBONTRACKER_CT2019B_2020-11-03, which uses obspack_co2_1_GLOBALVIEWplus_v5.0_2019-08-12

CT2017 includes simulated observations for ATom-2 (Jan/Feb 2017) but only fluxes for 2000-2016 so not included in the paper. CT2017 also missing ATom-3 and ATom-4.

### CarbonTracker Europe (CTE2018 and CTE2020)
https://www.carbontracker.eu/

Simulated observations in ObsPack format. For CTE2018 these correspond to: obspack_co2_1_GLOBALVIEWplus_v5.0_2019-08-12 and obspack_co2_1_ATom_v4.0_2020-04-06

For CTE2020 these correspond to: 
obspack_co2_1_GLOBALVIEWplus_v5.0_2019-08-12 
CTE2018 only covers 2000-2017 so missing ATom-4

At the time output for CTE2018 was generated, we did not include the SIO station records. For the SIO CO2 Program record at SPO and the SIO O2 Program records at SPO, PSA, and CGO, we substitute CTE2018 output for NOAA records at the same stations.

### MIROC4-ACTM (MIROC)
Simulated observations as netcdf files similar to ObsPack format but with reduced metadata. These correspond to: obspack_co2_1_GLOBALVIEWplus_v5.0_2019-08-12 and obspack_co2_1_ATom_v4.0_2020-04-06

### TM5-Flux (TM5-Flux-m0f, TM5-Flux-mmf, TM5-Flux-mrf, and TM5-Flux-mwf)
Simulated observations in ObsPack format corresponding to: obspack_co2_1_GLOBALVIEWplus_v5.0_2019-08-12 
Data extend through the end of 2017, so missing ATom-4.

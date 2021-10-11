# Obtaining observational data

Unfortunately, it is not possible to automate staging of the observational data used in this calculation. Here we provide instructions, however, for manually accessing these data products, which are all publicly available, but in several instances, require registration to access.

[_config_calc.yml](./_config_calc.yml) defines a variable `project_tmpdir_obs`, which specifies the directory in which to store the data referenced here.



## Aircraft observations (complete merge products)

For each campaign, download the following files into `${project_tmpdir_obs}/aircraft-merge-products` subdirectory.

### HIPPO

- HIPPO 10-sec merge file HIPPO_all_missions_merge_10s_20121129.tbl (Version 1.0) from https://doi.org/10.3334/CDIAC/HIPPO_010
- HIPPO Medusa merge file HIPPO_medusa_flasks_merge_insitu_20121129.tbl (Version 1.0) from https://doi.org/10.3334/CDIAC/HIPPO_014
- HIPPO PFP merge file HIPPO_noaa_flask_allparams_merge_insitu_20121129.tbl (Version 1.0) from https://doi.org/10.3334/CDIAC/HIPPO_013

### ORCAS

- ORCAS merge products ORCASall.merge10.tbl and ORCASall.mergeMED.tbl (Version 1.0) from https://doi.org/10.5065/D6SB445X

### ATom

- ATom merge products (Version 2.0): 
MER10_DC8_ATom-1.nc, MER10_DC8_ATom-2.nc, MER10_DC8_ATom-3.nc, MER10_DC8_ATom-4.nc, MER-MED_DC8_ATom-1.nc, MER-MED_DC8_ATom-2.nc, 
MER-MED_DC8_ATom-3.nc, MER-MED_DC8_ATom-4.nc, MER-PFP_DC8_ATom-2.nc, MER-PFP_DC8_ATom-3.nc, and MER-PFP_DC8_ATom-4.nc 
from https://doi.org/10.3334/ORNLDAAC/1925


## ObsPack data files

Download netcdf ObsPack files from:
https://gml.noaa.gov/ccgg/obspack
for ObsPack versions:
- obspack_co2_1_GLOBALVIEWplus_v4.2.2_2019-06-05 (only the hip, orc, and tom aircraft files are needed)
- obspack_co2_1_GLOBALVIEWplus_v5.0_2019-08-12 (only the hip, orc, and tom aircraft files are needed)
- obspack_co2_1_GLOBALVIEWplus_v6.0_2020-09-11 (only the Southern Ocean records listed in Table S1 and the hip, orc, and tom aircraft files are needed)
- obspack_co2_1_ATom_v4.0_2020-04-06
- obspack_co2_1_CARBONTRACKER_CT2017_2018-05-02 (only the hip, orc, and tom aircraft files are needed)
- obspack_co2_1_GLOBALVIEWplus_v7.0_2021-08-18 (only the LSCE AMS record and the NOAA in situ Gould record needed)

Expand the gzipped tar files into `${project_tmpdir_obs}` for each obspack; preserve the `${obspack}/data/nc` subdirectories.

## WDCGG

Download the following files into a `${project_tmpdir_obs}/WDCGG` subdirectory:

- MQA hourly netcdf file from http://doi.org/10.50849/WDCGG_0016-5015-1001-01-01-9999
- CGO hourly netcdf file from http://doi.org/10.50849/WDCGG_0016-5011-1001-01-01-9999
- CPT 222Rn, hourly netcdf file from https://gaw.kishou.go.jp/search/file/0007-1009-6002-01-01-9999

Expanding the gzipped tar files and preserve the /data/nc/* subdirectories.


## SIO O2

Download into `${project_tmpdir_obs}/sio` subdirectory, csv files for CGO, PSA, and SPO from https://scrippso2.ucsd.edu/cosub2sub-data.html


## SIO CO2

Download into the `${project_tmpdir_obs}/sio` subdirectory, csv file https://scrippsco2.ucsd.edu/assets/data/atmospheric/stations/merged_in_situ_and_flask/monthly/monthly_merge_co2_spo.csv


## NCAR in situ Gould record

Download into `${project_tmpdir_obs}/ncar-lmg` subdirectory gzipped text file https://archive.eol.ucar.edu/homes/stephens/GO2/lmg_all_results.txt.gz
and unzip


## NIWA

Download into `${project_tmpdir_obs}/niwa` subdirectory text file https://niwa.co.nz/static/tropac/co2/bhd/archive/co2_bhd_surface-insitu_57_1978-2019_hourly.txt


## Minimal file listing when finished (extra files can be kept if space allows):

Once your done, `${project_tmpdir_obs}` should look like this:
```bash
.
├── aircraft-merge-products
│   ├── HIPPO_all_missions_merge_10s_20121129.tbl
│   ├── HIPPO_medusa_flasks_merge_insitu_20121129.tbl
│   ├── HIPPO_noaa_flask_allparams_merge_insitu_20121129.tbl
│   ├── MER10_DC8_ATom-1.nc
│   ├── MER10_DC8_ATom-2.nc
│   ├── MER10_DC8_ATom-3.nc
│   ├── MER10_DC8_ATom-4.nc
│   ├── MER-MED_DC8_ATom-1.nc
│   ├── MER-MED_DC8_ATom-2.nc
│   ├── MER-MED_DC8_ATom-3.nc
│   ├── MER-MED_DC8_ATom-4.nc
│   ├── MER-PFP_DC8_ATom-1.nc
│   ├── MER-PFP_DC8_ATom-2.nc
│   ├── MER-PFP_DC8_ATom-3.nc
│   ├── MER-PFP_DC8_ATom-4.nc
│   ├── ORCASall.merge10.tbl
│   └── ORCASall.mergeMED.tbl
├── atom_xsect_filt_datetime.txt
├── hippo_xsect_filt_datetime.txt
├── ncar-lmg
│   └── lmg_all_results.txt
├── niwa
│   └── co2_bhd_surface-insitu_57_1978-2019_hourly.txt
├── obspack_co2_1_ATom_v4.0_2020-04-06
│   └── data
│       └── nc
│           └── co2_tom_aircraft-insitu_1_allvalid.nc
├── obspack_co2_1_CARBONTRACKER_CT2017_2018-05-02
│   └── data
│       └── nc
│           ├── co2_hip_aircraft-insitu_59_allvalid.nc
│           ├── co2_orc_aircraft-insitu_3_allvalid-merge10.nc
│           └── co2_tom_aircraft-insitu_1_allvalid.nc
├── obspack_co2_1_GLOBALVIEWplus_v4.2.2_2019-06-05
│   └── data
│       └── nc
│           ├── co2_hip_aircraft-insitu_59_allvalid.nc
│           ├── co2_orc_aircraft-insitu_3_allvalid-merge10.nc
│           └── co2_tom_aircraft-insitu_1_allvalid.nc
├── obspack_co2_1_GLOBALVIEWplus_v5.0_2019-08-12
│   └── data
│       └── nc
│           ├── co2_hip_aircraft-insitu_59_allvalid.nc
│           ├── co2_orc_aircraft-insitu_3_allvalid-merge10.nc
│           └── co2_tom_aircraft-insitu_1_allvalid.nc
├── obspack_co2_1_GLOBALVIEWplus_v6.0_2020-09-11
│   └── data
│       └── nc
│           ├── co2_ams_surface-insitu_11_representative.nc
│           ├── co2_cgo_surface-flask_1_representative.nc
│           ├── co2_cgo_surface-flask_2_representative.nc
│           ├── co2_cpt_surface-insitu_36_marine.nc
│           ├── co2_crz_surface-flask_1_representative.nc
│           ├── co2_cya_surface-flask_2_representative.nc
│           ├── co2_drp_shipboard-flask_1_representative.nc
│           ├── co2_hba_surface-flask_1_representative.nc
│           ├── co2_hip_aircraft-insitu_59_allvalid.nc
│           ├── co2_maa_surface-flask_2_representative.nc
│           ├── co2_orc_aircraft-insitu_3_allvalid-merge10.nc
│           ├── co2_psa_surface-flask_1_representative.nc
│           ├── co2_spo_surface-flask_1_representative.nc
│           ├── co2_spo_surface-flask_2_representative.nc
│           ├── co2_spo_surface-insitu_1_allvalid.nc
│           ├── co2_syo_surface-flask_1_representative.nc
│           ├── co2_syo_surface-insitu_8_allvalid.nc
│           └── co2_tom_aircraft-insitu_1_allvalid.nc
├── obspack_co2_1_GLOBALVIEWplus_v7.0_2021-08-18
│   └── data
│       └── nc
│           ├── co2_ams_surface-insitu_11_allvalid.nc
│           └── co2_gould_shipboard-insitu_1_allvalid.nc
├── orcas_xsect_filt_datetime.txt
├── sio
│   ├── cgocav.csv
│   ├── monthly_merge_co2_spo.csv
│   ├── psacav.csv
│   └── spocav.csv
└── WDCGG
    └── nc
        ├── 222rn
        │   └── hourly
        │       └── 222rn_cpt_surface-insitu_7_9999-9999_hourly.nc
        └── co2
            └── hourly
                ├── co2_cgo_surface-insitu_16_9999-9999_hourly.nc
                └── co2_mqa_surface-insitu_16_9999-9999_hourly.nc

28 directories, 60 files
```




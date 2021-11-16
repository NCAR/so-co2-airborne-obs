# Overview

This calculation relies on publicly available data. 
We apply preliminary data-reduction and/or filtering steps to the raw data assets and, where size constraints permit, cache these results locally in the [data](https://github.com/NCAR/so-co2-airborne-obs/tree/main/so-co2-airborne-obs/data) directory.


The file [_config_calc.yml](https://github.com/NCAR/so-co2-airborne-obs/blob/main/so-co2-airborne-obs/_config_calc.yml) includes several variables:

```yaml
project_tmpdir: # specify directory to store large intermediate data files
project_tmpdir_obs: # specify where to put observational data assets
model_data_dir_root: # root directory location for downloading DASH assets
```

The remainder of this section includes documentation on how to obtain the necessary data assets to reproduce the calculation from scratch.

```{note}
Since the repository includes the intermediate cached products from filtering and precompute steps, it should not be necessary to perform these steps---unless there is a desire to alter some feature of the processing or underlying data assets.
```

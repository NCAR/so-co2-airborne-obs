# Strong Southern Ocean Carbon Uptake Evident in Airborne Observations

Aircraft observations reveal Southern Ocean air-sea CO<sub>2</sub> exchange and indicate that the region acts as a strong carbon sink.

This repository contains the code necessary to reproduce the calculation presented in 
Long et al. (2021).

### Invoking the calculation

To reproduce the calculation:

1. Build conda Python and R environments for computations:
```bash
conda env create -f environment.yml
conda env create -f environment-R.yml
```
2. Set paths for input data and cache files by editing [_config_calc.yml](so-co2-airborne-obs/_config_calc.yml)
3. Run [_prestage-data.ipynb](so-co2-airborne-obs/_prestage-data.ipynb), which downloads model flux and concentration data.
4. Run [_precompute.ipynb](so-co2-airborne-obs/_precompute.ipynb), which performs some time-consuming computations and cache's the results in `project_tmpdir`.
5. Follow the steps in [_prestage-obs-instructions.md](so-co2-airborne-obs/_prestage-obs-instructions.md), to manually download atmospheric observation data.
6. Run all the notebooks in [_toc.yml](so-co2-airborne-obs/_toc.yml).


### Building the book

The JupyterBook rendition can be built with the following steps.

1. Clone this repository
2. Build environments as described in 1. above
3. (Optional) Edit the books source files located in the `so-co2-airborne-obs/` directory
4. Run `jupyter-book clean so-co2-airborne-obs/` to remove any existing builds
5. Run `jupyter-book build so-co2-airborne-obs/`

A fully-rendered HTML version of the book will be built in `so-co2-airborne-obs/_build/html/`.

## Credits

This project is created using the excellent open source [Jupyter Book project](https://jupyterbook.org/) and the [executablebooks/cookiecutter-jupyter-book template](https://github.com/executablebooks/cookiecutter-jupyter-book).

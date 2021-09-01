import os
import yaml
from . import dash_assets

path_to_here = os.path.dirname(os.path.realpath(__file__))

#project_tmpdir = f"{os.environ['TMPDIR']}/so-co2-airborne-obs/models/cache"
project_tmpdir = '/glade/p/eol/stephens/longcoll/cache'
cache_rootdir_local = f'{path_to_here}/data-cache'
model_data_dir_root = f"/glade/work/{os.environ['USER']}/so-co2-airborne-obs/model-data"

os.makedirs(project_tmpdir, exist_ok=True)
os.makedirs(cache_rootdir_local, exist_ok=True)
os.makedirs(model_data_dir_root, exist_ok=True)

model_data_dir = dash_assets.ensure_data(model_data_dir_root)

known_products = ['molefractions', 'fluxes', 'ObsPack']
preferred_time_units = 'days since 2000-01-01 00:00:00'

extended_domain_subset = dict(
    lat=slice(-90, -30)
)

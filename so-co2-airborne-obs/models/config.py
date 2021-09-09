import os
import yaml
from jinja2 import Template

path_to_here = os.path.dirname(os.path.realpath(__file__))


def get_config_dict():
    """return the configuration dictionary with environment variables replaced"""
    
    assert os.path.exists('_config_calc.yml'), (
        'Missing _config_calc.yml\n' +
        'set paths in _config_calc.yml\n' +
        'Example:\n' +
        'project_tmpdir: /glade/p/eol/stephens/longcoll/cache\n' +
        'model_data_dir_root: "/glade/work/{{env(USER)}}/so-co2-airborne-obs/model-data"\n'
    )
    
    with open('_config_calc.yml') as fid:
        config_dict_in = yaml.safe_load(fid)    
    
    required_keys = ['project_tmpdir', 'model_data_dir']
    for key in required_keys:
        assert key in config_dict_in, f'config missing {key}'
        
    config_dict = {}
    for key, value in config_dict_in.items():
        if isinstance(value, str):        
            t = Template(value)
            config_dict[key] = t.render(env=os.environ)
        else:
            config_dict[key] = value
    return config_dict    
    
    
# local cache path: files that are small enough to fit in the repo
cache_rootdir_local = f'{path_to_here}/data-cache'
os.makedirs(cache_rootdir_local, exist_ok=True)

# get configuration dictionary
config_dict = get_config_dict()

# cache directory for big file
project_tmpdir = f"{config_dict['project_tmpdir']}/cache-model-calcs"
os.makedirs(project_tmpdir, exist_ok=True)

os.environ['INTAKE_LOCAL_CACHE_DIR'] = f'{project_tmpdir}/intake-cache'    

# location of model data
model_data_dir = config_dict['model_data_dir']
assert os.path.exists(model_data_dir), (
    f'model_data_dir d.n.e.: {model_data_dir}'
)


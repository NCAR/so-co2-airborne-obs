import os
import yaml
from jinja2 import Template

def get_config_dict():
    """return the configuration dictionary with environment variables replaced"""
    
    with open('_config_calc.yml') as fid:
        config_dict_in = yaml.safe_load(fid)    
    
    required_keys = ['project_tmpdir', 'model_data_dir', 'dash_asset_fname']
    for key in required_keys:
        assert key in config_dict_in, f'config missing {key}'
        
    config_dict = {}
    for key, value in config_dict_in.items():
        t = Template(value)
        config_dict[key] = t.render(env=os.environ)
    
    return config_dict

# get configuration dictionary
config_dict = get_config_dict()

# cache directory for big file
project_tmpdir = config_dict['project_tmpdir']
os.makedirs(project_tmpdir, exist_ok=True)

# location of model data
model_data_dir = config_dict['model_data_dir']
model_data_dir_root = os.path.dirname(model_data_dir)
os.makedirs(model_data_dir_root, exist_ok=True)

dash_asset_fname = config_dict['dash_asset_fname']
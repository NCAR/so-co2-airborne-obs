import os
import yaml
from jinja2 import Template

path_to_here = os.path.dirname(os.path.realpath(__file__))

def _get_config_dict():
    """return the configuration dictionary with environment variables replaced"""
    
    with open(f'{path_to_here}/_config_calc.yml') as fid:
        config_dict_in = yaml.safe_load(fid)    
    
    required_keys = ['project_tmpdir', 'model_data_dir', 'dash_asset_fname']
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


# get configuration dictionary
def get(parameter):
    config_dict = _get_config_dict()
    
    derived = False
    if parameter in ["model_data_dir_root"]:
        derived = True
    else:
        assert parameter in config_dict, f"unknown parameter {parameter}"
    
    if not derived:
        value = config_dict[parameter]
    
    elif parameter == "model_data_dir_root":
        model_data_dir = config_dict['model_data_dir']
        value = os.path.dirname(model_data_dir)

    if parameter in [
        "project_tmpdir",
        "project_tmpdir_obs",
        "model_data_dir_root",
    ]:
        try:
            os.makedirs(value, exist_ok=True)    
        except OSError:
            print(f"error in config: cannot mkdir {value}")
            TMPDIR = os.environ["TMPDIR"]
            value = f"{TMPDIR}/so-co2-airborne-obs/{parameter}"            
            print(f"setting {parameter} to {value}")
            os.makedirs(value, exist_ok=True) 
                                
    return value
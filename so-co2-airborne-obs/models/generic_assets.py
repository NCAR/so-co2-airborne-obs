"""tools to support getting data assets in a general way
"""
import yaml
from glob import glob
from jinja2 import Template
from .config import path_to_here, model_data_dir

known_products = ['molefractions', 'fluxes', 'ObsPack']

def get_model_info(this_model):
    """return model info"""
    with open(f'{path_to_here}/info_models.yaml') as fid:
        model_info_all = yaml.safe_load(fid)

    model_info = {}
    for key in model_info_all:
        info = model_info_all[key]
        key_reg = key.replace('-', '_')
        if this_model not in info:
            model_info[key_reg] = {}
        else:
            if key_reg == 'products':
                validate_keys(
                    info[this_model], known_products, ['path', 'glob']
                )     
            if key_reg == 'ftp_access':
                validate_keys(
                    info[this_model], ['ftp-site', 'ftp-data-dir']
                )
            model_info[key_reg] = info[this_model]            
            
    return model_info


def list_assets(this_model, product):
    """return a list of file assets by type"""    
    model_info = get_model_info(this_model)
    
    if product not in model_info['products']:
        print(f'no "{product}" data for {this_model}')
        return []

    pinfo = model_info['products'][product]
    if not isinstance(pinfo, list):
        pinfo = [pinfo]
    
    groups = {}
    for i, pinfo_i in enumerate(pinfo):
        local_dir = Template(pinfo_i['path']).render(dash_data_directory=model_data_dir)
        glob_expression = pinfo_i['glob']
        groups[i] = sorted(glob(f'{local_dir}/{glob_expression}'))

    return groups


def validate_keys(info_dict, *valid_key_list_args):
    """
    recursive validation of keys in a dict
    """
    recursion_level = len(valid_key_list_args)
    if recursion_level > 1:
        validate_keys(info_dict, valid_key_list_args[0])
        for key, info_dict_i in info_dict.items():
            validate_keys(info_dict_i, *valid_key_list_args[1:])
    else:
        if isinstance(info_dict, list):
            for sub_info in info_dict:
                validate_keys(sub_info, *valid_key_list_args)
        elif isinstance(info_dict, dict):
            list_valid_keys = valid_key_list_args[0]
            if list_valid_keys:
                assert all(k in list_valid_keys for k in info_dict.keys()), (                
                    f'bad dict:\n {info_dict}\nexpected:\n{list_valid_keys}'
                )
        else:
            raise ValueError(f'unexpected type: {info_dict}')
            
     
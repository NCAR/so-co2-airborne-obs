import os
from glob import glob
import yaml
from jinja2 import Template

path_to_here = os.path.dirname(os.path.realpath(__file__))


def _get_config_dict():
    """return the configuration dictionary with environment variables replaced"""

    with open(f"{path_to_here}/_config_calc.yml") as fid:
        config_dict_in = yaml.safe_load(fid)

    required_keys = ["project_tmpdir", "model_data_dir", "dash_asset_fname"]
    for key in required_keys:
        assert key in config_dict_in, f"config missing {key}"

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

    if parameter in ["cache_dirs", "cache_dirs_all", "cache_dirs_ec"]:
        project_tmpdir = config_dict["project_tmpdir"]

        if parameter == "cache_dirs_ec":
            return [
                f"{project_tmpdir}/cache-emergent-constraint/pickles",
            ]

        cache_dirs = [
            "./data/cache",
            f"{project_tmpdir}/cache-emergent-constraint",
        ]

        if parameter == "cache_dirs_all":
            # this is a circular dependency, which is stupid
            # but points to some basic deficiencies in the overall
            # design. Anyway, confine import to here.
            import models.config_local

            cache_dirs.append(
                [
                    models.config_local.cache_rootdir_local,
                    models.config_local.project_tmpdir,
                ]
            )

        return cache_dirs

    elif parameter in config_dict:
        value = config_dict[parameter]

        if parameter in [
            "project_tmpdir",
            "project_tmpdir_obs",
            "model_data_dir_root",
        ]:
            try:
                os.makedirs(value, exist_ok=True)
            except OSError:
                print(f"config warning: cannot mkdir {value}")
                try:
                    TMPDIR = os.environ["TMPDIR"]
                except:
                    TMPDIR = os.environ["HOME"]
                value = f"{TMPDIR}/so-co2-airborne-obs/{parameter}"
                print(f"setting {parameter} to {value}")
                os.makedirs(value, exist_ok=True)

        return value

    else:
        raise ValueError(f"unknown parameter {parameter}")

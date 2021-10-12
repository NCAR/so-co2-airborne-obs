import os
import sys

sys.path.insert(0, "../")
import config

path_to_here = os.path.dirname(os.path.realpath(__file__))

# local cache path: files that are small enough to fit in the repo
cache_rootdir_local = f'{path_to_here}/data-cache'
os.makedirs(cache_rootdir_local, exist_ok=True)

project_tmpdir = config.get("project_tmpdir")

# cache directory for big file
project_tmpdir = f"{project_tmpdir}/cache-model-calcs"
os.makedirs(project_tmpdir, exist_ok=True)

os.environ['INTAKE_LOCAL_CACHE_DIR'] = f'{config.get("project_tmpdir")}/intake-cache'    

# location of model data
model_data_dir = config.get('model_data_dir')


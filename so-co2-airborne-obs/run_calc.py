import os
import yaml
import subprocess 
import json 
import pathlib
import nbterm 
import traceback
import asyncio


def get_toc_files(toc_dict, file_list=[]):
    
    for key, value in toc_dict.items():
        if key in ['root', 'file']:
            file_list.append(value)
        elif key == 'sections':
            file_list_ext = []
            for sub in value:
                file_list_ext = get_toc_files(sub, file_list_ext)
            file_list.extend(file_list_ext)
    return file_list


def get_conda_kernel_cwd(name: str):
    command = ['conda', 'env', 'list', '--json']
    output = subprocess.check_output(command).decode('ascii')
    envs = json.loads(output)['envs']
    for env in envs:
        env = pathlib.Path(env)
        if name == env.stem:
            return env 

    else:
        return None


def execute_notebook(notebook_path: str, kernel_cwd: str, output_dir=None):
    try:
        _nb_path = pathlib.Path(notebook_path)
        if not output_dir:
            output_dir = _nb_path.parent

        save_path = pathlib.Path(output_dir) / _nb_path.name
        nb = nbterm.Notebook(nb_path=_nb_path, kernel_cwd=kernel_cwd, save_path=save_path)
        asyncio.run(nb.run_all())
        nb.save(save_path)
        print(f"Executed notebook has been saved to: {save_path}")

    except Exception:
        msg = f'Error executing the notebook "{notebook_path}".\n'
        msg += f'See notebook "{notebook_path}" for the traceback.\n'
        print(f'{traceback.format_exc()}\n{msg}')




if __name__ == '__main__':
    
    output_dir = '/glade/u/home/mclong/test-calc'
    os.makedirs(output_dir, exist_ok=True)
    
    with open('_toc.yml') as fid:
        toc_dict = yaml.safe_load(fid)
    
    pre_notebooks = [] #['_prestage-data.ipynb', '_precompute.ipynb']
    notebooks = list(filter(lambda b: os.path.exists(f'{b}.ipynb'), get_toc_files(toc_dict)))
    notebooks = [f'{f}.ipynb' for f in notebooks]

    kernel_cwd = get_conda_kernel_cwd(name='so-co2')
    if kernel_cwd:
        for nb in pre_notebooks + notebooks:
            print(f'executing {nb}')
            execute_notebook(nb, kernel_cwd=kernel_cwd)
          

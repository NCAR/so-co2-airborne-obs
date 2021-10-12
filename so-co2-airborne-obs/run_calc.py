import os
from glob import glob

import subprocess 
import sys
import pathlib

import json 
import yaml

import nbformat
import nbterm 

import traceback
import asyncio

import config

import click

def get_toc_files(notebooks_only=True):
    """return a list of files in the _toc.yml"""
    with open('_toc.yml') as fid:
        toc_dict = yaml.safe_load(fid)
    
    def _toc_files(toc_dict, file_list=[]):
        for key, value in toc_dict.items():
            if key in ['root', 'file']:
                if notebooks_only and not os.path.exists(value + '.ipynb'):
                    continue
                if notebooks_only:
                    file_list.append(f'{value}.ipynb')
                else:
                    file_list.append(value)

            elif key == 'sections':
                file_list_ext = []
                for sub in value:
                    file_list_ext = _toc_files(sub, file_list_ext)
                file_list.extend(file_list_ext)
        return file_list
    
    return _toc_files(toc_dict)


def get_conda_kernel_cwd(name: str):
    """get the directory of a conda kernel by name"""
    command = ['conda', 'env', 'list', '--json']
    output = subprocess.check_output(command).decode('ascii')
    envs = json.loads(output)['envs']
    for env in envs:
        env = pathlib.Path(env)
        if name == env.stem:
            return env 
    else:
        return None


def nb_set_kernelname(file_in, kernel_name, file_out=None):
    """set the kernel name to python3"""
    if file_out is None:
        file_out = file_in        
    data = nbformat.read(file_in, as_version=nbformat.NO_CONVERT)        
    data['metadata']['kernelspec']['name'] = kernel_name
    nbformat.write(data, file_out)

    
def nb_get_kernelname(file_in):
    """get the kernel name of a notebook"""
    data = nbformat.read(file_in, as_version=nbformat.NO_CONVERT)
    return data['metadata']['kernelspec']['name']
    

def nb_clear_outputs(file_in, file_out=None):
    """clear output cells"""
    if file_out is None:
        file_out = file_in           
    data = nbformat.read(file_in, as_version=nbformat.NO_CONVERT)
    
    assert isinstance(data['cells'], list), 'cells is not a list'
    
    cells = []
    for cell in data['cells']:
        if cell['cell_type'] == 'code':
            cell['execution_count'] = None
            cell['outputs'] = []
        cells.append(cell)
    data['cells'] = cells
    nbformat.write(data, file_out)

    
def nb_execute_nbterm(notebook_path: str, kernel_cwd=None, output_dir=None):
    try:
        _nb_path = pathlib.Path(notebook_path)
        if not output_dir:
            output_dir = _nb_path.parent

        save_path = pathlib.Path(output_dir) / _nb_path.name
        nb = nbterm.Notebook(nb_path=_nb_path, save_path=save_path) #kernel_cwd=kernel_cwd, 
        asyncio.run(nb.run_all())
        nb.save(save_path)
        print(f"Executed notebook has been saved to: {save_path}")
        return True
    
    except Exception:
        msg = f'Error executing the notebook "{notebook_path}".\n'
        msg += f'See notebook "{notebook_path}" for the traceback.\n'
        print(f'{traceback.format_exc()}\n{msg}')
        return False

    
def nb_execute(notebook_filename, output_dir='.', kernel_name='python3'):
    """
    Execute a notebook.
    see http://nbconvert.readthedocs.io/en/latest/execute_api.html
    """
    import io
    import nbformat
    from nbconvert.preprocessors import ExecutePreprocessor
    from nbconvert.preprocessors import CellExecutionError

    #-- open notebook
    with io.open(notebook_filename, encoding='utf-8') as f:
        nb = nbformat.read(f, as_version=nbformat.NO_CONVERT)

    # config for execution
    ep = ExecutePreprocessor(timeout=None, kernel_name=kernel_name)

    # run with error handling
    try:
        out = ep.preprocess(nb, {'metadata': {'path': './'}})

    except CellExecutionError:
        out = None
        msg = f'Error executing the notebook "{notebook_filename}".\n'
        msg += f'See notebook "{notebook_filename}" for the traceback.\n'
        print(msg)

    finally:
        nb_out = os.path.join(output_dir, os.path.basename(notebook_filename))
        with io.open(nb_out, mode='w', encoding='utf-8') as f:
            nbformat.write(nb, f)
        print(f'wrote: {nb_out}')
        
    return out    


def kernel_munge(kernel_name):
    """return the kernel name as it's rendered in the notebook metadata"""
    return f'conda-env-miniconda3-{kernel_name}-py'


@click.command()
@click.option('--notebook', default=None)
@click.option('--run-pre', is_flag=True)
@click.option('--stop-on-fail', is_flag=True)
@click.option('--clear-cache', is_flag=True)
def main(run_pre, notebook, stop_on_fail, clear_cache=False):
    """run notebooks"""
    
    project_kernel = config.get('project_kernel')
    
    assert os.environ['CONDA_DEFAULT_ENV'] == project_kernel, (
        f'activate "{project_kernel}" conda environment before running'
    )

    if notebook is None:
        notebook_list = config.get('pre_notebooks') if run_pre else []
        notebook_list = notebook_list + get_toc_files()
    else:
        notebook_list = [notebook]    
    
    skip_notebooks = config.get('R_notebooks')
    notebook_list = [f for f in notebook_list if f not in skip_notebooks]
    
    # check kernels
    for nb in notebook_list:
        notebook_kernel = nb_get_kernelname(nb)
        assert notebook_kernel == kernel_munge(project_kernel), (
            f'{nb}: unexpected kernel: {notebook_kernel}'
        )
    
    # run the notebooks
    if clear_cache:
        cache_dirs = config.get("cache_dirs")
        for d in cache_dirs:
            subprocess.check_call(["rm", "-fr", f"{d}/*"])
    
    cwd = os.getcwd()
    failed_list = []
    for nb in notebook_list:
        print('-'*80)
        print(f'executing: {nb}')

        # set the kernel name to fool nbterm into running this
        nb_set_kernelname(nb, kernel_name='python3')

        # clear output
        nb_clear_outputs(nb)

        # run the notebook
        ok = nb_execute(nb, output_dir=cwd)
        if not ok:
            print('failed')
            if stop_on_fail:
                sys.exit(1)            
            failed_list.append(nb)

        # set the kernel back
        nb_set_kernelname(nb, kernel_name=kernel_munge(project_kernel))
        print()
        
    if failed_list:
        print('failed list')  
        print(failed_list)
        sys.exit(1)


if __name__ == '__main__':
    main()
    

    
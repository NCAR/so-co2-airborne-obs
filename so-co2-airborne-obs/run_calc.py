import os
import subprocess 
import sys
import pathlib

import json 
import yaml

import nbformat
import nbterm 

import traceback
import asyncio


def get_toc_files(toc_dict, file_list=[], notebooks_only=True):
    """return a list of files in the _toc.yml"""
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
                file_list_ext = get_toc_files(sub, file_list_ext, notebooks_only)
            file_list.extend(file_list_ext)
    return file_list


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


def list_notebooks_in_toc():
    """list notebooks found in _toc.yml"""
    with open('_toc.yml') as fid:
        toc_dict = yaml.safe_load(fid)
    
    pre_notebooks = ['_prestage-data.ipynb',] # '_precompute.ipynb']
    return pre_notebooks + get_toc_files(toc_dict)
    

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

    
def nb_execute(notebook_path: str, kernel_cwd=None, output_dir=None):
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


if __name__ == '__main__':
    
    project_kernel = 'so-co2'
        
    assert os.environ['CONDA_DEFAULT_ENV'] == project_kernel, (
        f'activate "{project_kernel}" conda environment before running'
    )

    print('notebooks in _toc.yml')
    notebooks = list_notebooks_in_toc()
    print(notebooks, end='\n\n')   
       
    print('checking notebook kernels')
    for nb in notebooks:
        notebook_kernel = nb_get_kernelname(nb)
        assert notebook_kernel == f'conda-env-miniconda3-{project_kernel}-py', (
            f'{nb}: unexpected kernel: {notebook_kernel}'
        )
    print()    
        
    cwd = os.getcwd()
    failed_list = []
    for nb in notebooks:
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
            failed_list.append(nb)

        # set the kernel back
        nb_set_kernelname(nb, kernel_name=project_kernel)
        print()

    if failed_list:
        print('failed list')  
        print(failed_list)

          

"""
so-co2-airborne-obs/emergent-constraint-s2n.ipynb so-co2-airborne-obs/emergent-constraint.ipynb so-co2-airborne-obs/fluxes.ipynb so-co2-airborne-obs/gradients-aircraft-sampling.ipynb so-co2-airborne-obs/gradients-main.ipynb so-co2-airborne-obs/gradients-methane.ipynb so-co2-airborne-obs/gradients-profiles.ipynb so-co2-airborne-obs/gradients-seasonal-amplitude.ipynb so-co2-airborne-obs/gradients-sf6.ipynb so-co2-airborne-obs/obs-aircraft.ipynb so-co2-airborne-obs/obs-main.ipynb so-co2-airborne-obs/obs-simulated-distributions.ipynb so-co2-airborne-obs/obs-surface-error.ipynb so-co2-airborne-obs/obs-surface.ipynb
"""
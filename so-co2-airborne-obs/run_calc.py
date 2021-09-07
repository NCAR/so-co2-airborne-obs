import os
from glob import glob
import yaml


def exec_nb(notebook_filename, output_dir='.', kernel_name='python3'):
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
        nb = nbformat.read(f, as_version=4)

    # config for execution
    ep = ExecutePreprocessor(timeout=600, kernel_name=kernel_name)

    # run with error handling
    try:
        out = ep.preprocess(nb, {'metadata': {'path': './'}})

    except CellExecutionError:
        out = None
        msg = f'Error executing the notebook "{notebook_filename}".\n'
        msg += f'See notebook "{notebook_filename}" for the traceback.\n'
        print(msg)
        raise

    finally:
        nb_out = os.path.join(output_dir, os.path.basename(notebook_filename))
        with io.open(nb_out, mode='w', encoding='utf-8') as f:
            nbformat.write(nb, f)

    return out


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


if __name__ == '__main__':
    
    output_dir = '/glade/u/home/mclong/test-calc'
    os.makedirs(output_dir, exist_ok=True)
    
    with open('_toc.yml') as fid:
        toc_dict = yaml.safe_load(fid)
    
    pre_notebooks = [] #['_prestage-data.ipynb', '_precompute.ipynb']
    notebooks = list(filter(lambda b: os.path.exists(f'{b}.ipynb'), get_toc_files(toc_dict)))
    notebooks = [f'{f}.ipynb' for f in notebooks]
    
    for nb in pre_notebooks + notebooks:
        print(f'executing {nb}')
        exec_nb(nb, kernel_name='conda-env-miniconda3-so-co2-py')
#!/usr/bin/env python
import os
from subprocess import Popen, PIPE
import tarfile


def ensure_data(model_data_dir_root):
    """get the data from dash repo"""
    
    dirname = f'{model_data_dir_root}/Long-etal-2021-SO-CO2-Science'    
    fname = 'Long-etal-2021-SO-CO2-Science.tar.gz'
    if os.path.isdir(dirname):
        return dirname
       
    # run wget to stage data
    # TODO: support curl too
    cwd = os.getcwd()
    script = f'{cwd}/wget-dash-archive.sh'
    
    os.chdir(model_data_dir_root)
    
    p = Popen(['bash', script], stdout=PIPE, stderr=PIPE)
    stdout, stderr = p.communicate()
    if p.returncode:    
        print(stderr.decode('UTF-8'))
        print(stdout.decode('UTF-8'))
        raise OSError('data transfer failed')    

    # untar archive
    assert os.path.isfile(fname), f'missing {fname}'
    tar = tarfile.open(fname, "r:gz")
    tar.extractall()
    tar.close()

    os.chdir(cwd)
    
    return dirname
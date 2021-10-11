import os
import sys
from glob import glob
import yaml
from ftplib import FTP
import hashlib

import numpy as np
import calendar
from collections.abc import Iterable

from . config_local import project_tmpdir


def calc_eomday(year, month):
    """end of month day"""
    if isinstance(year, Iterable):
        assert isinstance(month, Iterable)
        return np.array([calendar.monthrange(y, m)[-1] for y, m in zip(year, month)])
    else:
        return calendar.monthrange(year, month)[-1]


def get_ftp_md5(ftp, remote_file):
    """Compute checksum on remote ftp file."""
    m = hashlib.md5()
    ftp.retrbinary(f'RETR {remote_file}', m.update)
    return m.hexdigest()


def get_loc_md5(local_path):
    """Compute checksum on local file."""
    with open(local_path, 'rb') as fid:
        data = fid.read()
    return hashlib.md5(data).hexdigest()


# Ref: https://stackoverflow.com/questions/3173320/text-progress-bar-in-the-console
# https://gist.github.com/aubricus/f91fb55dc6ba5557fbab06119420dd6a
def print_progressbar(iteration, total, prefix='', suffix='', decimals=1, bar_length=100):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        bar_length  - Optional  : character length of bar (Int)
    """
    str_format = '{0:.' + str(decimals) + 'f}'
    percents = str_format.format(100 * (iteration / float(total)))
    filled_length = int(round(bar_length * iteration / float(total)))
    bar = '█' * filled_length + '-' * (bar_length - filled_length)

    #print('\r%s |%s| %s%% %s' % (prefix, bar, percents, suffix), end='\r')
    #print('\r%s |%s| %s%s %s' % (prefix, bar, percents, '%', suffix))
    sys.stdout.write('%s |%s| %s%s %s\r' % (prefix, bar, percents, '%', suffix)),

    if iteration == total:
        print()
    sys.stdout.flush() 

def ftp_files(ftp_site, ftp_data_dir, pinfo_all, clobber=False, check_md5=False):
    """
    get files from ftp
    """

    if not isinstance(pinfo_all, list):
        pinfo_all = [pinfo_all]

    local_cwd = os.getcwd()

    for pinfo in pinfo_all:
        local_dir = pinfo['path']
        remote_dir = os.path.join(ftp_data_dir, pinfo["path"].replace(project_tmpdir+"/", ""))

        glob_expression = pinfo['glob']

        # make a local directory
        os.makedirs(local_dir, exist_ok=True)

        os.chdir(local_dir)

        # login
        ftp = FTP(ftp_site)
        ftp.login()
        ftp_cwd = ftp.pwd()

        ftp.cwd(remote_dir)
        files = ftp.nlst(glob_expression)

        if clobber:
            rm_files = list(filter(lambda f: os.path.exists(f), files))
            print(f'removing {len(rm_files)} existing files')
            for f in rm_files:
                os.remove(f)

        if check_md5:
            check_files = list(filter(lambda f: os.path.exists(f), files))[::-1]
            if check_files:

                nfile = len(check_files)
                print('computing local checksums')
                local_md5s = []
                for filename in check_files:
                    local_md5s.append(dask.delayed(get_loc_md5)(f'{local_dir}/{filename}'))
                local_md5s = dask.compute(*local_md5s)

                print(f'checking {nfile} files')
                print_progressbar(0, nfile, prefix='md5 check:', suffix='', bar_length=50)
                for i, (filename, local_md5) in enumerate(zip(check_files, local_md5s)):
                    remote_md5 = get_ftp_md5(ftp, filename)
                    if local_md5 != remote_md5:
                        print(f'checksum mismatch, removing {filename}')
                        os.remove(filename)
                    print_progressbar(i+1, nfile, prefix='md5 check:', suffix='', bar_length=50)

        files = list(filter(lambda f: not os.path.exists(f), files))
        nfile = len(files)

        if files:
            print(f'transfering {nfile} files')
            print_progressbar(0, nfile, prefix='FTP transfer:',
                                   suffix='', bar_length=50)

            for i, filename in enumerate(files):
                with open(filename, 'wb') as fid:
                    ftp.retrbinary(f'RETR {filename}', fid.write)
                print_progressbar(i+1, nfile, prefix='FTP transfer:',
                                       suffix='', bar_length=50)

    os.chdir(local_cwd)
    ftp.quit()

    
def ecmwf_dataserver(ecmwf_credentials_file, products, 
                     product='molefractions', version='v18r2'):
    from ecmwfapi import ECMWFDataServer
    
    #products = get_model_info('products')['CAMS']
    pinfo = products[product]
    local_dir = pinfo['path']
    glob_expression = pinfo['glob']

    os.makedirs(local_dir, exist_ok=True)
        
    try:
        with open(ecmwf_credentials_file) as fid:
            credentials = yaml.safe_load(fid)
    except:
        raise

    existing_files = sorted(glob(f'{local_dir}/{glob_expression}'))  

    server = ECMWFDataServer(**credentials)
    print(server)
    
    if product == 'molefractions':
        for year in range(1995, 2019):
            for mon in range(1, 13):
                target = f'{local_dir}/z_cams_l_lsce_{product}_{year:04d}-{mon:02d}_{version}_ra_ml_3h_co2.tar'
                target_xvf = f'{local_dir}/z_cams_l_lsce_{year:04d}{mon:02d}_{version}_ra_ml_3h_co2.nc'

                if os.path.exists(target) or os.path.exists(target_xvf):
                    continue
    
                print('-'*40)
                print(f'transfering {year:04d}-{mon:02d}')
    
                eomday = calc_eomday(year, mon)
                server.retrieve({
                    "dataset": "cams_ghg_inversions",
                    "datatype": "ra",
                    "date": f"{year:04d}-{mon:02d}-01/to/{year:04d}-{mon:02d}-{eomday:02d}",
                    "frequency": "3h",
                    "param": "co2",
                    "quantity": "concentration",
                    "version": "v18r2",
                    "target": target,
                })
                check_call(['tar', '-xf', target, '-C', local_dir])
                check_call(['rm', '-f', target])
       
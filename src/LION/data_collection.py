# Author: Emily Costa
# Created on: May 3, 2021

# Currently only collects info from Darshan.

import numpy as np
import time
from multiprocessing import Pool, cpu_count, Process
import pandas as pd
from os import walk, mkdir
from os.path import join, isfile, exists
import pyarrow as pa
import pyarrow.parquet as pq

def collect_darshan_data(path_to_total, ranks=None, save_path='./run_info.parquet', chunksize=1000, verbose=False):
    '''
    Collects data from Darshan logs that can be used for clustering and
    analysis of those clusters.

    Parameters
    ----------
    path_to_total: string
        Path to Darshan logs parsed in the 'total' format. The logs should be
        sorted by user ID and executable name.
    ranks: int, optional
        Parallize the data collection by increasing the number of processes
        collecting and saving the data.
    save_path: string, optional
        Where to save the collected data.
    chunksize: int, optional
        In order to checkpoint data and continue if data collection is halted,
        set this to the number of runs to collect info on per "chunk". This will
        get the program to write the info to the output file with the final info.
    verbose: boolean, optional
        For debugging and info on amount of info being collected.
    
    Returns
    -------
    data: pd.DataFrame 
        All info for clustering.
    '''
    if ranks is None:
        ranks = cpu_count()
    if(verbose):
        print('# of ranks: %d'%ranks)
    f_fs = []
    if(exists(save_path)):
        f_fs = pd.read_parquet(save_path)['Filename'].tolist()
    pool = Pool(processes=ranks)
    # Find the applications that have more runs than the threshold.
    dirs_info = []
    for root, ds, fs in walk(path_to_total):
        fs = np.setdiff1d(fs,f_fs)
        for fn in fs:
            dirs_info.append(join(root,fn))
    if(verbose):
        print('Total Darshan logs: %d'%len(dirs_info))
    pqwriter = None
    # Now collect data
    data = pd.DataFrame()
    start = time.time()
    chunk_number = 1
    n = 0
    for i in pool.imap_unordered(_collect_data, dirs_info):
        if(i is None):
            continue
        data = data.append(i, ignore_index=True)
        #if(verbose and (chunk_number-1)*chunksize%10000==0):
        n = (((chunk_number-1)*chunksize)+data.shape[0])
        if(verbose and n%10000==0):
            end = time.time()
            print('It took %d minutes for %d files'%((end-start)/60,n))
        if(data.shape[0]>chunksize-1):
            table = pa.Table.from_pandas(data)
            if(chunk_number==1 and exists(save_path)==False):
                pqwriter = pq.ParquetWriter(save_path, table.schema)
            elif(chunk_number==1 and exists(save_path)==True):
                temp = pd.read_parquet(save_path)
                pqwriter = pq.ParquetWriter(save_path, table.schema)
                pqwriter.write_table(pa.Table.from_pandas(temp))
                temp = None
            if(verbose):
                print("Chunk #%d has been written to file."%chunk_number)
            chunk_number = chunk_number + 1
            pqwriter.write_table(table)
            data = pd.DataFrame()    
    table = pa.Table.from_pandas(data)
    if(chunk_number==1):
        pqwriter = pq.ParquetWriter(save_path, table.schema)
    pqwriter.write_table(table)
    if(pqwriter):
        pqwriter.close()
    total_files = data.shape[0]+chunk_number*chunksize
    if(verbose):
        print('Files collected total %d in %d chunks.'%(total_files,chunk_number+1))
    return data

def _get_runs(dir_info):
    dir = dir_info[0]
    E   = dir_info[1]
    runs= []
    for root, ds, fs in walk(dir):
        if(len(fs)<E):
            return None
        for f in fs:
            runs.append(join(root,f))
    return runs
        
def _collect_data(run):
    fn = run.split('/')[-1]
    with open(run,'r') as f:
        try:
            lines = f.read().split("\n")
        except UnicodeDecodeError:
            print('Read err%s'%f)
        f.close()
    try: 
        # add: start time, end time, performance
        start_time = lines[5].split(' ')[-1]
        end_time = lines[7].split(' ')[-1]
        read_size   = int([line for line in lines if 'total_POSIX_BYTES_READ:'    in line][0][23:])
        write_size  = int([line for line in lines if 'total_POSIX_BYTES_WRITTEN:' in line][0][26:])
        read_time   = float([line for line in lines if 'total_POSIX_F_READ_TIME:'   in line][0][24:])
        write_time  = float([line for line in lines if 'total_POSIX_F_WRITE_TIME:'  in line][0][25:])
        read_0_100       = int([line for line in lines if 'total_POSIX_SIZE_READ_0_100:'       in line][0][28:])
        read_100_1K      = int([line for line in lines if 'total_POSIX_SIZE_READ_100_1K:'      in line][0][29:])
        read_1K_10K      = int([line for line in lines if 'total_POSIX_SIZE_READ_1K_10K:'      in line][0][29:])
        read_10K_100K    = int([line for line in lines if 'total_POSIX_SIZE_READ_10K_100K:'    in line][0][31:])
        read_100K_1M     = int([line for line in lines if 'total_POSIX_SIZE_READ_100K_1M:'     in line][0][30:])
        read_1M_4M       = int([line for line in lines if 'total_POSIX_SIZE_READ_1M_4M:'       in line][0][28:])
        read_4M_10M      = int([line for line in lines if 'total_POSIX_SIZE_READ_4M_10M:'      in line][0][29:])
        read_10M_100M    = int([line for line in lines if 'total_POSIX_SIZE_READ_10M_100M:'    in line][0][31:])
        read_100M_1G     = int([line for line in lines if 'total_POSIX_SIZE_READ_100M_1G:'     in line][0][30:])
        read_1G_plus     = int([line for line in lines if 'total_POSIX_SIZE_READ_1G_PLUS:'     in line][0][30:])
        write_0_100       = int([line for line in lines if 'total_POSIX_SIZE_WRITE_0_100:'       in line][0][29:])
        write_100_1K      = int([line for line in lines if 'total_POSIX_SIZE_WRITE_100_1K:'      in line][0][30:])
        write_1K_10K      = int([line for line in lines if 'total_POSIX_SIZE_WRITE_1K_10K:'      in line][0][30:])
        write_10K_100K    = int([line for line in lines if 'total_POSIX_SIZE_WRITE_10K_100K:'    in line][0][32:])
        write_100K_1M     = int([line for line in lines if 'total_POSIX_SIZE_WRITE_100K_1M:'     in line][0][31:])
        write_1M_4M       = int([line for line in lines if 'total_POSIX_SIZE_WRITE_1M_4M:'       in line][0][29:])
        write_4M_10M      = int([line for line in lines if 'total_POSIX_SIZE_WRITE_4M_10M:'      in line][0][30:])
        write_10M_100M    = int([line for line in lines if 'total_POSIX_SIZE_WRITE_10M_100M:'    in line][0][32:])
        write_100M_1G     = int([line for line in lines if 'total_POSIX_SIZE_WRITE_100M_1G:'     in line][0][31:])
        write_1G_plus     = int([line for line in lines if 'total_POSIX_SIZE_WRITE_1G_PLUS:'     in line][0][31:])
        exe = lines[2]
        uid = lines[3][7:]
    except IndexError:
        return None
    exe = exe.split("/")
    if(len(exe)>1):
        exe = exe[-1]
    else:
        exe = exe[-1][7:]
    try:
        read_perf = read_size/read_time
        write_perf = write_size/write_time
    except ZeroDivisionError:
        return None
    exe = exe.split(" ")
    exe = exe[0]
    exe = exe.strip()
    application = '%s_%s'%(exe,uid)
    d = {'Filename': fn, 'Application': application, 'Amount of Read I/O': read_size, 'Amount of Write I/O': write_size,
            'Read 0-100': read_0_100, 'Read 100-1K': read_100_1K, 'Read 1K-10K': read_1K_10K, 'Read 10K-100K': read_10K_100K,
            'Read 100K-1M': read_100K_1M, 'Read 1M-4M': read_1M_4M, 'Read 4M-10M': read_4M_10M, 'Read 10M-100M': read_10M_100M,
            'Read 100M-1G': read_100M_1G, 'Read 1G+': read_1G_plus, 'Write 0-100': write_0_100, 'Write 100-1K': write_100_1K, 
            'Write 1K-10K': write_1K_10K, 'Write 10K-100K': write_10K_100K, 'Write 100K-1M': write_100K_1M, 'Write 1M-4M': write_1M_4M, 
            'Write 4M-10M': write_4M_10M, 'Write 10M-100M': write_10M_100M, 'Write 100M-1G': write_100M_1G, 'Write 1G+': write_1G_plus,
            'Read Performance': read_perf, 'Write Performance': write_perf, 'Start Time': start_time, 'End Time': end_time}
    return d


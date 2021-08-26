# Author: Emily Costa
# Created on: May 5, 2021
import pyarrow as pa
import pyarrow.parquet as pq
import time
from sklearn.cluster import AgglomerativeClustering
import pandas as pd
from os import walk, mkdir
from os.path import join, isfile, exists
import numpy as np
from multiprocessing import Pool, cpu_count
from sklearn.preprocessing import StandardScaler
from memory_profiler import profile
pd.options.mode.chained_assignment = None  # default='warn'

def cluster_runs(run_info, ranks=None, threshold=40, save_path=None, chunksize=1000, verbose=False):
    '''
    Parameters
    ----------
    run_info: pd.DataFrame
        Dataframe containing the run info for clustering, as collected from 
        a function in data_collection. Needs application, time, and cluster
        parameters.
    threshold: int, optional
        The threshold for how many times an application needs to be run in 
        order to be included in the data collection.
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
    clusters: pd.DataFrame
        Info on how the runs are clustered.
    '''
    if ranks is None:
        ranks = cpu_count()
    pool = Pool(processes=ranks)
    args = []
    for a in run_info['Application'].unique():
        mask = run_info['Application'] == a
        pos = np.flatnonzero(mask)
        tmp = run_info.iloc[pos]
        tmp = tmp.shape[0]
        if(verbose):
            print('Size of application %s is %d'%(a,tmp))
        if(tmp>threshold):
            args.append([a,run_info,threshold])
    clusters = pd.DataFrame()
    start = time.time()
    chunk_number = 1
    pqwriter = None
    n = 0
    for i in pool.imap_unordered(_cluster_with_run_info, args):
        if(i is None):
            continue
        clusters = clusters.append(i, ignore_index=True)
        n = (((chunk_number-1)*chunksize)+clusters.shape[0])
        if(verbose and n%10000==0):
            end = time.time()
            print('It took %d minutes for %d files'%((end-start)/60,n))
        if(clusters.shape[0]>chunksize-1):
            table = pa.Table.from_pandas(clusters)
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
            clusters = pd.DataFrame()    
    table = pa.Table.from_pandas(clusters)
    if(chunk_number==1):
        pqwriter = pq.ParquetWriter(save_path, table.schema)
    try:
        pqwriter.write_table(table)
    except ValueError:
        if(verbose):
            print('Chunk %d produced 0 clusters'%chunk_number)
    if(pqwriter):
        pqwriter.close()
    total_files = clusters.shape[0]+chunk_number*chunksize
    if(verbose):
        print('Files collected total >%d in %d chunks.'%(total_files,chunk_number+1))
    return clusters

@profile
def _cluster_with_run_info(args):
    application = args[0]
    df_results = args[1]
    threshold = args[2]
    mask = df_results['Application'] == application
    pos = np.flatnonzero(mask)
    df_results = df_results.iloc[pos]
    if(df_results.shape[0]<threshold):
        return None
    # Standardize
    df_results['Amount of Write I/O, Scaled'] = df_results['Amount of Write I/O']
    df_results['Amount of Read I/O, Scaled'] = df_results['Amount of Read I/O']
    df_results = df_results.set_index(['Filename', 'Application', 'Read Performance', 'Write Performance', 'Amount of Write I/O', 'Amount of Read I/O', 'Start Time', 'End Time'])
    scaler = StandardScaler() 
    try:
        df_scaled = scaler.fit_transform(df_results)
    except ValueError: 
        return None
    df_scaled = pd.DataFrame(df_scaled, index=df_results.index, columns=df_results.columns).reset_index()
    X = df_scaled[['Amount of Write I/O, Scaled', 'Write 0-100', 'Write 100-1K', 'Write 1K-10K', 'Write 10K-100K', 'Write 100K-1M', 'Write 1M-4M', 
                    'Write 4M-10M', 'Write 10M-100M', 'Write 100M-1G', 'Write 1G+']].copy()
    clustering_writes = AgglomerativeClustering(n_clusters=None, compute_full_tree=True, distance_threshold=0.1).fit(X)
    df_results['Cluster Write'] = clustering_writes.labels_
    X = df_scaled[['Amount of Read I/O, Scaled', 'Read 0-100', 'Read 100-1K', 'Read 1K-10K', 'Read 10K-100K','Read 100K-1M', 'Read 1M-4M', 
                    'Read 4M-10M', 'Read 10M-100M', 'Read 100M-1G', 'Read 1G+']].copy()
    clustering_reads  = AgglomerativeClustering(n_clusters=None, compute_full_tree=True, distance_threshold=0.1).fit(X)
    df_results['Cluster Read'] = clustering_reads.labels_
    # Divide by cluster
    max_read  = df_results['Cluster Read'].max()
    max_write = df_results['Cluster Write'].max()
    # For read
    cluster_no = 1
    df_return = pd.DataFrame()
    for i in range(0, max_read):
        #df_read = df_results[df_results['Cluster Read'] == i].sort_index().reset_index()
        mask = df_results['Cluster Read'] == i
        pos = np.flatnonzero(mask)
        df_read = df_results.iloc[pos].sort_index().reset_index()
        no_runs = df_read.shape[0]
        if(no_runs<threshold):
            continue
        for idx, row in df_read.iterrows():
            dict = {'Application': application, 'Operation': 'Read', 'Cluster Number': cluster_no,
            'Cluster Size': no_runs, 'Filename': row['Filename'], 'Performance': row['Read Performance'], 
            'I/O Amount': row['Amount of Read I/O'], 'Start Time': row['Start Time'], 'End Time': row['End Time']}
            df_return = df_return.append(dict, ignore_index=True)
        cluster_no = cluster_no + 1
    # Now write
    cluster_no = 1
    for i in range(0, max_write):
        #df_write = df_results[df_results['Cluster Write'] == i].sort_index().reset_index()
        mask = df_results['Cluster Write'] == i
        pos = np.flatnonzero(mask)
        df_write = df_results.iloc[pos].sort_index().reset_index()
        no_runs = df_write.shape[0]
        if(no_runs<threshold):
            continue
        for idx, row in df_write.iterrows():
            dict = {'Application': application, 'Operation': 'Write', 'Cluster Number': cluster_no,
            'Cluster Size': no_runs, 'Filename': row['Filename'], 'Performance': row['Write Performance'], 
            'I/O Amount': row['Amount of Write I/O'], 'Start Time': row['Start Time'], 'End Time': row['End Time']}
            df_return = df_return.append(dict, ignore_index=True)
        cluster_no = cluster_no + 1
    return df_return

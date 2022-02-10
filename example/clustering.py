# Author: Emily Costa
# Created on: May 5, 2021
import pyarrow as pa
import pyarrow.parquet as pq
import time
from sklearn.cluster import AgglomerativeClustering
import dask.dataframe as dd
import pandas as pd
from os import walk, mkdir
from os.path import join, isfile, exists
import numpy as np
from multiprocessing import Pool, cpu_count, Process
from sklearn.preprocessing import StandardScaler
from math import ceil as ceil

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
    block_size = 20000
    for a in run_info['Application'].unique().tolist():
        mask = run_info['Application'] == a
        pos = np.flatnonzero(mask)
        tmp = run_info.iloc[pos].sample(frac=1) # randomly shuffle so no time bias
        if(tmp.shape[0]>threshold):
            no_blocks = ceil(tmp.shape[0]/block_size)
            for n in range(0,no_blocks):
                adj_block_size = int(tmp.shape[0]/no_blocks)
                start_block = n*adj_block_size
                end_block = start_block + adj_block_size 
                tmp_b = tmp.iloc[start_block:end_block]
                args.append([a,tmp_b,threshold])
    columns=['Application', 'Cluster Number', 'Cluster Size', 'Operation', 'Filename', 'Performance', 'I/O Amount', 'Start Time', 'End Time']
    clusters = pd.DataFrame(columns=columns)
    start = time.time()
    chunk_number = 1
    pqwriter = None 
    schema_fields = [
            pa.field('Application', pa.string()),
            pa.field('Cluster Number', pa.int64()),
            pa.field('Cluster Size', pa.int64()),
            pa.field('Filename', pa.string()),
            pa.field('Operation', pa.string()),
            pa.field('Performance', pa.float32()),
            pa.field('I/O Amount', pa.int64()),
            pa.field('Start Time', pa.int64()),
            pa.field('End Time', pa.int64())]
    schema = pa.schema(schema_fields)
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
            table = pa.Table.from_pandas(clusters, schema=schema, preserve_index=False)
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
            clusters = pd.DataFrame(columns=columns)
    table = pa.Table.from_pandas(clusters, schema=schema, preserve_index=False)
    if(chunk_number==1):
        pqwriter = pq.ParquetWriter(save_path, table.schema)
    try:
        pqwriter.write_table(table)
    except KeyError:
        print('Columns in dataframe do not match the pyarrow schema.')
    if(pqwriter):
        pqwriter.close()
    total_files = clusters.shape[0]+chunk_number*chunksize
    if(verbose):
        print('Files collected total >%d in %d chunks.'%(total_files,chunk_number+1))
    return clusters

def _cluster_with_run_info(args):
    application = args[0]
    df_results = args[1]
    threshold = args[2]
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
    # Clustering
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
            'Cluster Size': no_runs, 'Filename': row['Filename'], 'Performance': float(row['Read Performance']), 
            'I/O Amount': int(row['Amount of Read I/O']), 'Start Time': int(row['Start Time']), 'End Time': int(row['End Time'])}
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
            'Cluster Size': no_runs, 'Filename': row['Filename'], 'Performance': float(row['Write Performance']), 
            'I/O Amount': int(row['Amount of Write I/O']), 'Start Time': int(row['Start Time']), 'End Time': int(row['End Time'])}
            df_return = df_return.append(dict, ignore_index=True)
        cluster_no = cluster_no + 1
    return df_return 

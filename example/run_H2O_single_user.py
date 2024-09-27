import sys
sys.path.insert(1, '../src/LION')
from data_collection import collect_darshan_data
from clustering import cluster_runs
from analysis_and_plots import cluster_characteristics, general_temporal_trends, io_performance_variability
import pandas as pd
from os.path import join

if __name__=='__main__':
    path_to_total    = '/scratch/costa.em/total/by_application_user'
    path_to_data     = '/scratch/costa.em/total/run_info.parquet'
    path_to_clusters = '/scratch/costa.em/total/cluster_info.parquet' 
    
    # Collect infos
    run_info = collect_darshan_data(path_to_total, save_path=path_to_data, verbose=False)

    # Cluster runs
    run_info = pd.read_parquet(path_to_data) # you need to feed the function a dataframe, not path to the info file
    cluster_info = cluster_runs(run_info, threshold=5, save_path=path_to_clusters, verbose=False)

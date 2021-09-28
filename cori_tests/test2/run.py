from data_collection import collect_darshan_data
from clustering import cluster_runs
from analysis_and_plots import cluster_characteristics, general_temporal_trends, io_performance_variability
import pandas as pd
from os.path import join

if __name__=='__main__':
    path_to_total = '/global/cscratch1/sd/emily/darshan-parse-parallel/parsed-logs'
    path_to_data = './run_info.parquet'
    path_to_clusters = './clusters.parquet'
    
    # Collect infos
    #run_info = collect_darshan_data(path_to_total, save_path=path_to_data, verbose=True)

    # Cluster runs
    #run_info = pd.read_parquet(path_to_data) # you need to feed the function a dataframe, not path to the info file
    #cluster_info = cluster_runs(run_info, threshold=100, ranks=None, save_path=path_to_clusters, verbose=True)

    # Analysis and Plotting
    clustered_runs    = pd.read_parquet(path_to_clusters)
    path_to_figures = './figures'
    #cluster_characteristics(clustered_runs, save_directory=path_to_figures)
    general_temporal_trends(clustered_runs, save_directory=path_to_figures)
    io_performance_variability(clustered_runs, save_directory=path_to_figures)

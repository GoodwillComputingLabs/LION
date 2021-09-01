from clustering import cluster_runs
import pandas as pd

if __name__=='__main__':
    #path_to_total = '/global/cscratch1/sd/emily/darshan_logs/parsed_darshan'
    #run_info = collect_darshan_data(path_to_total, save_path='/global/homes/e/emily/eju/project_incite/cori_results/test_chunking/run_info1.parquet', verbose=True)
    #print(run_info)
    run_info = pd.read_parquet('/global/homes/e/emily/eju/project_incite/cori_results/test_chunking/run_info.parquet')
    clusters = cluster_runs(run_info, threshold=100, save_path='./clusters.parquet', verbose=True)
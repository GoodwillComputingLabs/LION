# Author: Emily Costa
# Created on: Jun 3, 2021
# Functions for analyzing the I/O behaviors of applications on an HPC system using the clusters generated from `clustering`.

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math
import os
from os.path import join
from matplotlib import ticker
import seaborn as sns
import scipy.stats as stats
from datetime import datetime, timezone
from dateutil import tz

def cluster_characteristics(clustered_runs, save_directory='./', verbose=False):
    '''
    Analyzes the number of clusters by application and number of runs by cluster.
    Plots the CDF of both the metrics.

    Parameters
    ----------
    path_to_clusters: string
        Path to the parquet file created by `clustering` of the cluster info.
    save_path: string, optional
        Where to save the figure.
    verbose: boolean, optional
        For debugging and info on amount of info being collected.

    Returns
    -------
    None
    '''
    # 'Application' 'Operation' 'Cluster Number' 'Cluster Size' 'Filename'
    # Initialize plot
    fig, axes = plt.subplots(2,1)
    # Read the data
    df = clustered_runs
    # Collect Read Info
    operation = 'Read'
    if verbose:
        print('Analyzing %s clusters...'%operation)
    mask = df['Operation'] == operation 
    pos = np.flatnonzero(mask)
    tmp = df.iloc[pos]
    no_runs_by_cluster = []
    no_clusters_by_app = []
    apps = tmp['Application'].unique().tolist()
    for app in apps:
        mask = tmp['Application'] == app
        pos = np.flatnonzero(mask)
        tmp_app = tmp.iloc[pos]
        cluster_nos = tmp_app['Cluster Number'].unique().tolist()
        no_clusters_by_app.append(len(cluster_nos))
        for cluster_no in cluster_nos:
            no_runs_by_cluster.append(tmp_app.iloc[np.flatnonzero(tmp_app['Cluster Number'] == cluster_no)].shape[0])
    # Plot Read Info
    # First, number of cluster by applications
    if verbose:
        print('Plotting info of %s clusters...'%operation)
    median = np.median(no_clusters_by_app)-1
    if verbose:
        print("Median of %s clusters in the applications: %d"%(operation,median))
    bins = np.arange(0, int(math.ceil(max(no_clusters_by_app)))+1, 1)
    hist = np.histogram(no_clusters_by_app, bins=bins)[0]
    cdf = np.cumsum(hist)
    cdf = [x/cdf[-1] for x in cdf]
    axes[0].plot(bins[:-1], cdf, color='skyblue', linewidth=2, label=operation)
    axes[0].axvline(median, color='skyblue', zorder=0, linestyle='--', linewidth=2)
    # Second, number of runs by clusters
    median = np.median(no_runs_by_cluster)
    if verbose:
        print("Median of runs in the %s clusters: %d"%(operation,median))
    bins = np.arange(0, int(math.ceil(max(no_runs_by_cluster)))+1, 1)
    hist = np.histogram(no_runs_by_cluster, bins=bins)[0]
    cdf = np.cumsum(hist)
    cdf = [x/cdf[-1] for x in cdf]
    axes[1].plot(bins[:-1], cdf, color='skyblue', linewidth=2, label=operation)
    axes[1].axvline(median, color='skyblue', zorder=0, linestyle='--', linewidth=2)
    # Collect Write info
    operation = 'Write'
    if verbose:
        print('Analyzing %s clusters...'%operation)
    mask = df['Operation'] == operation
    pos = np.flatnonzero(mask)
    tmp = df.iloc[pos]
    no_runs_by_cluster = []
    no_clusters_by_app = []
    apps = tmp['Application'].unique().tolist()
    for app in apps:
        mask = tmp['Application'] == app
        pos = np.flatnonzero(mask)
        tmp_app = tmp.iloc[pos]
        cluster_nos = tmp_app['Cluster Number'].unique().tolist()
        no_clusters_by_app.append(len(cluster_nos))
        for cluster_no in cluster_nos:
            no_runs_by_cluster.append(tmp_app.iloc[np.flatnonzero(tmp_app['Cluster Number'] == cluster_no)].shape[0])
    # Plot Write Info
    # First, number of cluster by applications
    if verbose:
        print('Plotting info of %s clusters...'%operation)
    median = np.median(no_clusters_by_app)-1
    if verbose:
        print("Median of %s clusters in the applications: %d"%(operation,median))
    bins = np.arange(0, int(math.ceil(max(no_clusters_by_app)))+1, 1)
    hist = np.histogram(no_clusters_by_app, bins=bins)[0]
    cdf = np.cumsum(hist)
    cdf = [x/cdf[-1] for x in cdf]
    axes[0].plot(bins[:-1], cdf, color='maroon', linewidth=2, label=operation)
    axes[0].axvline(median, color='maroon', zorder=0, linestyle=':', linewidth=2)
    # Second, number of runs by clusters
    median = np.median(no_runs_by_cluster)
    if verbose:
        print("Median of runs in the %s clusters: %d"%(operation,median))
    bins = np.arange(0, int(math.ceil(max(no_runs_by_cluster)))+1, 1)
    hist = np.histogram(no_runs_by_cluster, bins=bins)[0]
    cdf = np.cumsum(hist)
    cdf = [x/cdf[-1] for x in cdf]
    axes[1].plot(bins[:-1], cdf, color='maroon', linewidth=2, label=operation)
    axes[1].axvline(median, color='maroon', zorder=0, linestyle=':', linewidth=2)
    # Overall figure aesthetics 
    axes[0].set_ylabel('CDF of Applications')
    axes[0].set_xlabel('Number of Clusters')
    axes[1].set_ylabel('CDF of Clusters')
    axes[1].set_xlabel('Number of Runs')
    axes[1].legend()
    axes[0].legend()
    axes[0].set_ylim(0,1)
    axes[1].set_ylim(0,1)
    axes[1].set_xlim(0,5000)
    axes[0].yaxis.grid(color='lightgrey', linestyle=':')
    axes[0].set_axisbelow(True)
    axes[1].yaxis.grid(color='lightgrey', linestyle=':')
    axes[1].set_axisbelow(True)
    vals = axes[0].get_yticks()
    axes[0].set_yticklabels(['{:,.0%}'.format(x) for x in vals])
    vals = axes[1].get_yticks()
    axes[1].set_yticklabels(['{:,.0%}'.format(x) for x in vals])
    fig.tight_layout()
    if not os.path.exists(save_directory):
        os.makedirs(save_directory)
    plt.savefig(join(save_directory,'cluster_characteristics.pdf'))
    return None

def general_temporal_trends(clustered_runs, save_directory='./', verbose=False):
    '''
    overall cluster temporal overlap, inter-arrival times variability and time 
    span, overall time spans of clusters, 

    Parameters
    ----------
    path_to_clusters: string
        Path to the parquet file created by `clustering` of the cluster info.
    path_to_total: string
        Path to Darshan logs parsed in the 'total' format. The logs should be
        sorted by user ID and executable name.
    ranks: int, optional
        Parallize the data collection by increasing the number of processes
        collecting and saving the data.
    save_path: string, optional
        Where to save the figure.
    verbose: boolean, optional
        For debugging and info on amount of info being collected.

    Returns
    -------
    None
    '''
    if not os.path.exists(save_directory):
        os.makedirs(save_directory)
    clustered_runs['End Time'] = clustered_runs['End Time'].astype(float)
    clustered_runs['Start Time'] = clustered_runs['Start Time'].astype(float)
    # Read
    operation = 'Read'
    read_df = pd.DataFrame()
    df = clustered_runs[clustered_runs['Operation']==operation]
    apps = df['Application'].unique()
    for app in apps:
        mask = df['Application'] == app
        pos = np.flatnonzero(mask)
        tmp_app = df.iloc[pos].sort_index().reset_index()
        c_nos = tmp_app['Cluster Number'].unique()
        for c_no in c_nos:
            mask = tmp_app['Cluster Number'] == c_no
            pos = np.flatnonzero(mask)
            tmp = tmp_app.iloc[pos].sort_index().reset_index()
            no_runs = tmp.shape[0]
            total_time = tmp['End Time'].max() - tmp['Start Time'].min()
            total_days              = total_time/86400
            time_differences = []
            for j in np.arange(0, no_runs-1):
                time_difference = abs(tmp.loc[j+1]['End Time']-tmp.loc[j]['Start Time'])
                time_differences.append(int(time_difference))
            time_differences_avg    = np.average(time_differences)
            time_differences_std    = np.std(time_differences)
            time_differences_cov    = (time_differences_std/time_differences_avg)*100
            read_df = read_df.append({'Cluster Number': c_no, 'Total Time': total_time, 'Average Runs per Day': no_runs/total_days,
            'Temporal Coefficient of Variation': time_differences_cov}, ignore_index=True)
    range = []
    for n in read_df['Total Time']:
        if(n<86400):
            range.append('<1d')
        elif(n<259200):
            range.append('1-\n3d')
        elif(n<604800):
            range.append('3d-\n1w')
        elif(n<(2592000/2)):
            range.append('1w-\n2w')
        elif(n<2592000):
            range.append('2w-\n1M')
        elif(n<7776000):
            range.append('1-\n3M')
        elif(n<15552000):
            range.append('3-\n6M')
        elif(n>15551999):
            range.append('6M+')
        else:
            continue
    read_df['Range'] = range
    # Write
    operation = 'Write'
    write_df = pd.DataFrame()
    df = clustered_runs[clustered_runs['Operation']==operation]
    apps = df['Application'].unique()
    for app in apps:
        mask = df['Application'] == app
        pos = np.flatnonzero(mask)
        tmp_app = df.iloc[pos].sort_index().reset_index()
        c_nos = tmp_app['Cluster Number'].unique()
        for c_no in c_nos:
            mask = tmp_app['Cluster Number'] == c_no
            pos = np.flatnonzero(mask)
            tmp = tmp_app.iloc[pos].sort_index().reset_index()
            no_runs = tmp.shape[0]
            total_time = tmp['End Time'].max() - tmp['Start Time'].min()
            total_days              = total_time/86400
            time_differences = []
            for j in np.arange(0, no_runs-1):
                time_difference = abs(tmp.loc[j+1]['End Time']-tmp.loc[j]['Start Time'])
                time_differences.append(int(time_difference))
            time_differences_avg    = np.average(time_differences)
            time_differences_std    = np.std(time_differences)
            time_differences_cov    = (time_differences_std/time_differences_avg)*100
            write_df = write_df.append({'Cluster Number': c_no, 'Total Time': total_time, 'Average Runs per Day': no_runs/total_days,
            'Temporal Coefficient of Variation': time_differences_cov}, ignore_index=True)
    range = []
    for n in write_df['Total Time']:
        if(n<86400):
            range.append('<1d')
        elif(n<259200):
            range.append('1-\n3d')
        elif(n<604800):
            range.append('3d-\n1w')
        elif(n<(2592000/2)):
            range.append('1w-\n2w')
        elif(n<2592000):
            range.append('2w-\n1M')
        elif(n<7776000):
            range.append('1-\n3M')
        elif(n<15552000):
            range.append('3-\n6M')
        elif(n>15551999):
            range.append('6M+')
        else:
            continue
    write_df['Range'] = range
    ########################################################### CDF of time periods and frequency ###########################################################
    fig, axes = plt.subplots(1, 2, sharey=True, figsize=[5, 2.2])
    fig.subplots_adjust(left=0.15, right=0.965, top=.94, bottom=0.34, wspace=0.05)
    read_info = read_df['Total Time']/86400
    write_info = write_df['Total Time']/86400
    read_median = np.median(read_info)
    write_median = np.median(write_info)
    read_info = np.log10(read_info)
    write_info = np.log10(write_info)
    read_median_plotting = np.median(read_info)
    write_median_plotting = np.median(write_info)
    read_bins = np.arange(0, int(math.ceil(max(read_info)))+1, 0.01)
    hist = np.histogram(read_info, bins=read_bins)[0]
    cdf_read = np.cumsum(hist)
    cdf_read = [x/cdf_read[-1] for x in cdf_read]
    write_bins = np.arange(0, int(math.ceil(max(write_info)))+1, 0.01)
    hist = np.histogram(write_info, bins=write_bins)[0]
    cdf_write = np.cumsum(hist)
    cdf_write = [x/cdf_write[-1] for x in cdf_write]
    axes[0].plot(read_bins[:-1], cdf_read, color='skyblue', linewidth=2, label='Read')
    axes[0].plot(write_bins[:-1], cdf_write, color='maroon', linewidth=2, label='Write')
    axes[0].set_ylabel('CDF of Clusters')
    axes[0].set_xlabel('(a) Cluster Time\nSpan (days)')
    axes[0].yaxis.grid(color='lightgrey', linestyle=':')
    axes[0].set_axisbelow(True)
    axes[0].set_ylim(0,1)
    axes[0].set_xlim(0,3)
    axes[0].set_yticks(np.arange(0,1.2,0.25))
    positions = [1, 2, 3]
    labels = ['$10^1$', '$10^2$', '$10^3$']
    axes[0].xaxis.set_major_locator(ticker.FixedLocator(positions))
    axes[0].xaxis.set_major_formatter(ticker.FixedFormatter(labels))
    vals = axes[0].get_yticks()
    axes[0].set_yticklabels(['{:,.0%}'.format(x) for x in vals])
    # Add minor ticks
    ticks = [1,2,3,4,5,6,7,8,9] 
    f_ticks = []
    tmp_ticks = [np.log10(x) for x in ticks]
    f_ticks = f_ticks + tmp_ticks
    tmp_ticks = [np.log10(x)+1 for x in ticks]
    f_ticks = f_ticks + tmp_ticks
    tmp_ticks = [np.log10(x)+2 for x in ticks]
    f_ticks = f_ticks + tmp_ticks
    axes[0].set_xticks(f_ticks, minor=True)
    # Add vertical lines for medians
    axes[0].axvline(read_median_plotting, color='skyblue', zorder=0, linestyle='--', linewidth=2)
    axes[0].axvline(write_median_plotting, color='maroon', zorder=0, linestyle=':', linewidth=2)
    # Add legend
    axes[0].legend(loc='lower right', fancybox=True)
    read_info = read_df['Average Runs per Day'].tolist()
    write_info = write_df['Average Runs per Day'].tolist()
    read_median = np.median(read_info)
    write_median = np.median(write_info)
    read_info = np.log10(read_info)
    write_info = np.log10(write_info)
    read_bins = np.arange(0, int(math.ceil(max(read_info)))+1, 0.01)
    hist = np.histogram(read_info, bins=read_bins)[0]
    cdf_read = np.cumsum(hist)
    cdf_read = [x/cdf_read[-1] for x in cdf_read]
    write_bins = np.arange(0, int(math.ceil(max(write_info)))+1, 0.01)
    hist = np.histogram(write_info, bins=write_bins)[0]
    cdf_write = np.cumsum(hist)
    cdf_write = [x/cdf_write[-1] for x in cdf_write]
    axes[1].plot(read_bins[:-1], cdf_read, color='skyblue', linewidth=2, label='Read')
    axes[1].plot(write_bins[:-1], cdf_write, color='maroon', linewidth=2, label='Write')
    axes[1].set_xlabel('(b) Run Frequency\n(runs/day)')
    axes[1].yaxis.grid(color='lightgrey', linestyle=':')
    axes[1].set_axisbelow(True)
    axes[1].set_ylim(0,1)
    axes[1].set_xlim(0,3)
    axes[1].set_yticks(np.arange(0,1.2,0.25))
    positions = [1, 2, 3]
    labels = ['$10^1$', '$10^2$', '$10^3$']
    axes[1].xaxis.set_major_locator(ticker.FixedLocator(positions))
    axes[1].xaxis.set_major_formatter(ticker.FixedFormatter(labels))
    vals = axes[0].get_yticks()
    axes[1].set_yticklabels(['{:,.0%}'.format(x) for x in vals])
    # Add minor ticks
    ticks = [1,2,3,4,5,6,7,8,9] 
    f_ticks = []
    tmp_ticks = [np.log10(x) for x in ticks]
    f_ticks = f_ticks + tmp_ticks
    tmp_ticks = [np.log10(x)+1 for x in ticks]
    f_ticks = f_ticks + tmp_ticks
    tmp_ticks = [np.log10(x)+2 for x in ticks]
    f_ticks = f_ticks + tmp_ticks
    axes[1].set_xticks(f_ticks, minor=True)
    # Add vertical lines for medians
    axes[1].axvline(np.log10(read_median), color='skyblue', zorder=0, linestyle='--', linewidth=2)
    axes[1].axvline(np.log10(write_median), color='maroon', zorder=0, linestyle=':', linewidth=2)
    # Add legend
    axes[0].legend(loc='lower right', fancybox=True)
    #axes[1].get_legend().remove()
    plt.savefig(join(save_directory, 'time_periods_freq.pdf'))
    plt.close()
    plt.clf()
    ########################################################### Temporal Variation by Length of Period ###########################################################
    rm = np.median(read_df[read_df['Range']=='1w-\n2w']['Temporal Coefficient of Variation'])
    wm = np.median(write_df[write_df['Range']=='1w-\n2w']['Temporal Coefficient of Variation'])
    # Barplot of time periods to temporal CoV
    fig, axes = plt.subplots(1, 2, sharey=True, figsize=[5,1.8])
    fig.subplots_adjust(left=0.19, right=0.990, top=0.96, bottom=0.48, wspace=0.03)
    order = ['<1d', '1-\n3d', '3d-\n1w', '1w-\n2w', '2w-\n1M', '1-\n3M', '3-\n6M']
    PROPS = {'boxprops':{'facecolor':'skyblue', 'edgecolor':'black'}, 'medianprops':{'color':'black'},
            'whiskerprops':{'color':'black'},'capprops':{'color':'black'}}
    sns.boxplot(ax=axes[0], x='Range', y='Temporal Coefficient of Variation', data=read_df, order=order, color='skyblue', fliersize=0, **PROPS)
    PROPS = {'boxprops':{'facecolor':'maroon', 'edgecolor':'black'}, 'medianprops':{'color':'white', 'linewidth': 1.25},
            'whiskerprops':{'color':'black'},'capprops':{'color':'black'}}
    sns.boxplot(ax=axes[1], x='Range', y='Temporal Coefficient of Variation', data=write_df, order=order,color='maroon', fliersize=0, **PROPS)
    # iterate over boxes
    for i,box in enumerate(axes[0].artists):
        box.set_edgecolor('black')
    axes[0].set_ylabel('')
    axes[1].set_ylabel('')
    fig.text(0.005, 0.48, 'Inter-arrival\nTimes CoV', rotation=90)
    axes[0].set_xlabel('')
    axes[1].set_xlabel('')
    fig.text(0.38, 0.13, '(a) Read', ha='center')
    fig.text(0.80, 0.13, '(b) Write', ha='center')
    fig.text(0.58, 0.03, 'Cluster Time Span', ha='center')
    #fig.text(0.001, 0.65, "Performance\nCoV (%)", rotation=90, va='center', multialignment='center')
    axes[1].yaxis.grid(color='lightgrey', linestyle=':')
    axes[1].set_axisbelow(True)
    axes[0].yaxis.grid(color='lightgrey', linestyle=':')
    axes[0].set_axisbelow(True)
    #axes[0].set_yticks([0,1000,2000,3000])
    iqr_r = stats.iqr(read_df['Temporal Coefficient of Variation'])
    iqr_w = stats.iqr(write_df['Temporal Coefficient of Variation'])
    q1_r = np.percentile(read_df['Temporal Coefficient of Variation'], 25)
    q3_r = np.percentile(read_df['Temporal Coefficient of Variation'], 75)
    E_low_outliers_r = q1_r-1.5*iqr_r
    E_high_outliers_r = q3_r+1.5*iqr_r
    non_outliers_r = [x for x in read_df['Temporal Coefficient of Variation'] if E_low_outliers_r<x<E_high_outliers_r]
    q1_w = np.percentile(write_df['Temporal Coefficient of Variation'], 25)
    q3_w = np.percentile(write_df['Temporal Coefficient of Variation'], 75)
    E_low_outliers_w = q1_w-1.5*iqr_w
    E_high_outliers_w = q3_w+1.5*iqr_w
    non_outliers_w = [x for x in write_df['Temporal Coefficient of Variation'] if E_low_outliers_w<x<E_high_outliers_w]
    non_outliers_max = [max(non_outliers_r), max(non_outliers_w)]
    top_ylim = math.ceil(max(non_outliers_max)/500)*500
    axes[0].set_ylim(0,top_ylim)
    plt.savefig(join(save_directory, 'time_period_v_temp_cov.pdf'))
    plt.close()
    plt.clf()
    ########################################################### Temporal Cluster Overlapping ###########################################################
    '''
    # Extract cluster info 
    cluster_info = pd.DataFrame()
    # For Read
    operation = 'Read'
    mask = clustered_runs['Operation'] == operation
    pos = np.flatnonzero(mask)
    tmp_o = clustered_runs.iloc[pos]
    applications = tmp_o['Application'].unique().tolist()
    for application in applications:
        mask = tmp_o['Application'] == application 
        pos = np.flatnonzero(mask)
        tmp_a = tmp_o.iloc[pos]
        c_nos = tmp_a['Cluster Number'].unique().tolist()
        for c_no in c_nos:
            mask = tmp_a['Cluster Number'] == c_no 
            pos = np.flatnonzero(mask)
            tmp = tmp_a.iloc[pos]
            start_time = tmp['Start Time'].min()
            end_time   = tmp['End Time'].max()
            total_time = end_time - start_time
            d = {'Application': application, 'Operation': operation, 'Cluster Number': c_no, 'Start Time': start_time, 'End Time': end_time, 'Total Time': total_time,'Overlap (%)': None}
            cluster_info = cluster_info.append(d, ignore_index=True)
    # For Write
    operation = 'Write'
    mask = clustered_runs['Operation'] == operation
    pos = np.flatnonzero(mask)
    tmp_o = clustered_runs.iloc[pos]
    applications = tmp_o['Application'].unique().tolist()
    for application in applications:
        mask = tmp_o['Application'] == application 
        pos = np.flatnonzero(mask)
        tmp_a = tmp_o.iloc[pos]
        c_nos = tmp_a['Cluster Number'].unique().tolist()
        for c_no in c_nos:
            mask = tmp_a['Cluster Number'] == c_no 
            pos = np.flatnonzero(mask)
            tmp = tmp_a.iloc[pos]
            start_time = tmp['Start Time'].min()
            end_time   = tmp['End Time'].max()
            total_time = end_time - start_time
            d = {'Application': application, 'Operation': operation, 'Cluster Number': c_no, 'Start Time': start_time, 'End Time': end_time, 'Total Time': total_time, 'Overlap (%)': None}
            cluster_info = cluster_info.append(d, ignore_index=True)
    # For Read clusters
    operation = 'Read'
    tmp_o = cluster_info[cluster_info['Operation']==operation]
    applications = tmp_o['Application'].unique().tolist()
    results = pd.DataFrame()
    for application in applications:
        mask = tmp_o['Application'] == application 
        pos = np.flatnonzero(mask)
        tmp_a = tmp_o.iloc[pos]
        print(tmp_a)
        for i, row0 in tmp_a.iterrows():

            t = {row0['Start Time']:row0['End Time']}
            # Find the intervals that do not overlap
            for j, row1 in tmp_a.iterrows():
                if(i==j):
                    continue
                try:
                    s_i = min(list(t.keys()))
                    e_i = max(list(t.values()))
                except ValueError:
                    continue
                s_j = row1['Start Time']
                e_j = row1['End Time']
                # First case: j spans through all of i
                for key, value in t.copy().items():
                    # this span will not be affect
                    if(value<s_j or key>e_j):
                        continue
                    # this span will be deleted
                    elif(key>s_j and value<e_j):
                        del t[key]
                    # this will have modified start
                    elif(key>s_j and key<e_j and value>e_j):
                        t[e_j] = value
                        del t[key]
                    # this will have modified end
                    elif(value>s_j and value<e_j and key<s_j):
                        t[key] = s_j
                    # other wise, divide the span
                    else:
                        t[key] = s_j
                        t[e_j] = value
            non_overlap = 0
            print(t)
            for key, value in t.items():
                non_overlap = non_overlap + (value-key)
            try:
                percent = ((non_overlap/row0['Total Time']))*100
            except ZeroDivisionError:
                percent = 100
            # Modify 'Overlap (%)'
            tmp_a.loc[i, 'Overlap (%)'] = percent
        results = results.append(tmp_a, ignore_index=True)
    '''
    '''
    # For Write clusters
    operation = 'Write'
    cluster_op_info = cluster_info[cluster_info['Operation']==operation]
    applications = cluster_op_info['Application'].unique().tolist()
    for application in applications:
        tmp = pd.DataFrame(columns=['Application', 'Operation', 'Cluster Number', 'Start Time (Hours)', 'End Time (Hours)', 'Total Time (Hours)', 'Overlap (%)'])
        clusters = cluster_op_info[cluster_op_info['Application']==application]
        no_clusters = 0
        n = clusters['Cluster Number'].max()
        clusters_start_time = clusters['Start Time'].min()
        for i in np.arange(0, n+1):
            cluster = clusters[clusters['Cluster Number']==i]
            cluster_size = cluster.shape[0]
            try:
                start_time = (cluster['Start Time'].min()-clusters_start_time)/3600
            except ZeroDivisionError:
                start_time = 0
            end_time = (cluster['End Time'].max()-clusters_start_time)/3600
            total_time = end_time - start_time
            d = {'Application': application, 'Operation': operation, 'Cluster Number': no_clusters, 'Start Time (Hours)': start_time, 
                'End Time (Hours)': end_time, 'Total Time (Hours)': total_time, 'Overlap (%)': None}
            tmp = tmp.append(d, ignore_index=True)
            no_clusters = no_clusters + 1
        #print('Application and size: %s %d'%(application, len(tmp)))
        for i, row0 in tmp.iterrows():
            t = {row0['Start Time (Hours)']:row0['End Time (Hours)']}
            #print(t)
            # Find the intervals that do not overlap
            for j, row1 in tmp.iterrows():
                if(i==j):
                    continue
                try:
                    s_i = min(list(t.keys()))
                    e_i = max(list(t.values()))
                except ValueError:
                    continue
                s_j = row1['Start Time (Hours)']
                e_j = row1['End Time (Hours)']
                # First case: j spans through all of i
                #if(s_i>s_j and e_i<e_j): # works
                #    t = {0:0}
                #    continue
                for key, value in t.copy().items():
                    # this span will not be affect
                    if(value<s_j or key>e_j):
                        continue
                    # this span will be deleted
                    elif(key>s_j and value<e_j):
                        del t[key]
                    # this will have modified start
                    elif(key>s_j and key<e_j and value>e_j):
                        t[e_j] = value
                        del t[key]
                    # this will have modified end
                    elif(value>s_j and value<e_j and key<s_j):
                        t[key] = s_j
                    # other wise, divide the span
                    else:
                        t[key] = s_j
                        t[e_j] = value
                    #print(t)
            non_overlap = 0
            for key, value in t.items():
                non_overlap = non_overlap + (value-key)
            try:
                percent = ((non_overlap/row0['Total Time (Hours)']))*100
            except ZeroDivisionError:
                percent = 100
            # Modify 'Overlap (%)'
            tmp.loc[i, 'Overlap (%)'] = percent
        results = results.append(tmp, ignore_index=True)
    '''
    #print(results)
    return None

def io_performance_variability(clustered_runs, save_directory='./'):
    '''
    overall I/O performance variability, day of week, time of day, time span, 
    I/O amount, check is cluster size is factor, temporal phasing.
    '''
    if not os.path.exists(save_directory):
        os.makedirs(save_directory)
    cluster_info = pd.DataFrame() # Average I/O Amount (bytes), Performance CoV (%), Number of Runs, Total Time
    clustered_runs['End Time'] = clustered_runs['End Time'].astype(float)
    clustered_runs['Start Time'] = clustered_runs['Start Time'].astype(float)
    # Read
    operation = 'Read'
    df = clustered_runs[clustered_runs['Operation']==operation]
    apps = df['Application'].unique()
    for app in apps:
        mask = df['Application'] == app
        pos = np.flatnonzero(mask)
        tmp_app = df.iloc[pos].sort_index().reset_index()
        c_nos = tmp_app['Cluster Number'].unique()
        for c_no in c_nos:
            mask = tmp_app['Cluster Number'] == c_no
            pos = np.flatnonzero(mask)
            tmp = tmp_app.iloc[pos].sort_index().reset_index()
            no_runs = tmp.shape[0]
            total_time = tmp['End Time'].max() - tmp['Start Time'].min()
            perf_avg    = np.average(tmp['Performance'])
            perf_std    = np.std(tmp['Performance']) 
            perf_cov    = (perf_std/perf_avg)*100
            avg_io      = np.average(tmp['I/O Amount'])
            cluster_info = cluster_info.append({'Cluster Number': c_no, 'Application': app, 'Operation': operation, 'Total Time': total_time, 'Number of Runs': no_runs, 
            'Average I/O Amount (bytes)': avg_io, 'Performance CoV (%)': perf_cov}, ignore_index=True)
    # Write
    operation = 'Write'
    df = clustered_runs[clustered_runs['Operation']==operation]
    apps = df['Application'].unique()
    for app in apps:
        mask = df['Application'] == app
        pos = np.flatnonzero(mask)
        tmp_app = df.iloc[pos].sort_index().reset_index()
        c_nos = tmp_app['Cluster Number'].unique()
        for c_no in c_nos:
            mask = tmp_app['Cluster Number'] == c_no
            pos = np.flatnonzero(mask)
            tmp = tmp_app.iloc[pos].sort_index().reset_index()
            no_runs = tmp.shape[0]
            total_time = tmp['End Time'].max() - tmp['Start Time'].min()
            perf_avg    = np.average(tmp['Performance'])
            perf_std    = np.std(tmp['Performance'])
            perf_cov    = (perf_std/perf_avg)*100
            avg_io      = np.average(tmp['I/O Amount'])
            cluster_info = cluster_info.append({'Cluster Number': c_no, 'Application': app, 'Operation': operation, 'Total Time': total_time, 'Number of Runs': no_runs,
            'Average I/O Amount (bytes)': avg_io, 'Performance CoV (%)': perf_cov}, ignore_index=True)
    ########################################################### Day of Week v. Performance ###########################################################
    df = clustered_runs
    E = 0.10
    fig, axes = plt.subplots(2, 1, sharex=True, figsize=[5, 2.2])
    fig.subplots_adjust(left=0.14, right=0.75, top=.90, bottom=0.20, hspace=0.35)
    range_tod = []
    range_tow = []
    from_zone = tz.gettz('UTC')
    to_zone = tz.gettz('America/Chicago')
    order = ['Mo','Tu', 'We', 'Th', 'Fr', 'Sa', 'Su']
    for n in df['Start Time']:
        datetime_time = datetime.fromtimestamp(n).replace(tzinfo=from_zone).astimezone(to_zone)
        h             = int(datetime_time.hour)
        d             = int(datetime_time.weekday())
        # Now for time of week
        if(d == 0):
            range_tow.append('Mo')
        elif(d == 1):
            range_tow.append('Tu')
        elif(d == 2):
            range_tow.append('We')
        elif(d == 3):
            range_tow.append('Th')
        elif(d == 4):
            range_tow.append('Fr')
        elif(d == 5):
            range_tow.append('Sa')
        elif(d == 6):
            range_tow.append('Su')
        else:
            print("don't forget: %d"%n)
    df['Range, Time of Week'] = range_tow
    # Read
    # Find performance variation
    operation = 'Read'
    cluster_op_info = df[df['Operation']==operation]
    applications = cluster_op_info['Application'].unique().tolist()
    results = pd.DataFrame()
    for application in applications:
        clusters = cluster_op_info[cluster_op_info['Application']==application]
        n = clusters['Cluster Number'].max()
        clusters_start_time = clusters['Start Time'].min()
        for i in np.arange(0, n+1):
            perf_covs = []
            cluster = clusters[clusters['Cluster Number']==i]
            cluster_size = cluster.shape[0]
            if(cluster_size<40):
                continue
            perf_cov = stats.variation(cluster['Performance'])
            for n in np.arange(0,cluster_size):
                perf_covs.append(perf_cov)
            #cluster['Performance CoV'] = perf_covs
            cluster.loc[:,'Performance CoV'] = perf_covs
            results = results.append(cluster, ignore_index=True)
    E_25 = int(results.shape[0]*E)
    l = results.nsmallest(E_25, ['Performance CoV'])
    l['Performance CoV Percentile'] = E*100
    h = results.nlargest(E_25, ['Performance CoV'])
    h['Performance CoV Percentile'] = 100-E*100
    results = h.append(l, ignore_index=True)
    #print(results)
    TODs = results['Range, Time of Week'].unique().tolist()
    plot = pd.DataFrame()
    for TOD in TODs:
        count = l[l['Range, Time of Week']==TOD].shape[0]
        d = {'DOW': TOD, 'Count': int(count), 'Performance CoV Percentile': 'Bottom 10%'}
        plot = plot.append(d, ignore_index=True)
        count = h[h['Range, Time of Week']==TOD].shape[0]
        d = {'DOW': TOD, 'Count': int(count), 'Performance CoV Percentile': 'Top 10%'}
        plot = plot.append(d, ignore_index=True)
    sns.barplot(ax=axes[0], data=plot, x='DOW', y='Count', hue='Performance CoV Percentile', edgecolor='black', linewidth=2, palette='Blues', order=order)
    # Write
    # Find performance variation
    operation = 'Write'
    cluster_op_info = df[df['Operation']==operation]
    applications = cluster_op_info['Application'].unique().tolist()
    results = pd.DataFrame()
    for application in applications:
        clusters = cluster_op_info[cluster_op_info['Application']==application]
        n = clusters['Cluster Number'].max()
        clusters_start_time = clusters['Start Time'].min()
        for i in np.arange(0, n+1):
            perf_covs = []
            cluster = clusters[clusters['Cluster Number']==i]
            cluster_size = cluster.shape[0]
            if(cluster_size<40):
                continue
            perf_cov = stats.variation(cluster['Performance'])
            for n in np.arange(0,cluster_size):
                perf_covs.append(perf_cov)
            #cluster['Performance CoV'] = perf_covs
            cluster.loc[:,'Performance CoV'] = perf_covs
            results = results.append(cluster, ignore_index=True)
    E_25 = int(results.shape[0]*E)
    l = results.nsmallest(E_25, ['Performance CoV'])
    l['Performance CoV Percentile'] = E*100
    h = results.nlargest(E_25, ['Performance CoV'])
    h['Performance CoV Percentile'] = 100-E*100
    results = h.append(l, ignore_index=True)
    #print(results)
    TODs = results['Range, Time of Week'].unique().tolist()
    plot = pd.DataFrame()
    for TOD in TODs:
        count = l[l['Range, Time of Week']==TOD].shape[0]
        d = {'DOW': TOD, 'Count': int(count), 'Performance CoV Percentile': 'Bottom 10%'}
        plot = plot.append(d, ignore_index=True)
        count = h[h['Range, Time of Week']==TOD].shape[0]
        d = {'DOW': TOD, 'Count': int(count), 'Performance CoV Percentile': 'Top 10%'}
        plot = plot.append(d, ignore_index=True)
    print(plot)
    sns.barplot(ax=axes[1], data=plot, x='DOW', y='Count', hue='Performance CoV Percentile', edgecolor='black', linewidth=2, palette='Reds', order=order)
    # Plot aesthetics
    axes[0].tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
    axes[1].tick_params(axis='x', labelrotation = 0)
    axes[0].yaxis.grid(color='lightgrey', linestyle=':')
    axes[0].set_axisbelow(True)
    axes[1].yaxis.grid(color='lightgrey', linestyle=':')
    axes[1].set_axisbelow(True)
    top = round(plot['Count'].max()+500, -3)
    axes[0].set_ylim(0,top)
    axes[1].set_ylim(0,top)
    #axes[0].set_yticks([x for x in np.arange(0,4001,2000)])
    #axes[1].set_yticks([x for x in np.arange(0,4001,2000)])
    fig.text(0.45, 0.02, 'Day of Week', ha='center')
    fig.text(0.001, 0.54, "Number of Runs", rotation=90, va='center')
    fig.text(0.35, 0.93, '(a) Read')
    fig.text(0.35, 0.515, '(b) Write')
    axes[0].legend(loc=(1.01,0.1),fontsize=8, title='Perf CoV Percentile', title_fontsize=8)
    axes[1].legend(loc=(1.01,0.1),fontsize=8, title='Perf CoV Percentile', title_fontsize=8)
    axes[0].set_ylabel('')
    axes[1].set_ylabel('')
    axes[0].set_xlabel('')
    axes[1].set_xlabel('')
    #axes[1].set_xticklabels(labels)
    plt.savefig(join(save_directory,'performance_cov_percentile_features_DOW.pdf'))
    plt.clf()
    plt.close()
    ########################################################### I/O Amount v. Performance ###########################################################
    df = cluster_info
    range = []
    for n in df['Average I/O Amount (bytes)']:
        if(n<100000000):
            range.append('<100M')
        elif(100000000<n<500000000):
            range.append('100M-\n500M')
        elif(500000000<n<1000000000):
            range.append('500M-\n1G')
        elif(1000000000<n<1500000000):
            range.append('1G-\n1.5G')
        elif(1500000000<n):
            range.append('1.5G+')
        else:
            print("don't forget: %d"%n)
    df['Range'] = range
    fig, axes = plt.subplots(1, 2, sharey=True, figsize=[5,2])
    fig.subplots_adjust(left=0.12, right=0.990, top=0.96, bottom=0.41, wspace=0.03)
    df_read  = df[df['Operation']=='Read']
    df_write = df[df['Operation']=='Write']
    print('I/O Amount vs. Performance Info...')
    m = np.median(df_read[df_read['Range']=='<100M']['Performance CoV (%)'])
    print('Read median, <100mb: %.3f'%m)
    m = np.median(df_read[df_read['Range']=='1.5G+']['Performance CoV (%)'])
    print('Read median, >1.5G: %.3f'%m)
    m = np.median(df_write[df_write['Range']=='<100M']['Performance CoV (%)'])
    print('Write median, <100mb: %.3f'%m)
    m = np.median(df_write[df_write['Range']=='1.5G+']['Performance CoV (%)'])
    print('Write median, >1.5G: %.3f'%m)
    order = ['<100M', '100M-\n500M', '500M-\n1G', '1G-\n1.5G', '1.5G+']
    PROPS = {'boxprops':{'facecolor':'skyblue', 'edgecolor':'black'}, 'medianprops':{'color':'black'},
            'whiskerprops':{'color':'black'},'capprops':{'color':'black'}}
    sns.boxplot(ax=axes[0], x='Range', y='Performance CoV (%)', data=df_read, order=order, color='skyblue', fliersize=0, **PROPS)
    PROPS = {'boxprops':{'facecolor':'maroon', 'edgecolor':'black'}, 'medianprops':{'color':'white', 'linewidth': 1.25},
            'whiskerprops':{'color':'black'},'capprops':{'color':'black'}}
    sns.boxplot(ax=axes[1], x='Range', y='Performance CoV (%)', data=df_write, order=order,color='maroon', fliersize=0, **PROPS)
    axes[0].set_ylabel('')
    axes[1].set_ylabel('')
    axes[0].set_xlabel('')
    axes[1].set_xlabel('')
    fig.text(0.33, 0.13, '(a) Read', ha='center')
    fig.text(0.77, 0.13, '(b) Write', ha='center')
    fig.text(0.58, 0.02, 'I/O Amount (bytes)', ha='center')
    fig.text(0.001, 0.55, "Performance CoV (%)", rotation=90, va='center', multialignment='center')
    axes[0].yaxis.grid(color='lightgrey', linestyle=':')
    axes[1].yaxis.grid(color='lightgrey', linestyle=':')
    axes[0].set_axisbelow(True)
    axes[1].set_axisbelow(True)
    #top = round(df['Performance CoV (%)'].max()+50,-2)
    y='Performance CoV (%)'
    iqr_r = stats.iqr(df_read[y])
    q1_r = np.percentile(df_read[y], 25)
    q3_r = np.percentile(df_read[y], 75)
    E_low_outliers_r = q1_r-1.5*iqr_r
    E_high_outliers_r = q3_r+1.5*iqr_r
    non_outliers_r = [x for x in df_read[y] if E_low_outliers_r<x<E_high_outliers_r]
    iqr_w = stats.iqr(df_write[y])
    q1_w = np.percentile(df_write[y], 25)
    q3_w = np.percentile(df_write[y], 75)
    E_low_outliers_w = q1_w-1.5*iqr_w
    E_high_outliers_w = q3_w+1.5*iqr_w
    non_outliers_w = [x for x in df_write[y] if E_low_outliers_w<x<E_high_outliers_w]
    top = max([max(non_outliers_r),max(non_outliers_w)])
    axes[0].set_ylim(0,top)
    axes[0].tick_params(axis='x', labelsize=10)
    axes[1].tick_params(axis='x', labelsize=10)
    plt.savefig(join(save_directory, 'info_amount.pdf'))
    plt.close()
    plt.clf()
    ########################################################### Time Span v. Performance ###########################################################
    df = cluster_info
    range = []
    for n in df['Total Time']:
        if(n<86400):
            range.append('<1d')
        elif(n<259200):
            range.append('1-3d')
        elif(n<604800):
            range.append('3d-1w')
        elif(n<(2592000/2)):
            range.append('1w-2w')
        elif(n<2592000):
            range.append('2w-1M')
        elif(n<7776000):
            range.append('1-3M')
        elif(n<15552000):
            range.append('3-6M')
        elif(n>15551999):
            range.append('6M+')
        else:
            print("don't forget: %d"%n)
    df['Range'] = range
    read_df = df[df['Operation']=='Read']
    write_df = df[df['Operation']=='Write']
    range_labels = read_df['Range'].unique()
    '''
    print("For Read:")
    for range_label in range_labels:
        print('Range Label, Number of Clusters: %s %d'%(range_label, len(read_df[read_df['Range']==range_label])))
    range_labels = write_df['Range'].unique()
    print('For Write')
    for range_label in range_labels:
        print('Range Label, Number of Clusters: %s %d'%(range_label, len(write_df[write_df['Range']==range_label])))
    '''
    # Barplot of time periods to performance CoV
    fig, axes = plt.subplots(1, 2, sharey=True, figsize=[5,2])
    fig.subplots_adjust(left=0.12, right=0.990, top=0.96, bottom=0.45, wspace=0.03)
    order = ['<1d', '1-3d', '3d-1w', '1w-2w', '2w-1M', '1-3M', '3-6M']
    labels = ['<1d', '1-\n3d', '3d-\n1w', '1w-\n2w', '2w-\n1M', '1-\n3M', '3-\n6M']
    PROPS = {'boxprops':{'facecolor':'skyblue', 'edgecolor':'black'}, 'medianprops':{'color':'black'},
            'whiskerprops':{'color':'black'},'capprops':{'color':'black'}}
    sns.boxplot(ax=axes[0], x='Range', y='Performance CoV (%)', data=read_df, order=order, fliersize=0, **PROPS)
    PROPS = {'boxprops':{'facecolor':'maroon', 'edgecolor':'black'}, 'medianprops':{'color':'white'},
            'whiskerprops':{'color':'black'},'capprops':{'color':'black'}}
    sns.boxplot(ax=axes[1], x='Range', y='Performance CoV (%)', data=write_df, order=order, linewidth=1.2, fliersize=0, **PROPS)
    axes[0].set_ylabel('')
    axes[1].set_ylabel('')
    axes[0].set_xlabel('')
    axes[1].set_xlabel('')
    axes[0].set_xticklabels(labels)
    axes[1].set_xticklabels(labels)
    fig.text(0.37, 0.12, '(a) Read', ha='center')
    fig.text(0.78, 0.12, '(b) Write', ha='center')
    fig.text(0.58, 0.02, 'Cluster Time Span', ha='center')
    fig.text(0.001, 0.55, "Performance CoV (%)", rotation=90, va='center', multialignment='center')
    axes[0].yaxis.grid(color='lightgrey', linestyle=':')
    axes[1].yaxis.grid(color='lightgrey', linestyle=':')
    axes[0].set_axisbelow(True)
    axes[1].set_axisbelow(True)
    #top = round(df['Performance CoV (%)'].max()+50,-2)
    y='Performance CoV (%)'
    iqr_r = stats.iqr(read_df[y])
    q1_r = np.percentile(read_df[y], 25)
    q3_r = np.percentile(read_df[y], 75)
    E_low_outliers_r = q1_r-1.5*iqr_r
    E_high_outliers_r = q3_r+1.5*iqr_r
    non_outliers_r = [x for x in read_df[y] if E_low_outliers_r<x<E_high_outliers_r]
    iqr_w = stats.iqr(write_df[y])
    q1_w = np.percentile(write_df[y], 25)
    q3_w = np.percentile(write_df[y], 75)
    E_low_outliers_w = q1_w-1.5*iqr_w
    E_high_outliers_w = q3_w+1.5*iqr_w
    non_outliers_w = [x for x in write_df[y] if E_low_outliers_w<x<E_high_outliers_w]
    top = max([max(non_outliers_r),max(non_outliers_w)])
    axes[0].set_ylim(0,top)
    plt.savefig(join(save_directory, 'time_period_v_perf_cov.pdf'))
    plt.close()
    plt.clf()
    ########################################################### Number of Runs v. Performance ###########################################################
    df = cluster_info
    range = []
    for n in df['Number of Runs']:
        # 40-60, 60-100, 100-200, 200-500, 500-1000, 1000+
        if(n<60):
            range.append('40-\n60')
        elif(n<100):
            range.append('60-\n100')
        elif(n<200):
            range.append('100-\n200')
        elif(n<500):
            range.append('200-\n500')
        elif(n<1000):
            range.append('500-\n1000')
        elif(n>=1000):
            range.append('1000+')
        else:
            print("don't forget: %d"%n)
    df['Range'] = range
    df_read = df[df['Operation']=='Read']
    df_write = df[df['Operation']=='Write']
    corr_read = df_read['Performance CoV (%)'].corr(df_read['Number of Runs'])
    corr_write = df_write['Performance CoV (%)'].corr(df_write['Number of Runs'])
    print("Info for number of runs vs. performance...")
    print('Spearman correlation of read: %f'%corr_read)
    print('Spearman correlation of write: %f'%corr_write)
    # Barplot of time periods to performance CoV
    fig, axes = plt.subplots(1, 2, sharey=True, figsize=[5,2])
    fig.subplots_adjust(left=0.11, right=0.990, top=0.96, bottom=0.42, wspace=0.03)
    order = ['40-\n60', '60-\n100', '100-\n200', '200-\n500', '500-\n1000', '1000+']
    PROPS = {'boxprops':{'facecolor':'skyblue', 'edgecolor':'black'}, 'medianprops':{'color':'black'},
            'whiskerprops':{'color':'black'},'capprops':{'color':'black'}}
    sns.boxplot(ax=axes[0], x='Range', y='Performance CoV (%)', data=df_read, order=order, color='skyblue', fliersize=0, **PROPS)
    PROPS = {'boxprops':{'facecolor':'maroon', 'edgecolor':'black'}, 'medianprops':{'color':'white', 'linewidth': 1.25},
            'whiskerprops':{'color':'black'},'capprops':{'color':'black'}}
    sns.boxplot(ax=axes[1], x='Range', y='Performance CoV (%)', data=df_write, order=order,color='maroon', fliersize=0, **PROPS)
    axes[0].set_ylabel('')
    axes[1].set_ylabel('')
    axes[0].set_xlabel('')
    axes[1].set_xlabel('')
    fig.text(0.33, 0.12, '(a) Read', ha='center')
    fig.text(0.77, 0.12, '(b) Write', ha='center')
    fig.text(0.55, 0.02, 'Cluster Size', ha='center')
    fig.text(0.001, 0.55, "Performance CoV (%)", rotation=90, va='center', multialignment='center')
    axes[0].yaxis.grid(color='lightgrey', linestyle=':')
    axes[1].yaxis.grid(color='lightgrey', linestyle=':')
    axes[0].set_axisbelow(True)
    axes[1].set_axisbelow(True)
    axes[0].tick_params(axis='x', labelsize=11)
    axes[1].tick_params(axis='x', labelsize=11)
    #top = round(df['Performance CoV (%)'].max()+50,-2)
    y='Performance CoV (%)'
    iqr_r = stats.iqr(df_read[y])
    q1_r = np.percentile(df_read[y], 25)
    q3_r = np.percentile(df_read[y], 75)
    E_low_outliers_r = q1_r-1.5*iqr_r
    E_high_outliers_r = q3_r+1.5*iqr_r
    non_outliers_r = [x for x in df_read[y] if E_low_outliers_r<x<E_high_outliers_r]
    iqr_w = stats.iqr(df_write[y])
    q1_w = np.percentile(df_write[y], 25)
    q3_w = np.percentile(df_write[y], 75)
    E_low_outliers_w = q1_w-1.5*iqr_w
    E_high_outliers_w = q3_w+1.5*iqr_w
    non_outliers_w = [x for x in df_write[y] if E_low_outliers_w<x<E_high_outliers_w]
    top = max([max(non_outliers_r),max(non_outliers_w)])
    axes[0].set_ylim(0,top)
    plt.savefig(join(save_directory, 'no_runs_v_perf_cov.pdf'))
    plt.close()
    plt.clf()
    ########################################################### CDF of Performance CoV ###########################################################
    df = cluster_info
    df['Performance CoV (%)'] = np.log10(df['Performance CoV (%)'])
    read_info = df[df['Operation']=='Read']['Performance CoV (%)'].tolist()
    read_median = np.median(read_info)
    print('Info for CDF of performance CoVs...')
    print("Median of Read: %d"%10**read_median)
    read_mean   = np.mean(read_info)
    read_bins = np.arange(0, int(math.ceil(max(read_info)))+1, 0.01)
    hist = np.histogram(read_info, bins=read_bins)[0]
    cdf_read = np.cumsum(hist)
    cdf_read = [x/cdf_read[-1] for x in cdf_read]
    write_info = df[df['Operation']=='Write']['Performance CoV (%)'].tolist()
    write_median = np.median(write_info)
    print("Median of Write: %d"%10**write_median)
    write_mean   = np.mean(write_info)
    write_bins = np.arange(0, int(math.ceil(max(write_info)))+1, 0.01)
    hist = np.histogram(write_info, bins=write_bins)[0]
    cdf_write = np.cumsum(hist)
    cdf_write = [x/cdf_write[-1] for x in cdf_write]
    fig, ax = plt.subplots(1, 1, sharey=True, figsize=[5, 2])
    fig.subplots_adjust(left=0.28, right=0.80, top=.94, bottom=0.26, wspace=0.12)
    ax.plot(read_bins[:-1], cdf_read, color='skyblue', linewidth=2, label='Read')
    ax.plot(write_bins[:-1], cdf_write, color='maroon', linewidth=2, label='Write')
    ax.set_ylim(0,1)
    ax.set_yticks(np.arange(0,1.2,0.25))
    vals = ax.get_yticks()
    ax.set_yticklabels(['{:,.0%}'.format(x) for x in vals])
    ax.set_ylabel('CDF of Clusters')
    ax.set_xlabel('Performance CoV (%)')
    ax.yaxis.grid(color='lightgrey', linestyle=':')
    ax.set_axisbelow(True)
    ax.set_xlim(0, 3)
    positions = [0,1,2,3]
    labels = ['$10^0$', '$10^1$', '$10^2$', '$10^3$']
    ax.xaxis.set_major_locator(ticker.FixedLocator(positions))
    ax.xaxis.set_major_formatter(ticker.FixedFormatter(labels))
    # Add minor ticks
    ticks = [1,2,3,4,5,6,7,8,9] 
    f_ticks = []
    tmp_ticks = [np.log10(x) for x in ticks]
    f_ticks = f_ticks + tmp_ticks
    tmp_ticks = [np.log10(x)+1 for x in ticks]
    f_ticks = f_ticks + tmp_ticks
    tmp_ticks = [np.log10(x)+2 for x in ticks]
    f_ticks = f_ticks + tmp_ticks
    ax.set_xticks(f_ticks, minor=True)
    # Add vertical lines for medians
    ax.axvline(read_median, color='skyblue', zorder=0, linestyle='--', linewidth=2)
    ax.axvline(write_median, color='maroon', zorder=0, linestyle=':', linewidth=2)
    # Add legend
    ax.legend(loc='lower right', fancybox=True)
    plt.savefig(join(save_directory, 'covs_cluster.pdf'))
    plt.clf()
    plt.close()
    return None

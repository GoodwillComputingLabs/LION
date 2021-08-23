# Library for final plots used in the paper
# Created on: Jan 7, 2021
# Author: Emily Costa

import seaborn as sns
import matplotlib.pyplot as plt 
import pandas as pd
from os.path import join
import numpy as np
from scipy.stats import norm, stats, spearmanr
import pylab as pl
import matplotlib.ticker as mtick
import math
import matplotlib as mpl
from datetime import datetime, timezone
from dateutil import tz
from scipy.stats import zscore
from matplotlib import ticker
from scipy.signal import find_peaks
from matplotlib.lines import Line2D

mpl.use("pgf")
text_size = 14
plt.rcParams.update({'font.size': text_size})
plt.rc('xtick',labelsize=text_size)
plt.rc('ytick',labelsize=text_size)

preamble = [r'\usepackage{fontspec}',
            r'\usepackage{physics}']
params = {'font.family': 'serif',
        'text.usetex': True,
        'pgf.rcfonts': False,
        'pgf.texsystem': 'xelatex',
        'pgf.preamble': preamble}
mpl.rcParams.update(params)

def plot_run_spread_temporally(path_to_cluster_info, save_dir):
    '''
    Plots the temporal behavior of runs in the clusters.

    Parameters
    ----------
    path_to_cluster_info: string
        Path to csv file with clustering temporal run information.

    Returns
    -------
    None
    '''
    df = pd.read_csv(path_to_cluster_info, index_col=0)
    range = []
    for n in df['Total Time']:
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
        else:
            print("don't forget: %d"%n)
    df['Range'] = range
    read_df = df[df['Operation']=='Read']
    write_df = df[df['Operation']=='Write']
    rm = np.median(read_df[read_df['Range']=='1w-\n2w']['Temporal Coefficient of Variation'])
    wm = np.median(write_df[write_df['Range']=='1w-\n2w']['Temporal Coefficient of Variation'])
    print('Median for read at 1-2w: %.3f'%rm)
    print('Median for write at 1-2w: %.3f'%wm)
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
    fig.text(0.005, 0.45, 'Inter-arrival\nTimes CoV', rotation=90)
    axes[0].set_xlabel('')
    axes[1].set_xlabel('')
    fig.text(0.38, 0.13, '(a) Read', ha='center')
    fig.text(0.80, 0.13, '(b) Write', ha='center')
    fig.text(0.58, 0.03, 'Cluster Time Span', ha='center')
    #fig.text(0.001, 0.65, "Performance\nCoV (%)", rotation=90, va='center', multialignment='center')
    axes[0].yaxis.grid(color='lightgrey', linestyle=':')
    axes[1].yaxis.grid(color='lightgrey', linestyle=':')
    axes[0].set_axisbelow(True)
    axes[1].set_axisbelow(True)
    axes[0].set_yticks([0,1000,2000,3000])
    axes[0].set_ylim(0,3000)
    plt.savefig(join('./time_period_v_temp_cov.pdf'))
    plt.close()
    plt.clf()

def plot_run_spread_span_frequency(path_to_cluster_info, save_dir):
    '''
    Plots the temporal behavior of runs in the clusters.

    Parameters
    ----------
    path_to_cluster_info: string
        Path to csv file with clustering temporal run information.

    Returns
    -------
    None
    '''
    df = pd.read_csv(path_to_cluster_info, index_col=0)
    read_df = df[df['Operation']=='Read']
    write_df = df[df['Operation']=='Write']
    # CDF of time periods and frequency
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
    print(cdf_write[100])
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
    axes[0].axvline(np.log10(4), color='skyblue', zorder=0, linestyle='--', linewidth=2)
    axes[0].axvline(write_median_plotting, color='maroon', zorder=0, linestyle=':', linewidth=2)
    print("Median of Read: %f"%read_median)
    print("Median of Write: %f"%write_median)
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
    print("Median of Read: %f"%read_median)
    print("Median of Write: %f"%write_median)
    # Add legend
    axes[0].legend(loc='lower right', fancybox=True)
    #axes[1].get_legend().remove()
    plt.savefig(join(save_dir, 'time_periods_freq.pdf'))
    plt.close()
    plt.clf()

    return None

def plot_time_of_day_v_perf(path_to_data, save_dir):
    '''
    Plots time period effects on performance.

    Parameters
    ----------
    path_to_data: string

    Returns
    -------
    None
    '''
    df = pd.read_csv(path_to_data, index_col=0)
    range_tod = []
    range_tow = []
    from_zone = tz.gettz('UTC')
    to_zone = tz.gettz('America/Chicago')
    for n in df['Start Time']:
        datetime_time = datetime.fromtimestamp(n).replace(tzinfo=from_zone).astimezone(to_zone)
        h             = int(datetime_time.hour)
        d             = int(datetime_time.weekday())
        # Group by time of day
        '''
        if(h == 0 or h == 1 or h == 2):
            range_tod.append('12-\n3am')
        elif(h == 3 or h == 4 or h == 5):
            range_tod.append('3-\n6am')
        elif(h == 6 or h == 7 or h == 8):
            range_tod.append('6-\n9am')
        elif(h == 9 or h == 10 or h == 11):
            range_tod.append('9am-\n12pm')
        elif(h == 12 or h == 13 or h == 14):
            range_tod.append('12-\n3pm')
        elif(h == 15 or h == 16 or h == 17):
            range_tod.append('3-\n6pm')
        elif(h == 18 or h == 19 or h == 20):
            range_tod.append('6-\n9pm')
        elif(h == 21 or h == 22 or h == 23):
            range_tod.append('9pm-\n12am')
        else:
            print("don't forget: %d"%n)
        '''
        if(h == 0 or h == 1 or h == 2):
            range_tod.append('0-\n3')
        elif(h == 3 or h == 4 or h == 5):
            range_tod.append('3-\n6')
        elif(h == 6 or h == 7 or h == 8):
            range_tod.append('6-\n9')
        elif(h == 9 or h == 10 or h == 11):
            range_tod.append('9-\n12')
        elif(h == 12 or h == 13 or h == 14):
            range_tod.append('12-\n15')
        elif(h == 15 or h == 16 or h == 17):
            range_tod.append('15-\n18')
        elif(h == 18 or h == 19 or h == 20):
            range_tod.append('18-\n21')
        elif(h == 21 or h == 22 or h == 23):
            range_tod.append('21-\n24')
        else:
            print("don't forget: %d"%n)
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
    df['Range, Time of Day'] = range_tod
    df['Range, Time of Week'] = range_tow
    # Rid of outliers to make cleaner plots
    order = ['0-\n3', '3-\n6', '6-\n9', '9-\n12', '12-\n15', '15-\n18', '18-\n21', '21-\n24']
    df_tod = pd.DataFrame(columns=['Range, Time of Day', 'Operation', 'Performance Z-Score'])
    for tod in order:
        working_df = df[df['Range, Time of Day']==tod].reset_index(drop=True)
        working_df['Z-Score of Z-Scores'] = (working_df['Performance Z-Score'] - working_df['Performance Z-Score'].mean())/working_df['Performance Z-Score'].std(ddof=0)
        working_df = working_df[working_df['Z-Score of Z-Scores'] < 2]
        working_df = working_df[working_df['Z-Score of Z-Scores'] > -2]
        working_df = working_df.drop(labels=['Application', 'Cluster Number', 'Start Time', 'Range, Time of Week', 'Z-Score of Z-Scores'], axis='columns')
        df_tod = df_tod.append(working_df, ignore_index=True)
    df_tow = pd.DataFrame(columns=['Range, Time of Week', 'Operation', 'Performance Z-Score'])
    for tow in ['Mo', 'Tu', 'We', 'Th', 'Fr', 'Sa', 'Su']:
        working_df = df[df['Range, Time of Week']==tow].reset_index(drop=True)
        working_df['Z-Score of Z-Scores'] = (working_df['Performance Z-Score'] - working_df['Performance Z-Score'].mean())/working_df['Performance Z-Score'].std(ddof=0)
        working_df = working_df[working_df['Z-Score of Z-Scores'] < 2]
        working_df = working_df[working_df['Z-Score of Z-Scores'] > -2]
        working_df = working_df.drop(labels=['Application', 'Cluster Number', 'Start Time', 'Range, Time of Day', 'Z-Score of Z-Scores'], axis='columns')
        df_tow = df_tow.append(working_df, ignore_index=True)
    # Barplot of time of day to performance CoV
    read_df = df_tod[df_tod['Operation']=='Read']
    write_df = df_tod[df_tod['Operation']=='Write']
    fig, axes = plt.subplots(1, 2, sharey=True, figsize=[5,2])
    fig.subplots_adjust(left=0.16, right=0.990, top=0.96, bottom=0.45, wspace=0.03)
    #order = ['12-\n3am', '3-\n6am', '6-\n9am', '9am-\n12pm', '12-\n3pm', '3-\n6pm', '6-\n9pm', '9pm-\n12am']
    #PROPS = {'boxprops':{'facecolor':'skyblue', 'edgecolor':'black'}, 'medianprops':{'color':'black'},
    #        'whiskerprops':{'color':'black'},'capprops':{'color':'black'}}
    sns.violinplot(ax=axes[0], x='Range, Time of Day', y='Performance Z-Score', data=read_df, order=order, color='skyblue', inner='quartile', linewidth=2)
    sns.violinplot(ax=axes[1], x='Range, Time of Day', y='Performance Z-Score', data=write_df, order=order, color='maroon', inner='quartile', linewidth=2)
    #violins = [art for art in axes[0].get_children()]
    #for i in range(len(violins)):
    #    violins[i].set_edgecolor('black')
    axes[0].set_ylabel('')
    axes[1].set_ylabel('')
    fig.text(0.37, 0.14, '(a) Read', ha='center')
    fig.text(0.78, 0.14, '(b) Write', ha='center')
    fig.text(0.58, 0.02, 'Time of Day (24-hr)', ha='center')
    fig.text(0.001, 0.65, "Performance\nZ-Score", rotation=90, va='center', multialignment='center')
    axes[0].set_xlabel('')
    axes[1].set_xlabel('')
    axes[0].yaxis.grid(color='lightgrey', linestyle=':')
    axes[1].yaxis.grid(color='lightgrey', linestyle=':')
    axes[0].set_axisbelow(True)
    axes[1].set_axisbelow(True)
    axes[0].set_ylim(-3,3)
    axes[0].set_yticks(range(-3,4,1))
    axes[0].tick_params(axis='x', labelsize=13)
    axes[1].tick_params(axis='x', labelsize=13)
    axes[0].tick_params(axis='y', labelsize=14)
    axes[1].tick_params(axis='y', labelsize=14)
    for l in axes[0].lines:
        l.set_linestyle('--')
        l.set_linewidth(0.6)
        l.set_color('black')
        l.set_alpha(0.8)
    for l in axes[0].lines[1::3]:
        l.set_linestyle('-')
        l.set_linewidth(1.2)
        l.set_color('black')
        l.set_alpha(0.8)
    for l in axes[1].lines:
        l.set_linestyle('--')
        l.set_linewidth(0.6)
        l.set_color('white')
        l.set_alpha(0.8)
    for l in axes[1].lines[1::3]:
        l.set_linestyle('-')
        l.set_linewidth(1.2)
        l.set_color('white')
        l.set_alpha(0.8)
    plt.savefig(join(save_dir, 'time_day_v_perf.pdf'))
    plt.close()
    plt.clf()
    # Barplot of time of week to performance CoV
    read_df = df_tow[df_tow['Operation']=='Read']
    write_df = df_tow[df_tow['Operation']=='Write']
    fig, axes = plt.subplots(1, 2, sharey=True, figsize=[5,1.9])
    fig.subplots_adjust(left=0.16, right=0.990, top=0.96, bottom=0.38, wspace=0.03)
    order = ['Mo', 'Tu', 'We', 'Th', 'Fr', 'Sa', 'Su']
    sns.violinplot(ax=axes[0], x='Range, Time of Week', y='Performance Z-Score', data=read_df, order=order, color='skyblue', inner='quartile', edgecolor='black')
    sns.violinplot(ax=axes[1], x='Range, Time of Week', y='Performance Z-Score', data=write_df, order=order, color='maroon', inner='quartile')
    axes[0].set_ylabel('')
    axes[1].set_ylabel('')
    fig.text(0.37, 0.135, '(a) Read', ha='center')
    fig.text(0.78, 0.135, '(b) Write', ha='center')
    fig.text(0.58, 0.02, 'Day of Week', ha='center')
    fig.text(0.001, 0.65, "Performance\nZ-Score", rotation=90, va='center', multialignment='center')
    axes[0].set_xlabel('')
    axes[1].set_xlabel('')
    axes[0].yaxis.grid(color='lightgrey', linestyle=':')
    axes[1].yaxis.grid(color='lightgrey', linestyle=':')
    axes[0].set_axisbelow(True)
    axes[1].set_axisbelow(True)
    axes[0].set_ylim(-3,3)
    axes[0].set_yticks(range(-3,4,1))
    axes[0].tick_params(axis='x', labelsize=14)
    axes[1].tick_params(axis='x', labelsize=14)
    axes[0].tick_params(axis='y', labelsize=14)
    axes[1].tick_params(axis='y', labelsize=14)
    for l in axes[0].lines:
        l.set_linestyle('--')
        l.set_linewidth(0.6)
        l.set_color('black')
        l.set_alpha(0.8)
    for l in axes[0].lines[1::3]:
        l.set_linestyle('-')
        l.set_linewidth(1.2)
        l.set_color('black')
        l.set_alpha(0.8)
    for l in axes[1].lines:
        l.set_linestyle('--')
        l.set_linewidth(0.6)
        l.set_color('white')
        l.set_alpha(0.8)
    for l in axes[1].lines[1::3]:
        l.set_linestyle('-')
        l.set_linewidth(1.2)
        l.set_color('white')
        l.set_alpha(0.8)
    plt.savefig(join(save_dir, 'time_week_v_perf.pdf'), backend='pgf')
    plt.close()
    plt.clf()
    return None

def plot_no_user_app_characterizations(path_to_data, path_to_normal_data, save_dir):
    '''
    Plots the cluster characterizations of clusters formed without being separated
    by user and application.

    Parameters
    ----------
    path_to_data: string
        Path to directory with data from clusters without user/app sorting.
    path_to_normal_data: string
        Path to directory with data from clusters with user/app sorting.
    save_dir: string
        Path to the directory to save the plots in.

    Returns
    -------
    None
    '''
    # Plot CoV of cluster sizes
    path = join(path_to_data, 'no_runs_in_clusters_read.txt')
    with open(path, 'r') as f:
        no_read_clusters = f.read().split("\n")
        f.close()
    no_read_clusters = pd.Series(no_read_clusters).astype(int)
    no_read_clusters = no_read_clusters[no_read_clusters > 40]
    path = join(path_to_data, 'no_runs_in_clusters_write.txt')
    with open(path, 'r') as f:
        no_write_clusters = f.read().split("\n")
        f.close()
    no_write_clusters = pd.Series(no_write_clusters).astype(int)
    no_write_clusters = no_write_clusters[no_write_clusters > 40]
    path = join(path_to_normal_data, 'no_runs_in_clusters_read.txt')
    with open(path, 'r') as f:
        no_read_clusters_o = f.read().split("\n")
        f.close()
    no_read_clusters_o = pd.Series(no_read_clusters_o).astype(int)
    no_read_clusters_o = no_read_clusters_o[no_read_clusters_o > 40]
    path = join(path_to_normal_data, 'no_runs_in_clusters_write.txt')
    with open(path, 'r') as f:
        no_write_clusters_o = f.read().split("\n")
        f.close()
    no_write_clusters_o = pd.Series(no_write_clusters_o).astype(int)
    no_write_clusters_o = no_write_clusters_o[no_write_clusters_o > 40]
    fig, axes = plt.subplots(1, 2, sharey=True, figsize=[12,5])
    fig.subplots_adjust(left=0.075, right=0.992, top=0.97, bottom=0.12, wspace=0.07)
    n_bins = 10000
    plt.setp(axes, xlim=(40,3000))
    hist = np.histogram(no_read_clusters, bins=range(max(no_read_clusters)+1))[0]
    cdf_read = np.cumsum(hist)
    cdf_read = [x/cdf_read[-1] for x in cdf_read]
    hist = np.histogram(no_write_clusters, bins=range(max(no_write_clusters)+1))[0]
    cdf_write = np.cumsum(hist)
    cdf_write = [x/cdf_write[-1] for x in cdf_write]
    hist = np.histogram(no_read_clusters_o, bins=range(max(no_read_clusters_o)+1))[0]
    cdf_read_o = np.cumsum(hist)
    cdf_read_o = [x/cdf_read_o[-1] for x in cdf_read_o]
    hist = np.histogram(no_write_clusters_o, bins=range(max(no_write_clusters_o)+1))[0]
    cdf_write_o = np.cumsum(hist)
    cdf_write_o = [x/cdf_write_o[-1] for x in cdf_write_o]
    axes[0].plot(cdf_read, color='skyblue', label='False', linewidth=4)
    axes[0].plot(cdf_read_o, color='mediumseagreen', label='True', linewidth=4, linestyle='--')
    axes[1].plot(cdf_write, color='maroon', label='False', linewidth=4)
    axes[1].plot(cdf_write_o, color='gold', label='True', linewidth=4, linestyle='--')
    axes[0].set_ylim(0,1)
    vals = axes[0].get_yticks()
    axes[0].set_yticklabels(['{:,.0%}'.format(x) for x in vals])
    axes[0].set_ylabel('Percent of Clusters')
    axes[0].set_xlabel('Number of Runs in a Read Cluster')
    axes[1].set_xlabel('Number of Runs in a Write Cluster')
    axes[0].legend(title='Clustered by Application', loc='lower right')
    axes[1].legend(title='Clustered by Application', loc='lower right')
    axes[0].yaxis.grid(color='lightgrey', linestyle=':')
    axes[1].yaxis.grid(color='lightgrey', linestyle=':')
    axes[0].set_axisbelow(True)
    axes[1].set_axisbelow(True)
    ticks = [40, 500, 1000, 1500, 2000, 2500, 3000]
    axes[0].set_xticks(ticks)
    axes[1].set_xticks(ticks)
    plt.savefig(join(save_dir, 'cluster_sizes_no_user_app.pdf'))
    plt.clf()
    plt.close()

def plot_no_runs_v_no_clusters(path_to_data, save_dir):
    '''
    shows correlation between an application having more runs and cluster.

    Parameters
    ----------
    path_to_data: string

    Returns
    -------
    None
    '''
    df = pd.read_csv(path_to_data, index_col=0)
    df = df[df['Number of Runs']!=0]
    fig, ax = plt.subplots(1, 1, figsize=[6,3])
    fig.subplots_adjust(left=0.115, right=0.97, top=.95, bottom=0.21, wspace=0.03)
    ax.set_ylim(-.1,3)
    ax.set_xlim(1,5)
    df['Number of Clusters'] = np.log10(df['Number of Clusters'])
    df['Number of Runs'] = np.log10(df['Number of Runs'])
    sns.regplot(data=df[df['Operation']=='Read'], x='Number of Runs', y='Number of Clusters', color='skyblue', ax=ax, 
                ci=None, order=0, label='Read', scatter_kws={'edgecolors':'black', 'zorder':1}, line_kws={'zorder':0})
    sns.regplot(data=df[df['Operation']=='Write'], x='Number of Runs', y='Number of Clusters', color='maroon', ax=ax, 
                ci=None, order=0, label='Write', scatter_kws={'edgecolors':'black', 'zorder':1}, line_kws={'zorder':0})
    ax.yaxis.grid(color='lightgrey', linestyle=':')
    ax.set_axisbelow(True)
    ax.legend(loc='upper left', fancybox=True)
    ax.set_ylabel('Number of Clusters')
    ax.set_xlabel('Number of Runs of an Application')
    positions = [0, 1, 2, 3]
    labels = ['$10^0$', '$10^1$', '$10^2$', '$10^3$']
    ax.yaxis.set_major_locator(ticker.FixedLocator(positions))
    ax.yaxis.set_major_formatter(ticker.FixedFormatter(labels))
    positions = [0, 1, 2, 3, 4 , 5]
    labels = ['0', '$10^1$', '$10^2$', '$10^3$', '$10^4$', '$10^5$']
    ax.xaxis.set_major_locator(ticker.FixedLocator(positions))
    ax.xaxis.set_major_formatter(ticker.FixedFormatter(labels))
    # Add minor ticks
    ticks = [1,2,3,4,5,6,7,8,9]  
    f_ticks = [np.log10(x) for x in ticks]
    tmp_ticks = [np.log10(x)+1 for x in ticks]
    f_ticks = f_ticks + tmp_ticks
    tmp_ticks = [np.log10(x)+2 for x in ticks]
    f_ticks = f_ticks + tmp_ticks
    ax.set_yticks(f_ticks, minor=True)
    f_ticks = []
    tmp_ticks = [np.log10(x)+1 for x in ticks]
    f_ticks = f_ticks + tmp_ticks
    tmp_ticks = [np.log10(x)+2 for x in ticks]
    f_ticks = f_ticks + tmp_ticks
    tmp_ticks = [np.log10(x)+3 for x in ticks]
    f_ticks = f_ticks + tmp_ticks
    tmp_ticks = [np.log10(x)+4 for x in ticks]
    f_ticks = f_ticks + tmp_ticks
    ax.set_xticks(f_ticks, minor=True)
    plt.savefig(join(save_dir, 'no_runs_v_no_clusters_in_app.pdf'))
    plt.clf()
    plt.close()

def plot_cluster_sizes(path_to_data, save_dir):
    '''
    CDFs of read and write cluster sizes.

    Parameters
    ----------
    path_to_data: string

    Returns
    -------
    None
    '''
    df = pd.read_csv(path_to_data, index_col=0) 
    df = df[df['Cluster Size']>40]
    df['Cluster Size'] = np.log10(df['Cluster Size'])
    read_info = df[df['Operation']=='Read']['Cluster Size'].tolist()
    read_median = np.median(read_info)
    read_75th   = np.percentile(read_info, 75)
    print("Median of Read: %d"%10**read_median)
    print("75th Percentile of Read: %d"%10**read_75th)
    read_mean   = np.mean(read_info)
    read_bins = np.arange(0, int(math.ceil(max(read_info)))+1, 0.01)
    hist = np.histogram(read_info, bins=read_bins)[0]
    cdf_read = np.cumsum(hist)
    cdf_read = [x/cdf_read[-1] for x in cdf_read]
    write_info = df[df['Operation']=='Write']['Cluster Size'].tolist()
    write_median = np.median(write_info)
    write_75th   = np.percentile(write_info, 75)
    print("Median of Write: %d"%10**write_median)
    print("75th Percentile of Write: %d"%10**write_75th)
    write_mean   = np.mean(write_info)
    write_bins = np.arange(0, int(math.ceil(max(write_info)))+1, 0.01)
    hist = np.histogram(write_info, bins=write_bins)[0]
    cdf_write = np.cumsum(hist)
    cdf_write = [x/cdf_write[-1] for x in cdf_write]
    # Get percentile info for each application
    applications = df['Application'].unique().tolist()
    for application in applications:
        print("\n%s\n-------------"%application)
        app_info = df[df['Application']==application]
        median = np.median(app_info[app_info['Operation']=='Read']['Cluster Size'].tolist())
        sfth   = np.percentile(app_info[app_info['Operation']=='Read']['Cluster Size'].tolist(), 75)
        print("Median of Read: %d"%10**median)
        print("75th Percentile of Read: %d"%10**sfth)
        median = np.median(app_info[app_info['Operation']=='Write']['Cluster Size'].tolist())
        sfth   = np.percentile(app_info[app_info['Operation']=='Write']['Cluster Size'].tolist(), 75)
        print("Median of Write: %d"%10**median)
        print("75th Percentile of Write: %d"%10**sfth)
    fig, ax = plt.subplots(1, 1, sharey=True, figsize=[5, 2])
    fig.subplots_adjust(left=0.30, right=0.85, top=.94, bottom=0.24, wspace=0.12)
    ax.plot(read_bins[:-1], cdf_read, color='skyblue', linewidth=2, label='Read')
    ax.plot(write_bins[:-1], cdf_write, color='maroon', linewidth=2, label='Write')
    ax.set_ylim(0,1)
    ax.set_yticks(np.arange(0,1.2,0.25))
    vals = ax.get_yticks()
    ax.set_yticklabels(['{:,.0%}'.format(x) for x in vals])
    ax.set_ylabel('CDF of Clusters')
    ax.set_xlabel('Number of Runs')
    ax.yaxis.grid(color='lightgrey', linestyle=':')
    ax.set_axisbelow(True)
    ax.set_xlim(np.log10(40),4)
    positions = [2, 3, 4 ]
    labels = ['$10^2$', '$10^3$', '$10^4$']
    ax.xaxis.set_major_locator(ticker.FixedLocator(positions))
    ax.xaxis.set_major_formatter(ticker.FixedFormatter(labels))
    # Add minor ticks
    ticks = [5,6,7,8,9] 
    f_ticks = []
    tmp_ticks = [np.log10(x)+1 for x in ticks]
    f_ticks = f_ticks + tmp_ticks
    ticks = [1,2,3,4,5,6,7,8,9] 
    tmp_ticks = [np.log10(x)+2 for x in ticks]
    f_ticks = f_ticks + tmp_ticks
    tmp_ticks = [np.log10(x)+3 for x in ticks]
    f_ticks = f_ticks + tmp_ticks
    ax.set_xticks(f_ticks, minor=True)
    # Add vertical lines for medians
    ax.axvline(read_median, color='skyblue', zorder=0, linestyle='--', linewidth=2)
    ax.axvline(write_median, color='maroon', zorder=0, linestyle=':', linewidth=2)
    # Add legend
    ax.legend(loc='lower right', fancybox=True)
    plt.savefig(join(save_dir, 'no_runs_in_clusters.pdf'))
    plt.clf()
    plt.close()

def plot_size_amount_v_perf_cov(path_to_data, save_dir):
    '''
    Plots boxplots to show how I/O amount affects performance variation. 

    Parameters
    ----------
    path_to_data: string
        Path to directory containing data to plot.
    save_dir: string
        Path to the directory to save the plots in.

    Returns
    -------
    None
    '''
    df = pd.read_csv(path_to_data, index_col=0)
    range = []
    for n in df['Average I/O Amount (bytes)']:
        if(100000<n<100000000):
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
    axes[0].set_ylim(0,50)
    axes[0].tick_params(axis='x', labelsize=10)
    axes[1].tick_params(axis='x', labelsize=10)
    plt.savefig(join(save_dir, 'info_amount.pdf'))
    plt.close()
    plt.clf()

def plot_perf_v_no_run(path_to_data, save_dir):
    '''
    Plots number of run affect on performance variation.

    Parameters
    ----------
    path_to_data: string

    Returns
    -------
    None
    '''
    df = pd.read_csv(path_to_data, index_col=0)
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
    axes[0].set_ylim(0,50)
    plt.savefig(join(save_dir, 'no_runs_v_perf_cov.pdf'))
    plt.close()
    plt.clf()
    return None

def plot_cluster_covs(path_to_data, save_dir):
    '''
    CDFs of read and write cluster CoVs.

    Parameters
    ----------
    path_to_data: string

    Returns
    -------
    None
    '''
    df = pd.read_csv(path_to_data, index_col=0) 
    df['Performance CoV (%)'] = np.log10(df['Performance CoV (%)'])
    read_info = df[df['Operation']=='Read']['Performance CoV (%)'].tolist()
    read_median = np.median(read_info)
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
    ax.set_xlim(0, 2)
    positions = [0,1,2]
    labels = ['$10^0$', '$10^1$', '$10^2$']
    ax.xaxis.set_major_locator(ticker.FixedLocator(positions))
    ax.xaxis.set_major_formatter(ticker.FixedFormatter(labels))
    # Add minor ticks
    ticks = [1,2,3,4,5,6,7,8,9] 
    f_ticks = []
    tmp_ticks = [np.log10(x) for x in ticks]
    f_ticks = f_ticks + tmp_ticks
    tmp_ticks = [np.log10(x)+1 for x in ticks]
    f_ticks = f_ticks + tmp_ticks
    ax.set_xticks(f_ticks, minor=True)
    # Add vertical lines for medians
    ax.axvline(read_median, color='skyblue', zorder=0, linestyle='--', linewidth=2)
    ax.axvline(write_median, color='maroon', zorder=0, linestyle=':', linewidth=2)
    # Add legend
    ax.legend(loc='lower right', fancybox=True)
    plt.savefig(join(save_dir, 'covs_cluster.pdf'))
    plt.clf()
    plt.close()

def plot_no_unique_behaviors(path_to_cluster_info, save_dir=''):
    '''
    Identify busier times that include runs from all
    three applications.

    Parameters
    ----------
    path_to_cluster_info: string
        Path to csv file with clustering information.

    Returns
    -------
    results: Pandas.DataFrame
        Information on the runs within the clusters.
    '''
    df = pd.read_csv(path_to_cluster_info, index_col=0)
    #df = cluster_info.groupby(['Application', 'Operation'])['Cluster Size'].nunique().reset_index(name='Number of Clusters')
    df = df[df['Cluster Size']>40]
    no_read_clusters  = df[df['Operation']=='Read']['Application'].value_counts().rename('Number of Clusters').reset_index()
    no_write_clusters = df[df['Operation']=='Write']['Application'].value_counts().rename('Number of Clusters').reset_index()
    no_read_clusters['Operation'] = 'Read'
    no_write_clusters['Operation'] = 'Write'
    df = no_read_clusters.append(no_write_clusters, ignore_index=True)
    df['Application'] = df['index']
    df =  df.drop('index', axis=1)
    d = {'Number of Clusters': int(no_read_clusters['Number of Clusters'].sum()), 'Operation': 'Read', 'Application': 'Overall'}
    df = df.append(d, ignore_index=True)
    d = {'Number of Clusters': int(no_write_clusters['Number of Clusters'].sum()), 'Operation': 'Write', 'Application': 'Overall'}
    df = df.append(d, ignore_index=True)
    df['Number of Clusters'] = np.log10(df['Number of Clusters'])
    print(df)
    fig, ax = plt.subplots(1, 1, figsize=[5, 2])
    fig.subplots_adjust(left=0.12, right=0.99, top=.945, bottom=0.45, wspace=0.25)
    order = ['Overall', 'vasp_gam_406746', 'mosst_dynamo.x_410575', 'pw.x_415566', 'pw.x_416364', 'vasp54withoutneb_397009', 'pw.x_381413', 'SpEC_383751', 'ideal.exe_309432', 'wrf.exe_309432', 'pp.x_381413']
    labels = ['Overall', 'vasp0', 'mosst0', 'QE0', 'QE1', 'vasp1', 'QE2', 'spec0', 'wrf0', 'wrf1', 'QE3']
    rects = sns.barplot(data=df, x='Application', y='Number of Clusters', hue='Operation', ax=ax, edgecolor='black', linewidth=2, palette={'Read': 'skyblue', 'Write': 'maroon'}, order=order)
    plt.setp(ax.artists, alpha=1, linewidth=2, fill=False, edgecolor="black")
    # Add number to bars
    '''
    for p in ax.patches:
        a = (10**float(p.get_height()))
        a = str(int(a))
        ax.annotate(a, (p.get_x() + p.get_width() / 2., p.get_height()-0.04), 
                            ha = 'center', va = 'center', xytext = (0, 9), textcoords = 'offset points', fontsize=8)
    '''
    # Fix x-axis labels
    new_labels = ['%s\n%s'%('_'.join(x.get_text().split('_')[0:-1]), x.get_text().split('_')[-1]) for x in ax.get_xticklabels()]
    # Labels for axes
    ax.set_xticklabels(labels, rotation=90)
    ax.set_xlabel('')
    ax.set_ylabel('')
    fig.text(0.50, 0.06, 'Application', ha='center', va='center')
    fig.text(0.003, 0.20, 'Number of Clusters', rotation=90)
    # Fix y-axis labels
    ax.set_ylim(0,3)
    positions = [0, 1, 2, 3, 4]
    labels = ['$10^0$', '$10^1$', '$10^2$', '$10^3$', '$10^4$']
    ax.yaxis.set_major_locator(ticker.FixedLocator(positions))
    ax.yaxis.set_major_formatter(ticker.FixedFormatter(labels))
    # Add minor ticks
    ticks = [1,2,3,4,5,6,7,8,9] 
    f_ticks = []
    tmp_ticks = [np.log10(x) for x in ticks]
    f_ticks = f_ticks + tmp_ticks
    tmp_ticks = [np.log10(x)+1 for x in ticks]
    f_ticks = f_ticks + tmp_ticks
    #ticks = [1,2,3,4,] 
    tmp_ticks = [np.log10(x)+2 for x in ticks]
    f_ticks = f_ticks + tmp_ticks
    ax.set_yticks(f_ticks, minor=True)
    # Add grid
    ax.yaxis.grid(color='lightgrey', linestyle=':')
    ax.set_axisbelow(True)
    # Add legend
    ax.legend(loc='upper right', fancybox=True)
    plt.savefig(join(save_dir, 'no_clusters_in_applications.pdf'))
    plt.clf()
    plt.close()

def plot_cluster_size_percentiles(path_to_data, save_dir):
    '''
    Barplot of read and write cluster size percentiles.

    Parameters
    ----------
    path_to_data: string

    Returns
    -------
    None
    '''
    df = pd.read_csv(path_to_data, index_col=0) 
    results = pd.DataFrame(columns=['Application', 'Operation', 'Percentile', 'Number of Runs'])
    df = df[df['Cluster Size']>40]
    df['Cluster Size'] = np.log10(df['Cluster Size'])
    read_info = df[df['Operation']=='Read']['Cluster Size'].tolist()
    read_median = np.median(read_info)
    read_75th   = np.percentile(read_info, 75)
    print("Median of Read: %d"%10**read_median)
    d = {'Application': 'Overall', 'Operation': 'Read', 'Percentile': 50, 'Number of Runs': 10**read_median}
    results = results.append(d, ignore_index=True)
    print("75th Percentile of Read: %d"%10**read_75th)
    d = {'Application': 'Overall', 'Operation': 'Read', 'Percentile': 75, 'Number of Runs': 10**read_75th}
    results = results.append(d, ignore_index=True)
    write_info = df[df['Operation']=='Write']['Cluster Size'].tolist()
    write_median = np.median(write_info)
    write_75th   = np.percentile(write_info, 75)
    print("Median of Write: %d"%10**write_median)
    d = {'Application': 'Overall', 'Operation': 'Write', 'Percentile': 50, 'Number of Runs': int(10**write_median)}
    results = results.append(d, ignore_index=True)
    print("75th Percentile of Write: %d"%10**write_75th)
    d = {'Application': 'Overall', 'Operation': 'Write', 'Percentile': 75, 'Number of Runs': int(10**write_75th)}
    results = results.append(d, ignore_index=True)
    # Get percentile info for each application
    applications = df['Application'].unique().tolist()
    for application in applications:
        print("\n%s\n-------------"%application)
        app_info = df[df['Application']==application]
        median = np.median(app_info[app_info['Operation']=='Read']['Cluster Size'].tolist())
        sfth   = np.percentile(app_info[app_info['Operation']=='Read']['Cluster Size'].tolist(), 75)
        print("Median of Read: %d"%10**median)
        d = {'Application': application, 'Operation': 'Read', 'Percentile': 50, 'Number of Runs': int(10**median)}
        results = results.append(d, ignore_index=True)
        print("75th Percentile of Read: %d"%10**sfth)
        d = {'Application': application, 'Operation': 'Read', 'Percentile': 75, 'Number of Runs': int(10**sfth)}
        results = results.append(d, ignore_index=True)
        median = np.median(app_info[app_info['Operation']=='Write']['Cluster Size'].tolist())
        sfth   = np.percentile(app_info[app_info['Operation']=='Write']['Cluster Size'].tolist(), 75)
        print("Median of Write: %d"%10**median)
        d = {'Application': application, 'Operation': 'Write', 'Percentile': 50, 'Number of Runs': int(10**median)}
        results = results.append(d, ignore_index=True)
        print("75th Percentile of Write: %d"%10**sfth)
        d = {'Application': application, 'Operation': 'Write', 'Percentile': 75, 'Number of Runs': int(10**sfth)}
        results = results.append(d, ignore_index=True)
    print(results)
    medians = results[results['Percentile']==50]
    #medians = medians.sort_values('Number of Runs')
    medians['Number of Runs'] = np.log10(medians['Number of Runs'])
    print(medians)
    fig, ax = plt.subplots(1, 1, figsize=[5, 2])
    fig.subplots_adjust(left=0.15, right=0.99, top=.945, bottom=0.45, wspace=0.25)
    order = ['vasp_gam_406746', 'mosst_dynamo.x_410575', 'pw.x_415566', 'pw.x_416364', 'vasp54withoutneb_397009', 'pw.x_381413', 'SpEC_383751', 'ideal.exe_309432', 'wrf.exe_309432', 'pp.x_381413']
    labels = ['vasp0', 'mosst0', 'QE0', 'QE1', 'vasp1', 'QE2', 'spec0', 'wrf0', 'wrf1', 'QE3']
    rects = sns.barplot(data=medians, x='Application', y='Number of Runs', hue='Operation', ax=ax, edgecolor='black', linewidth=2, palette={'Read': 'skyblue', 'Write': 'maroon'}, order=order)
    plt.setp(ax.artists, alpha=1, linewidth=2, fill=False, edgecolor="black")
    # Add number to bars
    '''
    for p in ax.patches:
        a = (10**float(p.get_height()))
        a = '%d'%a
        ax.annotate(a, (p.get_x() + p.get_width() / 2., p.get_height()), 
                            ha = 'center', va = 'center', xytext = (0, 9), textcoords = 'offset points')
    '''
    # Fix x-axis labels
    #new_labels = ['%s\n%s'%('_'.join(x.get_text().split('_')[0:-1]), x.get_text().split('_')[-1]) for x in ax.get_xticklabels()]
    # Labels for axes
    ax.set_xticklabels(labels, rotation=90)
    ax.set_xlabel('')
    ax.set_ylabel('')
    fig.text(0.50, 0.05, 'Application', ha='center', va='center')
    fig.text(0.04, 0.6, 'Median Number of\nRuns in Clusters', rotation=90, ha='center', va='center')
    # Fix y-axis labels
    ax.set_ylim(0,3)
    positions = [0, 1, 2, 3, 4]
    labels = ['$10^0$', '$10^1$', '$10^2$', '$10^3$', '$10^4$']
    ax.yaxis.set_major_locator(ticker.FixedLocator(positions))
    ax.yaxis.set_major_formatter(ticker.FixedFormatter(labels))
    # Add minor ticks
    ticks = [1,2,3,4,5,6,7,8,9] 
    f_ticks = []
    tmp_ticks = [np.log10(x) for x in ticks]
    f_ticks = f_ticks + tmp_ticks
    tmp_ticks = [np.log10(x)+1 for x in ticks]
    f_ticks = f_ticks + tmp_ticks
    #ticks = [1,2,3,4,] 
    tmp_ticks = [np.log10(x)+2 for x in ticks]
    f_ticks = f_ticks + tmp_ticks
    ax.set_yticks(f_ticks, minor=True)
    # Add grid
    ax.yaxis.grid(color='lightgrey', linestyle=':')
    ax.set_axisbelow(True)
    # Add legend
    ax.get_legend().remove()
    ax.legend(loc=[0.02,0.05], fancybox=True, fontsize=12)
    plt.savefig(join(save_dir, 'median_no_runs_in_clusters_by_application.pdf'))
    plt.clf()
    plt.close()

def plot_cluster_cmp_perf_tod(path_to_data, E=0.10):
    df = pd.read_csv(path_to_data, index_col=0)
    fig, axes = plt.subplots(2, 1, sharex=True, figsize=[5, 3.3])
    fig.subplots_adjust(left=0.14, right=0.965, top=.94, bottom=0.21, wspace=0.12)
    range_tod = []
    range_tow = []
    from_zone = tz.gettz('UTC')
    to_zone = tz.gettz('America/Chicago')
    order = ['0-\n3', '3-\n6', '6-\n9', '9-\n12', '12-\n15', '15-\n18', '18-\n21', '21-\n24']
    for n in df['Start Time']:
        datetime_time = datetime.fromtimestamp(n).replace(tzinfo=from_zone).astimezone(to_zone)
        h             = int(datetime_time.hour)
        d             = int(datetime_time.weekday())
        # Group by time of day
        if(h == 0 or h == 1 or h == 2):
            range_tod.append('0-\n3')
        elif(h == 3 or h == 4 or h == 5):
            range_tod.append('3-\n6')
        elif(h == 6 or h == 7 or h == 8):
            range_tod.append('6-\n9')
        elif(h == 9 or h == 10 or h == 11):
            range_tod.append('9-\n12')
        elif(h == 12 or h == 13 or h == 14):
            range_tod.append('12-\n15')
        elif(h == 15 or h == 16 or h == 17):
            range_tod.append('15-\n18')
        elif(h == 18 or h == 19 or h == 20):
            range_tod.append('18-\n21')
        elif(h == 21 or h == 22 or h == 23):
            range_tod.append('21-\n24')
        else:
            print("don't forget: %d"%n)
    df['Range, Time of Day'] = range_tod
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
        for i in range(0, n+1):
            perf_covs = []
            cluster = clusters[clusters['Cluster Number']==i]
            cluster_size = cluster.shape[0]
            if(cluster_size<40):
                continue
            perf_cov = stats.variation(cluster['Performance'])
            for n in range(0,cluster_size):
                perf_covs.append(perf_cov)
            cluster['Performance CoV'] = perf_covs
            results = results.append(cluster, ignore_index=True)
    E_25 = int(results.shape[0]*E)
    l = results.nsmallest(E_25, ['Performance CoV'])
    l['Performance CoV Percentile'] = E*100
    h = results.nlargest(E_25, ['Performance CoV'])
    h['Performance CoV Percentile'] = 100-E*100
    results = h.append(l, ignore_index=True)
    #print(results)
    TODs = results['Range, Time of Day'].unique().tolist()
    plot = pd.DataFrame()
    for TOD in TODs:
        count = l[l['Range, Time of Day']==TOD].shape[0]
        d = {'TOD': TOD, 'Count': int(count), 'Performance CoV Percentile': 'Bottom 10%'}
        plot = plot.append(d, ignore_index=True)
        count = h[h['Range, Time of Day']==TOD].shape[0]
        d = {'TOD': TOD, 'Count': int(count), 'Performance CoV Percentile': 'Top 10%'}
        plot = plot.append(d, ignore_index=True)
    sns.barplot(ax=axes[0], data=plot, x='TOD', y='Count', hue='Performance CoV Percentile', edgecolor='black', linewidth=2, palette='Blues', order=order)
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
        for i in range(0, n+1):
            perf_covs = []
            cluster = clusters[clusters['Cluster Number']==i]
            cluster_size = cluster.shape[0]
            if(cluster_size<40):
                continue
            perf_cov = stats.variation(cluster['Performance'])
            for n in range(0,cluster_size):
                perf_covs.append(perf_cov)
            cluster['Performance CoV'] = perf_covs
            results = results.append(cluster, ignore_index=True)
    E_25 = int(results.shape[0]*E)
    l = results.nsmallest(E_25, ['Performance CoV'])
    l['Performance CoV Percentile'] = E*100
    h = results.nlargest(E_25, ['Performance CoV'])
    h['Performance CoV Percentile'] = 100-E*100
    results = h.append(l, ignore_index=True)
    #print(results)
    TODs = results['Range, Time of Day'].unique().tolist()
    plot = pd.DataFrame()
    for TOD in TODs:
        count = l[l['Range, Time of Day']==TOD].shape[0]
        d = {'TOD': TOD, 'Count': int(count), 'Performance CoV Percentile': 'Bottom 10%'}
        plot = plot.append(d, ignore_index=True)
        count = h[h['Range, Time of Day']==TOD].shape[0]
        d = {'TOD': TOD, 'Count': int(count), 'Performance CoV Percentile': 'Top 10%'}
        plot = plot.append(d, ignore_index=True)
    sns.barplot(ax=axes[1], data=plot, x='TOD', y='Count', hue='Performance CoV Percentile', edgecolor='black', linewidth=2, palette='Reds', order=order)
    # Plot aesthetics
    axes[0].tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
    axes[1].tick_params(axis='x', labelrotation = 0)
    axes[0].yaxis.grid(color='lightgrey', linestyle=':')
    axes[0].set_axisbelow(True)
    axes[1].yaxis.grid(color='lightgrey', linestyle=':')
    axes[1].set_axisbelow(True)
    axes[0].set_ylim(0,2000)
    axes[1].set_ylim(0,2000)
    fig.text(0.55, 0.01, 'Time of Day (24-hr)', ha='center')
    fig.text(0.001, 0.55, "Number of Runs", rotation=90, va='center')
    fig.text(0.50, 0.95, 'Read')
    fig.text(0.50, 0.55, 'Write')
    axes[0].legend(loc=(0.10,0.43),fontsize=9, title='Perf CoV Percentile', title_fontsize=10)
    axes[1].legend(loc=(0.60,0.01),fontsize=9, title='Perf CoV Percentile', title_fontsize=10)
    axes[0].set_ylabel('')
    axes[1].set_ylabel('')
    axes[0].set_xlabel('')
    axes[1].set_xlabel('')
    #axes[1].set_xticklabels(labels)
    plt.savefig('./performance_cov_percentile_features_TOD.pdf')

def plot_cluster_cmp_perf_dow(path_to_data, E=0.10):
    df = pd.read_csv(path_to_data, index_col=0)
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
        for i in range(0, n+1):
            perf_covs = []
            cluster = clusters[clusters['Cluster Number']==i]
            cluster_size = cluster.shape[0]
            if(cluster_size<40):
                continue
            perf_cov = stats.variation(cluster['Performance'])
            for n in range(0,cluster_size):
                perf_covs.append(perf_cov)
            cluster['Performance CoV'] = perf_covs
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
        for i in range(0, n+1):
            perf_covs = []
            cluster = clusters[clusters['Cluster Number']==i]
            cluster_size = cluster.shape[0]
            if(cluster_size<40):
                continue
            perf_cov = stats.variation(cluster['Performance'])
            for n in range(0,cluster_size):
                perf_covs.append(perf_cov)
            cluster['Performance CoV'] = perf_covs
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
    axes[0].set_ylim(0,4000)
    axes[1].set_ylim(0,4000)
    axes[0].set_yticks([x for x in range(0,4001,2000)])
    axes[1].set_yticks([x for x in range(0,4001,2000)])
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
    plt.savefig('./performance_cov_percentile_features_DOW.pdf')

def plot_perf_v_temporal(path_to_data, save_dir):
    '''
    Plots temporal effects on performance.

    Parameters
    ----------
    path_to_data: string

    Returns
    -------
    None
    '''
    df = pd.read_csv(path_to_data, index_col=0)
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
        else:
            print("don't forget: %d"%n)
    df['Range'] = range
    read_df = df[df['Operation']=='Read']
    write_df = df[df['Operation']=='Write']
    range_labels = read_df['Range'].unique()
    print("For Read:")
    for range_label in range_labels:
        print('Range Label, Number of Clusters: %s %d'%(range_label, len(read_df[read_df['Range']==range_label])))
    range_labels = write_df['Range'].unique()
    print('For Write')
    for range_label in range_labels:
        print('Range Label, Number of Clusters: %s %d'%(range_label, len(write_df[write_df['Range']==range_label])))
    # Barplot of time periods to performance CoV
    fig, axes = plt.subplots(1, 2, sharey=True, figsize=[5,2])
    fig.subplots_adjust(left=0.12, right=0.990, top=0.96, bottom=0.45, wspace=0.03)
    order = ['<1d', '1-3d', '3d-1w', '1w-2w', '2w-1M', '1-3M', '3-6M']
    labels = ['<1d', '1-\n3d', '3d-\n1w', '1w-\n2w', '2w-\n1M', '1-\n3M', '3-\n6M']
    print(read_df)
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
    axes[0].set_ylim(0,50)
    plt.savefig(join(save_dir, 'time_period_v_perf_cov.pdf'))
    plt.close()
    plt.clf()
    return None

def plot_cluster_covs_by_application(path_to_data, save_dir):
    '''
    CDFs of read and write cluster CoVs.

    Parameters
    ----------
    path_to_data: string

    Returns
    -------
    None
    '''
    df_all = pd.read_csv(path_to_data, index_col=0) 
    applications = df_all['Application'].unique().tolist()
    fig, axes = plt.subplots(2, 2, sharex=True, sharey=True, figsize=[5, 3])
    fig.subplots_adjust(left=0.15, right=0.97, top=.97, bottom=0.18, hspace=0.20, wspace=0.20)
    i = 0
    for application in applications:
        print(application)
        df = df_all[df_all['Application']==application]
        if(len(df)<10):
            continue
        df['Performance CoV (%)'] = np.log10(df['Performance CoV (%)'])
        read_info = df[df['Operation']=='Read']['Performance CoV (%)'].tolist()
        read_median = np.median(read_info)
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
        if(i==0): 
            a = 0
            b = 0
        elif(i==1): 
            a = 0
            b = 1
        elif(i==2): 
            a = 1
            b = 0
        elif(i==3): 
            a = 1
            b = 1
        axes[a][b].plot(read_bins[:-1], cdf_read, color='skyblue', linewidth=2)
        axes[a][b].plot(write_bins[:-1], cdf_write, color='maroon', linewidth=2)
        axes[a][b].set_ylim(0,1)
        axes[a][b].set_yticks(np.arange(0,1.2,0.50))
        vals = axes[a][b].get_yticks()
        axes[a][b].set_yticklabels(['{:,.0%}'.format(x) for x in vals])
        #axes[i].set_ylabel('Percent of Clusters')
        #axes[i].set_xlabel('Performance CoV (%)')
        axes[a][b].yaxis.grid(color='lightgrey', linestyle=':')
        axes[a][b].set_axisbelow(True)
        axes[a][b].set_xlim(0, 2)
        positions = [0,1,2]
        labels = ['$10^0$', '$10^1$', '$10^2$']
        axes[a][b].xaxis.set_major_locator(ticker.FixedLocator(positions))
        axes[a][b].xaxis.set_major_formatter(ticker.FixedFormatter(labels))
        # Add minor ticks
        ticks = [1,2,3,4,5,6,7,8,9] 
        f_ticks = []
        tmp_ticks = [np.log10(x) for x in ticks]
        f_ticks = f_ticks + tmp_ticks
        tmp_ticks = [np.log10(x)+1 for x in ticks]
        f_ticks = f_ticks + tmp_ticks
        axes[a][b].set_xticks(f_ticks, minor=True)
        # Add vertical lines for medians
        axes[a][b].axvline(read_median, color='skyblue', zorder=0, linestyle='--', linewidth=2)
        axes[a][b].axvline(write_median, color='maroon', zorder=0, linestyle=':', linewidth=2)
        i = i + 1
    # Add legend
    legend_elements = [Line2D([0], [0], color='skyblue', lw=2, label='Read', alpha=1), Line2D([0], [0], color='maroon', lw=2, label='Write', alpha=1)]
    axes[0][0].legend(handles=legend_elements, loc='upper left', fancybox=True, fontsize=12)
    fig.text(0.001, 0.56, 'CDF of Clusters', rotation=90, ha='left', va='center')
    fig.text(0.56, 0.008, 'Performance CoV (%)', ha='center', va='bottom')
    x0 = 0.603
    x1 = 0.153
    h = 0.21
    s = 0.215
    labels = ['QE1','QE0','mosst0','vasp0']
    fig.text(x0, h, s=labels[3])
    fig.text(x1, h, s=labels[2])
    fig.text(x0, h+2*s, s=labels[1])
    fig.text(x1, h+2*s, s=labels[0])
    plt.savefig(join(save_dir, 'covs_cluster_%s.pdf'%'overall'))
    plt.clf()
    plt.close()

def plot_barplot_ex_no_overlaps_normalized(path_to_data, save_path):
    # pw.x_416364, max: 8
    df_all = pd.read_csv(path_to_data, index_col=0)
    #applications = df_all['Application'].unique().tolist()
    applications = ['pw.x_416364','pw.x_415566','mosst_dynamo.x_410575','vasp_gam_406746']
    labels = ['%s\n%s'%('_'.join(x.split('_')[0:-1]), x.split('_')[-1]) for x in applications]
    overall_results = pd.DataFrame(columns=['Operation', 'Range', 'Number of Clusters', 'Percent of Clusters'])
    r_clusters = len(df_all[df_all['Operation']=='Read'])
    w_clusters = len(df_all[df_all['Operation']=='Write'])
    fig, axes = plt.subplots(2, 2, sharex=True, sharey=True, figsize=[5, 3])
    fig.subplots_adjust(left=0.15, right=0.99, top=.975, bottom=0.23, hspace=0.20, wspace=0.06)
    a = 0
    for application in applications:
        df = df_all[df_all['Application']==application]
        results = pd.DataFrame(columns=['Operation', 'Range', 'Number of Clusters', 'Percent of Clusters'])
        range = []
        ranges = ['0-\n20%', '20-\n40%', '40-\n60%', '60-\n80%', '80-\n100%']
        operation = 'Read'
        counts = [0, 0, 0, 0, 0, 0, 0]
        n_clusters = len(df[df['Operation']==operation])
        for i in df[df['Operation']==operation]['Number of Overlaps']:
            if a==0:
                f=0
                g=0
            elif a==1:
                f=0
                g=1
            elif a==2:
                f=1
                g=0
            elif a==3:
                f=1
                g=1
            n = (i/n_clusters)*100
            if(n<20):
                counts[0] = counts[0] + 1
            elif(n<40):
                counts[1] = counts[1] + 1
            elif(n<60):
                counts[2] = counts[2] + 1
            elif(n<80):
                counts[3] = counts[3] + 1
            elif(n<=100):
                counts[4] = counts[4] + 1
            else:
                print("don't forget: %d"%n)
        for i in np.arange(0,5):
            d = {'Operation': operation, 'Range': ranges[i], 'Number of Clusters': counts[i], 'Percent of Clusters': (counts[i]/n_clusters)}
            results = results.append(d, ignore_index=True)
            d = {'Operation': operation, 'Range': ranges[i], 'Number of Clusters': counts[i], 'Percent of Clusters': (counts[i]/r_clusters)}
            overall_results = overall_results.append(d, ignore_index=True)
        operation = 'Write'
        counts = [0, 0, 0, 0, 0, 0, 0]
        n_clusters = len(df[df['Operation']==operation])
        for i in df[df['Operation']==operation]['Number of Overlaps']:
            n = (i/n_clusters)*100
            if(n<20):
                counts[0] = counts[0] + 1
            elif(n<40):
                counts[1] = counts[1] + 1
            elif(n<60):
                counts[2] = counts[2] + 1
            elif(n<80):
                counts[3] = counts[3] + 1
            elif(n<=100):
                counts[4] = counts[4] + 1
            else:
                print("don't forget: %d"%n)
        #df['Range'] = range
        for i in np.arange(0,5):
            d = {'Operation': operation, 'Range': ranges[i], 'Number of Clusters': counts[i], 'Percent of Clusters': (counts[i]/n_clusters)}
            results = results.append(d, ignore_index=True)
            d = {'Operation': operation, 'Range': ranges[i], 'Number of Clusters': counts[i], 'Percent of Clusters': (counts[i]/w_clusters)}
            overall_results = overall_results.append(d, ignore_index=True)
        print(results)
        sns.barplot(data=results, x='Range', y='Percent of Clusters', hue='Operation', ax=axes[f][g], edgecolor='black', linewidth=2, palette={'Read': 'skyblue', 'Write': 'maroon'}, order=ranges)
        axes[f][g].yaxis.grid(color='lightgrey', linestyle=':')
        axes[f][g].set_axisbelow(True)
        axes[f][g].get_legend().remove()
        axes[f][g].set_ylim(0,1)
        vals = axes[f][g].get_yticks()
        axes[f][g].set_yticklabels(['{:,.0%}'.format(x) for x in vals])
        axes[f][g].set_xlabel('')
        axes[f][g].set_ylabel('')
        axes[f][g].set_yticks([0,.25,.50,.75,1.00],minor=True)
        axes[f][g].yaxis.grid(color='lightgrey', linestyle=':', which='minor')
        axes[f][g].set_axisbelow(True)
        a = a + 1
    axes[0][0].legend(loc='upper left', prop={'size': 10})
    fig.text(0.001, 0.56, 'Percent of Clusters', rotation=90, ha='left', va='center')
    fig.text(0.56, 0.005, 'Percent of Clusters Overlapped', ha='center', va='bottom')
    x1 = 0.40
    x0 = x1 + 0.45
    h = 0.505
    s = 0.206
    labels = ['QE1','QE0','mosst0','vasp0']
    fig.text(x0, h, s=labels[3])
    fig.text(x1, h, s=labels[2])
    fig.text(x0, h+2*s, s=labels[1])
    fig.text(x1, h+2*s, s=labels[0])
    plt.savefig(join(save_path, 'application_examples.pdf'))
    plt.clf()
    plt.close()
        
    overall_results = overall_results.groupby(['Operation', 'Range']).sum().reset_index()
    print(overall_results)
    fig, ax = plt.subplots(1, 1, figsize=[5, 1.4])
    fig.subplots_adjust(left=0.20, right=0.70, top=.945, bottom=0.47, wspace=0.25)
    sns.barplot(data=overall_results, x='Range', y='Percent of Clusters', hue='Operation', ax=ax, edgecolor='black', linewidth=2, palette={'Read': 'skyblue', 'Write': 'maroon'}, order=ranges)
    ax.yaxis.grid(color='lightgrey', linestyle=':')
    ax.set_axisbelow(True)
    ax.get_legend().remove()
    ax.legend(title='', loc=[1.02,0.01])
    ax.set_ylim(0,1)
    vals = ax.get_yticks()
    ax.set_yticklabels(['{:,.0%}'.format(x) for x in vals])
    #ax.legend('upper right', fancybox=True)
    ax.set_xlabel('')
    ax.set_ylabel('')
    fig.text(0.46, 0.03, 'Percent of Clusters Overlapped', ha='center')
    fig.text(0.05, 0.41, 'Percent of\nClusters', rotation=90, ha='center')
    plt.savefig(join('./overall.pdf'))
    plt.clf()
    plt.close()

def plot_time_spans_by_application(path_to_cluster_info, save_dir):
    '''
    Plots the temporal behavior of runs in the clusters.

    Parameters
    ----------
    path_to_cluster_info: string
        Path to csv file with clustering temporal run information.

    Returns
    -------
    None
    '''
    df = pd.read_csv(path_to_cluster_info, index_col=0)
    fig, ax = plt.subplots(1, 1, sharey=True, figsize=(5, 2))
    fig.subplots_adjust(left=0.18, right=0.99, top=.95, bottom=0.28, wspace=0.12)
    order = ['pw.x_416364','pw.x_415566','mosst_dynamo.x_410575','vasp_gam_406746']
    labels = ['(a)','(b)','(c)','(d)']
    df['Time Span'] = df['Total Time']/86400
    df['Time Span'] = np.log10(df['Time Span'])
    sns.violinplot(data=df, x='Application', y='Time Span', ax=ax, order=order, inner='quartile', linewidth=2, hue='Operation', split=True, palette={'Read':'skyblue', 'Write':'maroon'})
    # Labels for axes
    ax.set_xticklabels(labels)
    ax.set_xlabel('')
    ax.set_ylabel('')
    fig.text(0.59, 0.05, 'Application', ha='center', va='center')
    fig.text(0.003, 0.65, 'Cluster Time\nSpan (Days)', rotation=90, va='center', multialignment='center')
    positions = [-2, 0, 2, 4]
    labels = ['$10^{-2}$', '$10^{0}$', '$10^{2}$', '$10^4$']
    ax.yaxis.set_major_locator(ticker.FixedLocator(positions))
    ax.yaxis.set_major_formatter(ticker.FixedFormatter(labels))
    # Add minor ticks
    ticks = [1,2,3,4,5,6,7,8,9] 
    f_ticks = []
    tmp_ticks = [np.log10(x)-3 for x in ticks]
    f_ticks = f_ticks + tmp_ticks
    tmp_ticks = [np.log10(x)-2 for x in ticks]
    f_ticks = f_ticks + tmp_ticks
    tmp_ticks = [np.log10(x)-1 for x in ticks]
    f_ticks = f_ticks + tmp_ticks
    tmp_ticks = [np.log10(x) for x in ticks]
    f_ticks = f_ticks + tmp_ticks
    tmp_ticks = [np.log10(x)+1 for x in ticks]
    f_ticks = f_ticks + tmp_ticks
    tmp_ticks = [np.log10(x)+2 for x in ticks]
    f_ticks = f_ticks + tmp_ticks
    tmp_ticks = [np.log10(x)+3 for x in ticks]
    f_ticks = f_ticks + tmp_ticks
    ax.set_yticks(f_ticks, minor=True)
    # Add Major ticks
    ax.set_ylim(-3,4)
    ax.yaxis.grid(color='lightgrey', linestyle=':')
    ax.set_axisbelow(True)
    ax.get_legend().remove()
    ax.legend(loc='lower left', fancybox=True, fontsize=10)
    print(len(ax.lines))
    for i, l in enumerate(ax.lines):
        if(i%6==0):
            l.set_linestyle('--')
            l.set_linewidth(1.2)
            l.set_color('black')
            l.set_alpha(0.8)
        elif(i%6==1):
            l.set_linestyle('-')
            l.set_linewidth(1.2)
            l.set_color('black')
            l.set_alpha(0.8)
        elif(i%6==2):
            l.set_linestyle('--')
            l.set_linewidth(1.2)
            l.set_color('black')
            l.set_alpha(0.8)
        elif(i%6==3):
            l.set_linestyle('--')
            l.set_linewidth(1.2)
            l.set_color('white')
            l.set_alpha(0.8)
        elif(i%6==4):
            l.set_linestyle('-')
            l.set_linewidth(1.2)
            l.set_color('white')
            l.set_alpha(0.8)
        elif(i%6==5):
            l.set_linestyle('--')
            l.set_linewidth(1.2)
            l.set_color('white')
            l.set_alpha(0.8)
    plt.savefig(join(save_dir, 'cluster_time_spans_by_application.pdf'))
    plt.clf()
    plt.close()

def plot_interarrival_times_by_application(path_to_data, save_dir):
    '''
    Plot violinplots showing the inter-arrival times of runs within each application.
    '''
    df = pd.read_csv(path_to_data, index_col=0) 
    fig, ax = plt.subplots(1, 1, sharey=True, figsize=(5, 2))
    fig.subplots_adjust(left=0.18, right=0.99, top=.95, bottom=0.28, wspace=0.12)
    #df['Inter-Arrival Time (Hours)'] = np.log10(df['Inter-Arrival Time (Hours)'])
    order = ['pw.x_416364','pw.x_415566','mosst_dynamo.x_410575','vasp_gam_406746']
    labels = ['(a)','(b)','(c)','(d)']
    df['Inter-Arrival Time (Hours)'] = np.log10(df['Inter-Arrival Time (Hours)'])
    sns.violinplot(data=df, x='Application', y='Inter-Arrival Time (Hours)', ax=ax, order=order, inner='quartile', linewidth=2, hue='Operation', split=True, palette={'Read':'skyblue', 'Write':'maroon'})
    # Labels for axes
    ax.set_xticklabels(labels)
    ax.set_xlabel('')
    ax.set_ylabel('')
    fig.text(0.58, 0.05, 'Application', ha='center', va='center')
    fig.text(0.003, 0.625, 'Inter-Arrival\nTimes (Hours)', rotation=90, va='center', multialignment='center')
    positions = [-4, -2, 0, 2, 4]
    labels = ['$10^{-4}$', '$10^{-2}$', '$10^{0}$', '$10^{2}$', '$10^{4}$']
    ax.yaxis.set_major_locator(ticker.FixedLocator(positions))
    ax.yaxis.set_major_formatter(ticker.FixedFormatter(labels))
    # Add minor ticks
    ticks = [1,2,3,4,5,6,7,8,9] 
    f_ticks = []
    tmp_ticks = [np.log10(x)-5 for x in ticks]
    f_ticks = f_ticks + tmp_ticks
    tmp_ticks = [np.log10(x)-4 for x in ticks]
    f_ticks = f_ticks + tmp_ticks
    tmp_ticks = [np.log10(x)-3 for x in ticks]
    f_ticks = f_ticks + tmp_ticks
    tmp_ticks = [np.log10(x)-2 for x in ticks]
    f_ticks = f_ticks + tmp_ticks
    tmp_ticks = [np.log10(x)-1 for x in ticks]
    f_ticks = f_ticks + tmp_ticks
    tmp_ticks = [np.log10(x) for x in ticks]
    f_ticks = f_ticks + tmp_ticks
    tmp_ticks = [np.log10(x)+1 for x in ticks]
    f_ticks = f_ticks + tmp_ticks
    tmp_ticks = [np.log10(x)+2 for x in ticks]
    f_ticks = f_ticks + tmp_ticks
    tmp_ticks = [np.log10(x)+3 for x in ticks]
    f_ticks = f_ticks + tmp_ticks
    ax.set_yticks(f_ticks, minor=True)
    # Add Major ticks
    ax.set_ylim(-4,4)
    ax.yaxis.grid(color='lightgrey', linestyle=':')
    ax.set_axisbelow(True)
    ax.get_legend().remove()
    ax.legend(loc='upper right', fancybox=True)
    print(len(ax.lines))
    for i, l in enumerate(ax.lines):
        if(i%6==0):
            l.set_linestyle('--')
            l.set_linewidth(1.2)
            l.set_color('black')
            l.set_alpha(0.8)
        elif(i%6==1):
            l.set_linestyle('-')
            l.set_linewidth(1.2)
            l.set_color('black')
            l.set_alpha(0.8)
        elif(i%6==2):
            l.set_linestyle('--')
            l.set_linewidth(1.2)
            l.set_color('black')
            l.set_alpha(0.8)
        elif(i%6==3):
            l.set_linestyle('--')
            l.set_linewidth(1.2)
            l.set_color('white')
            l.set_alpha(0.8)
        elif(i%6==4):
            l.set_linestyle('-')
            l.set_linewidth(1.2)
            l.set_color('white')
            l.set_alpha(0.8)
        elif(i%6==5):
            l.set_linestyle('--')
            l.set_linewidth(1.2)
            l.set_color('white')
            l.set_alpha(0.8)
    plt.savefig(join(save_dir, 'interarrival_times_by_application.pdf'))
    plt.clf()
    plt.close()

def plot_cluster_cmp_perf(path_to_data):
    df = pd.read_csv(path_to_data, index_col=0)
    fig, axes = plt.subplots(2, 1, figsize=[5, 2.2])
    fig.subplots_adjust(left=0.12, right=0.70, top=.91, bottom=0.30, wspace=0.40, hspace=0.40)
    results = pd.DataFrame()
    E = 2
    labels = ['I/O\nAmount','Shared\nFiles','Unique\nFiles']#,'1g+','100m-\n1g','10m-\n100m','4m-\n10m','1m-\n4m','100k-\n1m','10k-\n100k','1k-\n10k','100-\n1k','0-\n100']
    features = ['I/O Amount (bytes)','Number of Shared Files','Number of Unique Files']#,'1g+','100m-1g','10m-100m','4m-10m','1m-4m','100k-1m','10k-100k','1k-10k','100-1k','0-100']
    # Read
    operation = 'Read'
    results = pd.DataFrame()
    op_df = df[df['Operation']==operation]
    # Get lowest and highest 25% perf CoV info
    E_25 = int(op_df.shape[0]*0.10)
    l = op_df.nsmallest(E_25, ['Performance CoV'])
    l['Performance CoV Percentile'] = 25
    h = op_df.nlargest(E_25, ['Performance CoV'])
    h['Performance CoV Percentile'] = 75
    # Clean the data from outliers
    #feature_max = []
    for feature in features:
        n = '%s Z-Score'%feature
        l[n] = np.abs(stats.zscore(l[feature]))
        h[n] = np.abs(stats.zscore(h[feature]))
    op_df = h.append(l, ignore_index=True)
    for i, row in op_df.iterrows():
        for feature in features:
            n = '%s Z-Score'%feature
            if(row[n]>E):
                continue
            tmp = op_df[op_df[n]<E]
            try:
                value = row[feature]/tmp[feature].max()
            except ZeroDivisionError:
                value = 0
            if(row['Performance CoV Percentile']==25):
                p = 'Bottom 10%'
            else:
                p = 'Top 10%'
            d = {'Feature': feature, 'Value': value, 'Performance CoV Percentile': p}
            results = results.append(d, ignore_index=True)
    PROPS = {'boxprops':{'edgecolor':'black'}, 'medianprops':{'color':'black'},
            'whiskerprops':{'color':'black'},'capprops':{'color':'black'}}
    sns.boxplot(ax=axes[0], data=results, x='Feature', y='Value', hue='Performance CoV Percentile', palette='Blues', order=features, **PROPS)  
    # Write
    operation = 'Write'
    results = pd.DataFrame()
    op_df = df[df['Operation']==operation]
    # Get lowest and highest 25% perf CoV info
    E_25 = int(op_df.shape[0]*0.25)
    l = op_df.nsmallest(E_25, ['Performance CoV'])
    l['Performance CoV Percentile'] = 25
    h = op_df.nlargest(E_25, ['Performance CoV'])
    h['Performance CoV Percentile'] = 75
    # Clean the data from outliers
    for feature in features:
        n = '%s Z-Score'%feature
        l[n] = np.abs(stats.zscore(l[feature]))
        h[n] = np.abs(stats.zscore(h[feature]))
    op_df = h.append(l, ignore_index=True)
    for i, row in op_df.iterrows():
        for feature in features:
            n = '%s Z-Score'%feature
            if(row[n]>E):
                continue
            tmp = op_df[op_df[n]<E]
            try:
                value = row[feature]/tmp[feature].max()
            except ZeroDivisionError:
                value = 0
            if(row['Performance CoV Percentile']==25):
                p = 'Bottom 10%'
            else:
                p = 'Top 10%'
            d = {'Feature': feature, 'Value': value, 'Performance CoV Percentile': p}
            results = results.append(d, ignore_index=True)
    print(results)
    PROPS = {'boxprops':{'edgecolor':'black'}, 'medianprops':{'color':'black'},
            'whiskerprops':{'color':'black'},'capprops':{'color':'black'}}
    sns.boxplot(ax=axes[1], data=results, x='Feature', y='Value', hue='Performance CoV Percentile', palette='Reds', order=features, **PROPS)  
    axes[0].tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
    axes[0].tick_params(axis='x', labelrotation = 0)
    axes[1].tick_params(axis='x', labelrotation = 0)
    axes[0].yaxis.grid(color='lightgrey', linestyle=':')
    axes[0].set_axisbelow(True)
    axes[1].yaxis.grid(color='lightgrey', linestyle=':')
    axes[1].set_axisbelow(True)
    axes[0].set_ylim(0,1.01)
    axes[1].set_ylim(0,1.01)
    axes[0].set_yticks([0,0.5,1.0])
    axes[1].set_yticks([0,0.5,1.0])
    fig.text(0.42, 0.002, 'Cluster Feature', ha='center')
    fig.text(0.002, 0.58, "Norm'd Value", rotation=90, va='center')
    fig.text(0.30, 0.93, '(a) Read')
    fig.text(0.30, 0.58, '(b) Write')
    axes[0].legend(loc=(1.02,0.00),fontsize=10, title='Perf CoV Percentile', title_fontsize=10)
    axes[1].legend(loc=(1.02,0.00),fontsize=10, title='Perf CoV Percentile', title_fontsize=10)
    axes[0].set_ylabel('')
    axes[1].set_ylabel('')
    axes[0].set_xlabel('')
    axes[1].set_xlabel('')
    axes[0].set_xticklabels('')
    axes[1].set_xticklabels(labels)
    plt.savefig('./performance_cov_percentile_features.pdf')

def run_spread_temporally_examples(path_to_cluster_info, save_path):
    '''
    Identifies and gives examples for the temporal burstiness of clusters.

    Parameters
    ----------
    path_to_cluster_info: string
        Path to csv file with clustering information.

    Returns
    -------
    results: Pandas.DataFrame
        Temporal information on the runs within the clusters.
    '''
    cluster_info = pd.read_csv(path_to_cluster_info, index_col=0)
    read_clusters = cluster_info[cluster_info['Operation']=='Read']
    write_clusters = cluster_info[cluster_info['Operation']=='Write']
    read_applications = read_clusters['Application'].unique().tolist()
    write_applications = write_clusters['Application'].unique().tolist()
    results = pd.DataFrame(columns=['Cluster Number', 'Times', 'Total Time', 'Inter-Arrival Times CoV'])
    n_clusters = 0
    clusters = read_clusters[read_clusters['Application']=='vasp_gam_406746']
    no_clusters = clusters['Cluster Number'].max()
    for i in range(0, no_clusters):
        cluster = clusters[clusters['Cluster Number']==i]
        no_runs = cluster.shape[0]
        if(no_runs!=51):
            continue
        time_differences = []
        times = []
        total_time = (cluster['Time'].max() - cluster['Time'].min())/3600
        cluster = cluster.sort_values(by='Time').reset_index(drop=True)
        for j in range(0, no_runs-2):
            time_difference = abs(cluster.loc[j+1]['Time']-cluster.loc[j]['Time'])
            time_differences.append(int(time_difference))
        for j in range(0, no_runs):
            time = ((cluster.loc[j]['Time']-cluster['Time'].min())/(cluster['Time'].max()-cluster['Time'].min()))*100
            times.append(time)
        time_differences_avg    = np.average(time_differences)
        time_differences_std    = np.std(time_differences) 
        interarrival_times_cov    = (time_differences_std/time_differences_avg)*100
        if(interarrival_times_cov<100):
            print('at %d with cov of %d'%(no_runs, interarrival_times_cov))
        d = {'Cluster Number': n_clusters, 'Times': times, 'Total Time': total_time, 'Inter-Arrival Times CoV': interarrival_times_cov}
        results = results.append(d, ignore_index=True)
        n_clusters = n_clusters + 1
    max_clusters = int(results['Cluster Number'].max())
    fig, ax = plt.subplots(1, 1, sharey=True, figsize=[5,1.7])
    fig.subplots_adjust(left=0.08, right=0.99, top=.98, bottom=0.27, wspace=0.03)
    #sns.heatmap(variances, ax=axes[0])
    ax.tick_params(axis='y', which='both', bottom=False, top=False, labelbottom=False)
    #sns.scatterplot(data=results, x='Time', y='Cluster Number', hue='Operation', palette={'Read': 'skyblue', 'Write': 'maroon'}, ax=ax, alpha=1, markers='|')
    # scatter each cluster
    results = results.sort_values(by='Inter-Arrival Times CoV')
    print(results)
    n = 0
    cs = [1,3,8,9,10]
    variances = []
    total_times = []
    for idx, row in results.iterrows():
        if(not row['Cluster Number'] in cs):
            continue
        ax.scatter(x=row['Times'],y=[n]*len(row['Times']),
                        c='black', edgecolors='black', marker='|', s=180)
        mx = max(row['Times'])
        mn = min(row['Times'])
        ax.hlines(y=n, xmin=mn, xmax=mx, alpha=0.4, color='skyblue', linewidth=15)
        variances.append('%0.2f'%row['Inter-Arrival Times CoV'])
        total_times.append('%0.2f'%row['Total Time'])
        n = n + 1
    '''
    s = 0.18
    fs = .135
    x = 0.775
    fig.text(x, s, s='(f)      %s'%variances[0])
    fig.text(x, s+fs, s='(e)      %s'%variances[1])
    fig.text(x, s+2*fs, s='(d)      %s'%variances[2])
    fig.text(x, s+3*fs, s='(c)      %s'%variances[3])
    fig.text(x, s+4*fs, s='(b)      %s'%variances[4])
    fig.text(x, s+5*fs, s='(a)      %s'%variances[5])
    fig.text(0.85, 0.965, s='Inter-Arrival\nTimes CoV (%)', va='center', ha='center')
    '''
    s = 0.31
    fs = .14
    x = 0.05
    fig.text(x, s, s='1')
    fig.text(x, s+fs, s='2')
    fig.text(x, s+2*fs, s='3')
    fig.text(x, s+3*fs, s='4')
    fig.text(x, s+4*fs, s='5')
    #fig.text(x, s+5*fs, s='6')
    fig.text(0.018, 0.62, s='Cluster', va='center', ha='center', rotation=90)
    corr = spearmanr(variances, total_times)[0]
    print('Correlation: %.3f'%corr)
    ax.set_ylabel('')
    fig.text(0.26,0.025,'Percent of Cluster Time Span (%)')
    ax.xaxis.grid(color='lightgrey', linestyle=':')
    ax.set_axisbelow(True)
    plt.tick_params(left=False)
    plt.setp(ax.get_yticklabels(), visible=False)
    ax.set_ylim(-0.5, 4+0.5)
    plt.savefig(save_path)
    plt.clf()
    plt.close()


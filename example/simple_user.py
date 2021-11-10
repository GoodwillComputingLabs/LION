# input: run_info.parquet, user name
# output: figures

# plot ideas: cluster time spans, 
# list of characteristics shown that lead to greater perf var (less I/O amounts, processes share less files, more deviated z-scored runs do xyz)
# scatter + regression: amount i/o vs perf var zscore

from os import read
import pandas as pd
from pandas.core.frame import DataFrame
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns

all_df = pd.read_parquet('/scratch/costa.em/total/cluster_info.parquet')
top_apps = all_df['Application'].value_counts().head(5).to_dict().keys()
for app in top_apps:
    mask = all_df.loc[:,'Application'] == app 
    pos = np.flatnonzero(mask)
    df = all_df.iloc[pos]

    cluster_nos = df.loc[:,'Cluster Number'].unique().tolist()
    zscored_df = DataFrame()
    for cluster_no in cluster_nos:
        mask = df.loc[:,'Cluster Number'] == cluster_no 
        pos = np.flatnonzero(mask)
        tmp = df.iloc[pos]
        # now add all interesting metrics
        tmp.loc[:,'Performance Z-Score'] = stats.zscore(tmp['Performance'])
        tmp.loc[:,'Run Time Span'] = tmp.loc[:,'End Time']-tmp.loc[:,'Start Time']
        zscored_df = zscored_df.append(tmp,ignore_index=True)

    df = zscored_df
    read_info = df[df['Operation']=='Read']['Performance Z-Score'].tolist()
    read_median = np.median(read_info)
    read_mean   = np.mean(read_info)
    read_bins = np.arange(-3.,3., 0.01)
    hist = np.histogram(read_info, bins=read_bins)[0]
    cdf_read = np.cumsum(hist)
    cdf_read = [x/cdf_read[-1] for x in cdf_read]
    write_info = df[df['Operation']=='Write']['Performance Z-Score'].tolist()
    write_median = np.median(write_info)
    write_mean   = np.mean(write_info)
    write_bins = np.arange(-3.,3., 0.01)
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
    ax.set_xlabel('Performance Z-Scores')
    ax.yaxis.grid(color='lightgrey', linestyle=':')
    ax.set_axisbelow(True)
    ax.set_xlim(-3, 3)
    # Add vertical lines for medians
    ax.axvline(read_median, color='skyblue', zorder=0, linestyle='--', linewidth=2)
    ax.axvline(write_median, color='maroon', zorder=0, linestyle=':', linewidth=2)
    # Add legend
    ax.legend(loc='lower right', fancybox=True)
    plt.savefig('./figures/single_user/perf_zscores_CDF.pdf')
    plt.clf()
    plt.close()

    df = zscored_df
    fig, axes = plt.subplots(1, 2, sharey=True, figsize=[5, 2.5])
    x_axis = 'I/O Amount'
    # First, plot read
    op = 'Read'
    mask = df.loc[:,'Operation'] == op 
    pos = np.flatnonzero(mask)
    df = df.iloc[pos]
    x = df.loc[:,x_axis]
    y = df.loc[:,'Performance Z-Score']
    axes[0].scatter(x, y, marker='.', color='skyblue',label=op)
    m, b = np.polyfit(x, y, 1)
    axes[0].plot(x, m*x+b, color='skyblue',ls='--',lw=2)
    axes[0].set_ylim(-3, 3)
    axes[0].set_yticks(np.arange(-3,4,1))
    axes[0].set_title(op)
    axes[0].set_xlabel(" ")
    axes[0].margins(0)
    axes[0].set_xlim([np.min(x), np.max(x)])
    # Now, write
    op = 'Write'
    df = zscored_df
    mask = df.loc[:,'Operation'] == op
    pos = np.flatnonzero(mask)
    df = df.iloc[pos]
    x = df.loc[:, x_axis]
    y = df.loc[:,'Performance Z-Score']
    axes[1].scatter(x, y, marker='.', color='maroon',label=op)
    m, b = np.polyfit(x, y, 1)
    axes[1].plot(x, m*x+b, color='maroon',ls=':',lw=2)
    axes[1].set_ylim(-3, 3)
    axes[1].set_title(op)
    axes[1].set_xlabel(" ")
    fig.subplots_adjust(left=0.11, bottom=0.25, right=.98, top=.97, wspace=0.05, hspace=0)
    fig.text(0.5, 0.01, x_axis, ha='center', va='bottom')
    fig.text(0.01, 0.6, 'Performance Score', ha='left', va='center', rotation=90)
    #plt.tight_layout()
    plt.savefig('./figures/single_user/io-amount_v_perf_var.pdf')
    plt.clf()
    plt.close()

    df = zscored_df
    fig, axes = plt.subplots(1, 2, sharey=True, figsize=[5, 2.5])
    x_axis = 'Run Time Span'
    # First, plot read
    op = 'Read'
    mask = df.loc[:,'Operation'] == op 
    pos = np.flatnonzero(mask)
    df = df.iloc[pos]
    x = df.loc[:,x_axis]
    y = df.loc[:,'Performance Z-Score']
    axes[0].scatter(x, y, marker='.', color='skyblue',label=op)
    m, b = np.polyfit(x, y, 1)
    axes[0].plot(x, m*x+b, color='skyblue',ls='--',lw=2)
    axes[0].set_ylim(-3, 3)
    axes[0].set_yticks(np.arange(-3,4,1))
    axes[0].set_title(op)
    axes[0].set_xlabel(" ")
    axes[0].margins(0)
    axes[0].set_xlim([np.min(x), np.max(x)])
    # Now, write
    op = 'Write'
    df = zscored_df
    mask = df.loc[:,'Operation'] == op
    pos = np.flatnonzero(mask)
    df = df.iloc[pos]
    x = df.loc[:, x_axis]
    y = df.loc[:,'Performance Z-Score']
    axes[1].scatter(x, y, marker='.', color='maroon',label=op)
    m, b = np.polyfit(x, y, 1)
    axes[1].plot(x, m*x+b, color='maroon',ls=':',lw=2)
    axes[1].set_ylim(-3, 3)
    axes[1].set_title(op)
    axes[1].set_xlabel(" ")
    fig.subplots_adjust(left=0.11, bottom=0.2, right=.98, top=.97, wspace=0.05, hspace=0)
    fig.text(0.5, 0.01, x_axis, ha='center', va='bottom')
    fig.text(0.01, 0.6, 'Performance Score', ha='left', va='center', rotation=90)
    #plt.tight_layout()
    plt.savefig('./figures/single_user/run-span_v_perf-var.pdf')
    plt.clf()
    plt.close()

    # collect: date, date by week, time of day, day of week (Monday, etc)
    df = zscored_df

    df.loc[:,'Datetime'] = pd.to_datetime(df.loc[:,'Start Time'],unit='s')

    df.loc[:,'Day'] = df.loc[:,'Datetime'].dt.to_period('D').dt.to_timestamp()
    day_range = pd.date_range(min(df.loc[:,'Day']),max(df.loc[:,'Day']),freq='D').tolist()
    d = {}
    for i in range(1,len(day_range)+1):
        d[day_range[i-1]] = i
    df.loc[:,'Day'] = df.loc[:,'Day'].map(d)

    df.loc[:,'Week'] = df.loc[:,'Datetime'].dt.to_period('W').dt.to_timestamp()
    week_range = pd.date_range(min(df.loc[:,'Week']),max(df.loc[:,'Week']),freq='W').tolist()
    if week_range==[]:
        week_range = [min(df.loc[:,'Week'])]
    d = {}
    for i in range(1,len(week_range)+1):
        d[week_range[i-1]] = i
    df.loc[:,'Week'] = df.loc[:,'Week'].map(d)

    df.loc[:,'Month'] = df.loc[:,'Datetime'].dt.to_period('M').dt.to_timestamp()
    month_range = pd.date_range(min(df.loc[:,'Month']),max(df.loc[:,'Month']),freq='M').tolist()
    if month_range==[]:
        month_range = [min(df.loc[:,'Month'])]
    d = {}
    for i in range(1,len(month_range)+1):
        d[month_range[i-1]] = i
    df.loc[:,'Month'] = df.loc[:,'Month'].map(d)

    df.loc[:,'Day of Week'] = df.loc[:,'Datetime'].dt.day_name()
    df.loc[:,'Hour of Day'] = df.loc[:,'Datetime'].dt.hour

    zscored_df = df

    fig, axes = plt.subplots(3, 2, sharey=True, figsize=[5, 4])
    x_axes = ['Day','Week','Month']
    for n in range(len(x_axes)):
        x_axis = x_axes[n]
        # First, plot read
        op = 'Read'
        df = zscored_df
        mask = df.loc[:,'Operation'] == op 
        pos = np.flatnonzero(mask)
        df = df.iloc[pos]
        x = df.loc[:,x_axis]
        y = df.loc[:,'Performance Z-Score']
        #axes[n][0].scatter(x, y, marker='.', color='skyblue',label=op)
        #m, b = np.polyfit(x, y, 1)
        #axes[n][0].plot(x, m*x+b, color='skyblue',ls='--',lw=2)
        sns.regplot(x, y, ax=axes[n][0], color='skyblue',label=op)
        axes[n][0].set_ylim(-3, 3)
        axes[n][0].set_yticks(np.arange(-3,4,1))
        axes[n][0].set_xticks(np.arange(min(x),max(x)+1,1))
        #axes[n][0].set_title(op)
        axes[n][0].set_xlabel(" ")
        axes[n][0].margins(0)
        axes[n][0].set_xlim([np.min(x), np.max(x)])
        # Now, write
        op = 'Write'
        df = zscored_df
        mask = df.loc[:,'Operation'] == op
        pos = np.flatnonzero(mask)
        df = df.iloc[pos]
        x = df.loc[:, x_axis]
        y = df.loc[:,'Performance Z-Score']
        #axes[n][1].scatter(x, y, marker='.', color='maroon',label=op)
        #m, b = np.polyfit(x, y, 1)
        #axes[n][1].plot(x, m*x+b, color='maroon',ls=':',lw=2)
        sns.regplot(x, y, ax=axes[n][0], color='maroon',label=op)
        axes[n][1].set_ylim(-3, 3)
        axes[n][1].set_xticks(np.arange(min(x),max(x)+1,1))
        axes[n][1].set_xlim([np.min(x), np.max(x)])
        #axes[n][1].set_title(op)
        axes[n][1].set_xlabel(" ")

    fig.subplots_adjust(left=0.11, bottom=0.12, right=.98, top=.95, wspace=0.1, hspace=0.35)
    fig.text(0.5, 0.01, x_axis, ha='center', va='bottom')
    fig.text(0.01, 0.6, 'Performance Score', ha='left', va='center', rotation=90)
    fig.text(0.31, 0.99, 'Read', ha='center', va='top')
    fig.text(0.78, 0.99, 'Write', ha='center', va='top')
    plt.savefig('./figures/single_user/temp_v_perf_var.pdf')
    plt.clf()
    plt.close()

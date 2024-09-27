# input: run_info.parquet, user name
# output: figures

# plot ideas: cluster time spans, 
# list of characteristics shown that lead to greater perf var (less I/O amounts, processes share less files, more deviated z-scored runs do xyz)
# scatter + regression: amount i/o vs perf var zscore

from os import read
import os
import pandas as pd
from pandas.core.frame import DataFrame
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns
pd.set_option('mode.chained_assignment', None)

all_df = pd.read_parquet('/scratch/costa.em/total/cluster_info.parquet')
top_apps = all_df['Application'].value_counts().head(5).to_dict().keys()
for app in top_apps:
    fig_path = os.path.join('./figures/single_user',app)
    if not os.path.exists(fig_path):
        os.mkdir(fig_path)
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
    #'''
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
    plt.savefig(os.path.join(fig_path,'perf_zscores_CDF.jpg'))
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
    plt.savefig(os.path.join(fig_path,'io-amount_v_perf_var.jpg'))
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
    plt.savefig(os.path.join(fig_path,'run-span_v_perf-var.jpg'))
    plt.clf()
    plt.close()
    #'''
    # collect: date, date by week, time of day, day of week (Monday, etc)
    df = zscored_df

    df.loc[:,'Datetime'] = pd.to_datetime(df.loc[:,'Start Time'],unit='s')

    df.loc[:,'Day'] = df.loc[:,'Datetime'].dt.to_period('D').dt.to_timestamp()
    day_range = pd.date_range(min(df.loc[:,'Day'])-pd.DateOffset(days=1),max(df.loc[:,'Day']),freq='D').tolist()
    d = {}
    for i in range(1,len(day_range)+1):
        d[day_range[i-1]] = int(i)
    df.loc[:,'Day'] = df.loc[:,'Day'].map(d)

    df.loc[:,'Week'] = df.loc[:,'Datetime'].dt.to_period('W').dt.to_timestamp()
    week_range = pd.date_range(min(df.loc[:,'Week'])-pd.DateOffset(weeks=1),max(df.loc[:,'Week']),freq='W').tolist()
    if week_range==[]:
        week_range = [min(df.loc[:,'Week'])]
    d = {}
    for i in range(1,len(week_range)+1):
        d[week_range[i-1]+pd.Timedelta(days=1)] = int(i)
    df.loc[:,'Week'] = df.loc[:,'Week'].map(d)

    df.loc[:,'Month'] = df.loc[:,'Datetime'].dt.to_period('M').dt.to_timestamp()
    month_range = pd.date_range(min(df.loc[:,'Month'])-pd.DateOffset(months=1),max(df.loc[:,'Month']),freq='M').tolist()
    if month_range==[]:
        month_range = [min(df.loc[:,'Month'])]
    d = {}
    for i in range(1,len(month_range)+1):
        d[month_range[i-1]+pd.Timedelta(days=1)] = i
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
        axes[n][0].scatter(x, y, marker='.', color='skyblue',label=op)
        axes[n][0].set_ylim(-3, 3)
        axes[n][0].set_ylabel('')
        axes[n][0].set_yticks(np.arange(-3,4,1))
        axes[n][0].set_xlabel(" ")
        axes[n][0].margins(0)
        # Now, write
        op = 'Write'
        df = zscored_df
        mask = df.loc[:,'Operation'] == op
        pos = np.flatnonzero(mask)
        df = df.iloc[pos]
        x = df.loc[:, x_axis]
        y = df.loc[:,'Performance Z-Score']
        axes[n][1].scatter(x, y, marker='.', color='maroon',label=op)
        axes[n][1].set_ylim(-3, 3)
        axes[n][1].set_ylabel('')
    fig.subplots_adjust(left=0.11, bottom=0.09, right=.98, top=.95, wspace=0.1, hspace=0.55)
    fig.text(0.31, 0.65, x_axes[0], ha='center', va='bottom')
    fig.text(0.31, 0.32, x_axes[1], ha='center', va='bottom')
    fig.text(0.31, 0.00, x_axes[2], ha='center', va='bottom')
    fig.text(0.78, 0.65, x_axes[0], ha='center', va='bottom')
    fig.text(0.78, 0.32, x_axes[1], ha='center', va='bottom')
    fig.text(0.78, 0.00, x_axes[2], ha='center', va='bottom')
    fig.text(0.01, 0.5, 'Performance Score', ha='left', va='center', rotation=90)
    fig.text(0.31, 0.99, 'Read', ha='center', va='top')
    fig.text(0.78, 0.99, 'Write', ha='center', va='top')
    plt.savefig(os.path.join(fig_path,'temp_v_perf_var.jpg'))
    plt.clf()
    plt.close()

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
        df = df[['Performance Z-Score',x_axis]].groupby(x_axis, as_index=False).mean()
        x = df.loc[:,x_axis]
        y = df.loc[:,'Performance Z-Score']
        #axes[n][0].scatter(x, y, marker='.', color='skyblue',label=op)
        #axes[n][0].set_ylim(-3, 3)
        for i in range(0,len(y)):
            alpha = 0.2
            if y[i]>1:
                axes[n][0].axvspan(x[i]-0.5, x[i]+0.5, alpha=alpha, color='green', lw=0, zorder=2)
            elif y[i]<-1:
                axes[n][0].axvspan(x[i]-0.5, x[i]+0.5, alpha=alpha, color='yellow', lw=0, zorder=2)
            elif y[i]>0.5:
                l = x[i]-1
                while(l not in x and l>min(x)):
                    l = l-1
                r = x[i]+1
                while(r not in x and r<max(x)):
                    r = r+1
                try:
                    if l!=min(x) and r!=max(x) and y[l]>0.5 and y[r]>0.5:
                        axes[n][0].axvspan(x[l]-0.5, x[r]+0.5, alpha=alpha, color='yellow', lw=0, zorder=2)
                except KeyError:
                    continue
            elif y[i]<-0.5:
                l = x[i]-1
                while(l not in x and l>min(x)):
                    l = l-1
                r = x[i]+1
                while(r not in x and r<max(x)):
                    r = r+1
                try:
                    if l!=min(x) and r!=max(x) and y[l]<-0.5 and y[r]<-0.5:
                        axes[n][0].axvspan(x[l]-0.5, x[r]+0.5, alpha=alpha, color='yellow', lw=0, zorder=2)
                except KeyError:
                    continue
            #for r in np.arange(0.5,3.5,.5):
            #    alpha = alpha+0.1
            #    if y[i]>r:
            #        axes[n][0].axvspan(x[i]-0.5, x[i]+0.5, alpha=alpha, color='green', lw=0, zorder=2)
            #    elif y[i]<-r:
            #        axes[n][0].axvspan(x[i]-0.5, x[i]+0.5, alpha=alpha, color='yellow', lw=0, zorder=2)
        axes[n][0].scatter(x, y, marker='.', color='skyblue',label=op, zorder=1)
        axes[n][0].set_ylim(-3, 3)
        axes[n][0].set_ylabel('')
        axes[n][0].set_yticks(np.arange(-3,4,1))
        axes[n][0].yaxis.grid(color='lightgrey', linestyle=':')
        axes[n][0].axhline(y=0, color='darkgrey', linestyle=':')
        axes[n][0].set_axisbelow(True)
        axes[n][0].set_xlabel(" ")
        axes[n][0].margins(0)
        # Now, write
        op = 'Write'
        df = zscored_df
        mask = df.loc[:,'Operation'] == op
        pos = np.flatnonzero(mask)
        df = df.iloc[pos][['Performance Z-Score',x_axis]].groupby(x_axis, as_index=False).mean()
        x = df.loc[:, x_axis]
        y = df.loc[:,'Performance Z-Score']
        #axes[n][1].scatter(x, y, marker='.', color='maroon',label=op)
        #axes[n][1].set_ylim(-3, 3)
        for i in range(0,len(y)):
            alpha = 0.2
            if y[i]>1:
                axes[n][1].axvspan(x[i]-0.5, x[i]+0.5, alpha=alpha, color='green', lw=0, zorder=2)
            elif y[i]<-1:
                axes[n][1].axvspan(x[i]-0.5, x[i]+0.5, alpha=alpha, color='yellow', lw=0, zorder=2)
            elif y[i]>0.5:
                l = x[i]-1
                while(l not in x and l>min(x)):
                    l = l-1
                r = x[i]+1
                while(r not in x and r<max(x)):
                    r = r+1
                try:
                    if l!=min(x) and r!=max(x) and y[l]>0.5 and y[r]>0.5:
                         axes[n][1].axvspan(x[l]-0.5, x[r]+0.5, alpha=alpha, color='yellow', lw=0, zorder=2)
                except KeyError:
                    continue
            elif y[i]<-0.5:
                l = x[i]-1
                while(l not in x and l>min(x)):
                    l = l-1
                r = x[i]+1
                while(r not in x and r<max(x)):
                    r = r+1
                try:
                    if l!=min(x) and r!=max(x) and y[l]<-0.5 and y[r]<-0.5:
                        axes[n][1].axvspan(x[l]-0.5, x[r]+0.5, alpha=alpha, color='yellow', lw=0, zorder=2)
                except KeyError:
                    continue
            #for r in np.arange(0.5,3.5,.5):
            #    alpha = alpha+0.1
            #    if y[i]>r:
            #        axes[n][1].axvspan(x[i]-0.5, x[i]+0.5, alpha=alpha, color='green', lw=0, zorder=2)
            #    elif y[i]<-r:
            #        axes[n][1].axvspan(x[i]-0.5, x[i]+0.5, alpha=alpha, color='yellow', lw=0, zorder=2)
        axes[n][1].scatter(x, y, marker='.', color='maroon',label=op, zorder=1)
        axes[n][1].set_ylim(-3, 3)
        axes[n][1].set_ylabel('')
        axes[n][1].yaxis.grid(color='lightgrey', linestyle=':')
        axes[n][1].set_axisbelow(True)
        axes[n][1].axhline(y=0, color='darkgrey', linestyle=':')
    fig.subplots_adjust(left=0.11, bottom=0.09, right=.98, top=.95, wspace=0.1, hspace=0.55)
    fig.text(0.31, 0.65, x_axes[0], ha='center', va='bottom')
    fig.text(0.31, 0.32, x_axes[1], ha='center', va='bottom')
    fig.text(0.31, 0.00, x_axes[2], ha='center', va='bottom')
    fig.text(0.78, 0.65, x_axes[0], ha='center', va='bottom')
    fig.text(0.78, 0.32, x_axes[1], ha='center', va='bottom')
    fig.text(0.78, 0.00, x_axes[2], ha='center', va='bottom')
    fig.text(0.01, 0.5, 'Avg Performance Score', ha='left', va='center', rotation=90)
    fig.text(0.31, 0.99, 'Read', ha='center', va='top')
    fig.text(0.78, 0.99, 'Write', ha='center', va='top')
    plt.savefig(os.path.join(fig_path,'temp_v_perf_var-AVG.jpg'))
    plt.clf()
    plt.close()
    #'''
    df = zscored_df
    read_df = df[df['Operation']=='Read']
    write_df = df[df['Operation']=='Write']
    fig, axes = plt.subplots(1, 2, sharey=True, figsize=[5,1.9])
    fig.subplots_adjust(left=0.16, right=0.990, top=0.96, bottom=0.38, wspace=0.03)
    order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    labels = ['Mo', 'Tu', 'We', 'Th', 'Fr', 'Sa', 'Su']
    sns.violinplot(ax=axes[0], x='Day of Week', y='Performance Z-Score', data=read_df, order=order, color='skyblue', inner='quartile', edgecolor='black')
    sns.violinplot(ax=axes[1], x='Day of Week', y='Performance Z-Score', data=write_df, order=order, color='maroon', inner='quartile')
    axes[0].set_ylabel('')
    axes[1].set_ylabel('')
    fig.text(0.37, 0.135, '(a) Read', ha='center')
    fig.text(0.78, 0.135, '(b) Write', ha='center')
    fig.text(0.58, 0.02, 'Day of Week', ha='center')
    fig.text(0.001, 0.65, "Performance\nZ-Score", rotation=90, va='center', multialignment='center')
    axes[0].set_xlabel('')
    axes[1].set_xlabel('')
    axes[0].set_xticklabels(labels)
    axes[1].set_xticklabels(labels)
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
    plt.savefig(os.path.join(fig_path,'dow_v_perf_var.jpg'))
    plt.clf()
    plt.close()
    #'''

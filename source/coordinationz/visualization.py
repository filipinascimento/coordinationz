import pandas as pd
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.colors as pltc
from tqdm import tqdm
import os
import matplotlib.ticker as tck
import math

def ccdf(parameters):
    '''
    Plots ccdf for data
    
    :param parameters: parameters to set for the plot
    '''
    
    # {
    #     'data': df,
    #     'fontsize': 14,
    #     'complementary': True,
    #     'columns': [
    #         {'column': ''
    #          'label': '',
    #         },{
    #         'column': '',
    #          'label': ''
    #         }
    #     ],
    #     'xlabel': '',
    #     'ylabel': '',
    #     'legend_location': '',
    #     'log_yscale': True,
    #     'log_xscale': True,
    #     'save': {
    #         'path': '',
    #         'filename': ''
    #     },
        # 'random_color': False
    # }
    
    keys = parameters.keys()
    if 'figsize' in keys:
        size = parameters['figsize']
    else:
        size = (8,8)
        
    fig, ax = plt.subplots(figsize=size)
    # fig = plt.figure(figsize=size)

    # Add an axes at position [left, bottom, width, height]
    # where each value is between 0 and 1

    # ax = fig.add_axes([0.2, 0.2, 0.9, 0.9])
    
    fontsize = parameters['fontsize']
    colors = ['red', 'blue', 'green', 'orange', 'olive', 'pink', 'lime', 'maroon']
    total_columns = len(parameters['columns'])
    
    if parameters['random_color'] == True:
        all_colors =  [k for k,v in pltc.cnames.items()]
        colors = sample(all_colors, total_columns)
    
    symbols = ['.', 'o', '+', 'x', '*', 'v', '^', '>']
    
    i = 0
    cmap = plt.cm.get_cmap('hsv', total_columns)
    max_n = 0
    for data in parameters['data']:
        column = parameters['columns'][i]['column']
        data = parameters['data'][i][column]
        label = parameters['columns'][i]['label']
        
        if 'color' in parameters['columns'][i].keys():
            assigned_color = parameters['columns'][i]['color']
        else:
            assigned_color = colors[i]
        
        if max_n < max(data):
            max_n = max(data)
            
        sns.ecdfplot(data, 
                     complementary=parameters['complementary'],
                     label=label,
                     # marker=symbols[i],
                     color=assigned_color,
                     ax=ax,
                     linewidth=2,)

        i = i + 1
        
    if 'log_yscale' in keys and parameters['log_yscale'] == True:
        ax.set_yscale('log')
        # ax.yaxis.set_minor_locator(AutoMinorLocator())
       
        
    if 'log_xscale' in keys and parameters['log_xscale'] == True:
        ax.set_xscale('log')
        
        n = int(math.log10(max_n) + 1)
        all_ticks = []
        for x in range(0, int(n)):
            for i in range(1, 10):
                all_ticks.append(i * (10**x))
        
        ax.xaxis.set_minor_locator(tck.FixedLocator(all_ticks))


    if parameters['complementary'] == True:
        parameters['ylabel'] = 'CCDF'
        
    ax.set_xlabel(parameters['xlabel'], 
                  fontsize=fontsize)
    ax.set_ylabel(parameters['ylabel'], 
                  fontsize=fontsize)
    
    if 'tick_size' in keys:
        tick_size = parameters['tick_size']
    else:
        tick_size = fontsize
        
        
    ax.tick_params(axis='both', 
                   which='both', 
                   labelsize=tick_size,
                   labelbottom=True
                  )

    # ax.xaxis.set_minor_locator(tck.AutoMinorLocator())

    if 'legend_location' in keys:
        if 'legend_font' in keys:
            legend_font = parameters['legend_font']
        else:
            legend_font = fontsize
            
        ax.legend(loc=parameters['legend_location'], 
                  frameon=True, 
                  fontsize=legend_font
                 )
        
    if 'legend_lower' in keys:
        box = ax.get_position()
        ax.set_position([box.x0, box.y0 + box.height * 0.1,
                         box.width, box.height * 0.9])

        # Put a legend below current axis
        ax.legend(loc='upper center', 
                  bbox_to_anchor=(1, -0.06),
                  fancybox=True, 
                  shadow=True, ncol=3)
    
    
        # ax.xaxis.set_minor_locator(AutoMinorLocator())

    if 'title' in keys:
        plt.title(parameters['title'])
        
        
    if 'figure_text' in keys:
        plt.text(parameters['figure_text_x'], 
                 parameters['figure_text_y'], 
                 parameters['figure_text'], 
                 fontsize=parameters['figure_font'],
                 ha="center", 
                 va="center", 
                )
        
    if 'subplot_adjust' in keys:
        plt.subplots_adjust(
            bottom=parameters['subplot_adjust']
        )
   
    fig.tight_layout()
    
    if 'save' in keys:
        path = parameters['save']['path']
        filename = parameters['save']['filename']
        fig_path = os.path.join(path, filename)
        
        print(fig_path)
        
        fig.savefig(fig_path, 
              facecolor='white', 
              transparent=False)
        
    plt.show()
    
    
def scatter_plot(parameters):
    '''
    Plots the scatterplot
    
    :param parameters: parameters for the plot
    '''
    
    # parameters =  {
    #     'data': df_jaccard,
    #     'fontsize': 14,
    #     'columns': {
    #         'x': 'ratio',
    #         'y': 'count_total_replies',
    #     },
    #     'alpha': 0.5,
    #     'marker_size': 5,
    #     'marker': None,
    #     'xlabel': 'Jaccard coefficent \n (for each IO account, each poster)',
    #     'ylabel': 'Number of daily tweets from poster  ',
    #     'legend_location': '',
    #     'log_yscale': False,
    #     'log_xscale': False,
    #     'bins': None,
    #     'title': f'{year}_{campaign}_per_poster_per_tweet_1day',
    #     'save': {
    #         'path': f'{time_plot_path}',
    #         'filename': f'{year}_{campaign}_jaccard_1day.png'
    #     },
    # }
    
    keys = parameters.keys()
    
    if 'size' in keys:
        size = parameters['size']
    else:
        size = (8,8)
        
    fig, ax = plt.subplots(figsize=size)
    fontsize = parameters['fontsize']
    
    colors = ['blue', 'red', 'green', 'orange', 'olive', 'pink', 'lime', 'maroon']
    symbols = ['.', 'o', '+', 'x', '*', 'v', '^', '>']
    
    x_column = parameters['columns']['x']
    y_column = parameters['columns']['y']
    data = parameters['data']
    color = colors[0]
    
    alpha = parameters['alpha'] if 'alpha' in keys else 0.5
    marker_size = parameters['marker_size'] if 'marker_size' in keys else 3
    
    if 'marker' in keys:
        marker = symbols[1]
    else:
        marker = symbols[0]
        
    ax.scatter(data[x_column], 
               data[y_column], 
               marker_size, 
               c=color, 
               alpha=alpha, 
               marker=marker,
           label="Luck")

    ax.set_xlabel(parameters['xlabel'], 
                  fontsize=fontsize)
    ax.set_ylabel(parameters['ylabel'], 
                  fontsize=fontsize)

    ax.tick_params(axis='both', labelsize=fontsize) 
    
    if 'log_yscale' in keys and parameters['log_yscale'] == True:
        ax.set_yscale('log')
    if 'log_xscale' in keys and parameters['log_xscale'] == True:
        ax.set_xscale('log')
        
    if 'title' in keys:
        plt.title(parameters['title'])

    if 'save' in keys:
        path = parameters['save']['path']
        filename = parameters['save']['filename']
        
        file_path = os.path.join(path, filename)
        
        fig.savefig(f'{file_path}', 
              facecolor='white', 
              transparent=False)
        
    plt.show()
    
    
    
def plot_histogram(parameters):
    '''
    Plots histogram
    :param parameters: parameters for the plot
    '''
       
    # parameters =  {
    #     'data': df,
    #     'fontsize': 14,
    #     'columns': [
    #         {'column': 'count',
    #          'label': 'Count of Replies',
    #         }
    #     ],
    #     'xlabel': '',
    #     'ylabel': '',
    #     'legend_location': '',
    #     'log_yscale': True,
    #     'log_xscale': True,
    #     'bins': 60,
    #     'save': {
    #         'path': '',
    #         'filename': ''
    #     },
    #     'title': ''
    # }
    
    
    num_bins = parameters['bins']
    column = parameters['columns'][0]['column']
    df = parameters['data']
    
    if 'size' in parameters:
        size = parameters['size']
    else:
        size = (10, 10)
    
    if 'fontsize' in parameters:
        fontsize = parameters['fontsize']
    else:
        fontsize = 14
    
    if parameters['bins'] == None:
        num_bins = df[column].nunique()
        
    if parameters['log_xscale'] == True:
        num_bins=np.logspace(start=np.log10(min(df[column])), 
                             stop=np.log10(max(df[column])),
                             num=num_bins
                            )
    fig, ax = plt.subplots(figsize=size)
    
    n = ax.hist(df[column],
                bins=num_bins, 
               )
    ax.set_xlabel(parameters['xlabel'],
                  fontsize=fontsize
                 )
    ax.set_ylabel(parameters['ylabel'],
                  fontsize=fontsize
                 )
    
    if parameters['log_yscale'] == True:
        plt.yscale('log')
    if parameters['log_xscale'] == True:
        plt.xscale('log')
        
    plt.title(parameters['title'],
              fontsize=fontsize
             )
    
    if 'save' in parameters:
        plot_path = parameters['save']['path']
        title = parameters['save']['filename']

        path = os.path.join(plot_path, title)
        fig.savefig(f'{path}', 
                  facecolor='white', 
                  transparent=False)
    plt.show()
    
    
def line_plot(parameters):
    '''
    Plots ccdf for data
    
    :param parameters: parameters to set for the plot
    '''
    
    # {
    #     'data': df,
    #     'fontsize': 14,
    #     'complementary': True,
    #     'columns': [
    #         {'column': ''
    #          'label': '',
    #         },{
    #         'column': '',
    #          'label': ''
    #         }
    #     ],
        # 'x': '',
    #     'xlabel': '',
    #     'ylabel': '',
    #     'legend_location': '',
    #     'log_yscale': True,
    #     'log_xscale': True,
    #     'save': {
    #         'path': '',
    #         'filename': ''
    #     },
        # 'random_color': False
    # }
    
    keys = parameters.keys()
    
    if 'size' in keys:
        size = parameters['size']
    else:
        size = (8,8)
        
    fig, ax = plt.subplots(figsize=size)
    
    fontsize = parameters['fontsize']
    colors = ['red', 'blue', 'black', 'orange', 'olive', 'pink', 'lime', 'maroon']
    total_columns = len(parameters['columns'])
    
    if parameters['random_color'] == True:
        all_colors =  [k for k,v in pltc.cnames.items()]
        colors = sample(all_colors, total_columns)
    
    symbols = ['x', 'o', '+', '*', '.', 'v', '^', '>']
    
    i = 0
    cmap = plt.cm.get_cmap('hsv', total_columns)
    x = parameters['data'][parameters['x']]
    for i, column in enumerate(parameters['columns']):
        data = parameters['data'][column['column']]
        label = column['label']
            
        ax.plot(x,
                data, 
                     # complementary=parameters['complementary'],
                 label=label,
                 marker=symbols[i],
                 color=colors[i],
                 # ax=ax,
                 linewidth=2,
                markersize=14
               )

    if 'x_ticks' in keys:
        labels = parameters['data'][parameters['x_ticks']].tolist()
        plt.xticks(x)
        
        ax.set_xticklabels(labels, fontsize=fontsize + 2)
    
    ax.set_xlabel(parameters['xlabel'], 
                  fontsize=fontsize + 2)
    ax.set_ylabel(parameters['ylabel'], 
                  fontsize=fontsize + 2)

    ax.tick_params(axis='both', labelsize=fontsize) 
    
    if 'legend_location' in keys:
        legend_size = parameters['legend_size']
        ax.legend(loc=parameters['legend_location'], 
                  frameon=True, fontsize=legend_size)
        
    if 'legend_lower' in keys:
        box = ax.get_position()
        ax.set_position([box.x0, box.y0 + box.height * 0.1,
                         box.width, box.height * 0.9])

        # Put a legend below current axis
        ax.legend(loc='upper center', 
                  bbox_to_anchor=(1, -0.06),
                  fancybox=True, 
                  shadow=True, ncol=3)
    
    if 'log_yscale' in keys and parameters['log_yscale'] == True:
        ax.set_yscale('log')
    if 'log_xscale' in keys and parameters['log_xscale'] == True:
        ax.set_xscale('log')
    
    ax.set_aspect('auto')

    if 'title' in keys:
        plt.title(parameters['title'])
        
    fig.tight_layout()

    if 'save' in keys:
        path = parameters['save']['path']
        filename = parameters['save']['filename']
        print(f'{path}/{filename}')
        fig_path = os.path.join(path, filename)
        print(fig_path)
        fig.savefig(fig_path, 
              facecolor='white', 
              transparent=False)
        
    plt.show()
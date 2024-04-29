import pandas as pd
import numpy as np
import warnings
import datetime
import gzip
import glob
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
import itertools
import difflib
import json
import os


def read_ops_control_data(path, campaign, year):
    '''
    Reads the control and influence operation data
    '''
    
    file_path = f'{path}/all_tweets/{year}/{campaign}'
    filename_ops = f'{file_path}/{campaign}_tweets.pkl.gz'
    filename_control = f'{file_path}/DriversControl/{campaign}_control.pkl.gz'

    df_ops = pd.read_pickle(filename_ops)
    df_control = pd.read_pickle(filename_control)

    data = {
            'ops': df_ops,
            'control': df_control
            }
    
    return data


def similarity_in_display_name(df, campaign, key='ops', 
                               fields=['user_screen_name', 'user_display_name']):
    '''
    Calculates the similarity between two screen name and display name
    '''
    
    ratios = []
    
    for field in fields:
        names = df[field].unique()
        combinations = itertools.combinations(names, 2)

        for combination in combinations:
            ratio = round(difflib.SequenceMatcher(None, 
                                                  combination[0],
                                                  combination[1]).ratio(), 2)

            ratios.append([combination, ratio, campaign, key, field])
        
    return ratios


def combine_similarity_in_name(campaigns, path, save_path, filename):
    '''
    Combine all the data from all campaigns
    '''
    
    all_ratios = []
    for year in campaigns:
        for campaign in campaigns[year]:
            print(f'\n ----------- {campaign} starts ---------- \n')
            
            data = read_ops_control_data(path, campaign, year)

            for key in data:
                ratios = similarity_in_display_name(data[key], campaign, key)

                all_ratios.extend(ratios)
                
            print(f'\n ----------- {campaign} ends ---------- \n')
            
    (pd.DataFrame(data=all_ratios, 
                  columns=['name', 'ratio', 'campaign', 'type', 'field'])
     .to_pickle(f'{save_path}/{filename}.pkl.gz'))



if __name__ == "__main__":
    
    path = '/N/slate/potem/data/derived'
    all_data = '/N/slate/potem/YYYY_MM'
    plot_path = '/N/u/potem/Carbonate/Projects/infoOps-strategy/plots'
    save_path = '/N/slate/potem/data/derived/combined'

    campaigns = {
        # '2021_12': ['Venezuela_0621', ],
        '2021_12': ['CNHU_0621', 'CNCC_0621', 'MX_0621', 
                    'REA_0621', 'RNA_0621', 'Tanzania_0621', 
                    'uganda_0621', 'Venezuela_0621'],
        '2020_12': ['armenia_202012', 'GRU_202012', 'IRA_202012', 'iran_202012']
    }


    combine_similarity_in_name(campaigns, path=path, 
                               save_path=save_path, filename='similarity_user_name')

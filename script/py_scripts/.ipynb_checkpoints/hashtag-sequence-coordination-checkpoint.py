import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
import matplotlib
import datetime
from tqdm import tqdm
import itertools
import re
import sys
import argparse
import networkx as nx


def parse_args():
    '''
    Parses the arguments
    
    :return arguments passed in command
    '''
    parser = argparse.ArgumentParser(description='Create hashtag sequence coordination network')
    
    parser.add_argument('--input',
                        dest='input_path',
                        help='Input file path')
    
    parser.add_argument('--output',
                        dest='output_path',
                        help='Output file path')
    
    parser.add_argument('--campaign-name',
                        dest='campaign_name',
                        help='Name of campaign')
    
    parser.add_argument('--filename',
                        dest='tweet_filename',
                        help='Tweet file name')
    
    parser.add_argument('--filter-tweet',
                        dest='filter_tweet',
                        type=int,
                        help='Filter threshold for number of tweets')
    
    parser.add_argument('--filter-hashtag-sequence',
                        dest='filter_hashtag_sequence',
                        type=int,
                        help='Filter threshold for number of hashtag sequence')
                        
    return parser.parse_args()
                        
    
    
def extract_hashtags(df) -> pd.DataFrame:
    '''
    Extracts the hashtags from tweets
    
    :param df: Tweet Dataframe
    
    :return pandas Dataframe
    '''
    df['hashtags_frm_text'] = df.tweet_text.apply(
        lambda x: list(set(re.findall(r'\B\#(\w+)', x))))
    
    
    return df.loc[df['hashtags_frm_text'].map(len) != 0]


def filter_hastag_sequence(df, filter_hashtag_sequence=5) -> pd.DataFrame:
    '''
    Filter the dataframe based on number of hashtag sequence in tweet
    
    :param df: Pandas dataframe of tweets
    :param filter_hashtag_sequence: Threshold to filter tweets based on squence of hashtags
    
    :return Pandas DataFrame
    '''
    return df.loc[df['hashtags_frm_text'].map(len) >= filter_hashtag_sequence]

    
def filter_tweet_count(df,
                       retweet=False, 
                       filter_tweet=5)-> pd.DataFrame:
    '''
    Filter users based on filter_tweet threshold
    
    :param df: Pandas dataframe of tweets
    :param retweet: Flag to filter retweet
    :param filter_tweet: Threshold for tweet count to filter
    
    :return pandas dataframe
    '''

    userids = (df.loc[(df['is_retweet'] == retweet)]
        .groupby(['userid'])['tweetid']
        .size()
        .to_frame('tweet_count')
        .reset_index()
        .query('tweet_count >= {}'.format(filter_tweet))
        )['userid'].tolist()
    
    return df.loc[df['userid'].isin(userids)]



def bin_tweets_in_time(df, time_bin=24, time_part='H') -> dict:
    '''
    Bins the tweets based on time_bin
    
    :param time_bin: time to bin the tweets with
    
    :return dictionary with end time as key and dataframe of tweets within time range \
    as values
    '''
    df['tweet_time'] = pd.to_datetime(df['tweet_time'])
    #Time bins
    time_lists = (pd.DataFrame(columns=['NULL'],
                               index=pd.date_range(df['tweet_time'].min(),
                                                   df['tweet_time'].max() + \
                                                datetime.timedelta(hours=24),
                                                freq =f'{time_bin}{time_part}'))
                  .index
                  .tolist())
                  
    
    #Filter tweets in bins
    df_time = pd.DataFrame()

    for i in range(1, len(time_lists)):
        temp_df = df.loc[(time_lists[i-1] <= df['tweet_time']) & \
                           (df['tweet_time'] < time_lists[i])]

        if len(temp_df) == 0:
            continue

        temp_df = filter_tweet_count(temp_df)
        # print(temp_df.info())
        temp_df = (temp_df
                   .groupby('userid')['hashtags_frm_text']
                    .apply(list)
                    .reset_index(name='hashtag_list')
                  )
        # temp_df = temp_df.loc[temp_df['hashtag_list'].map(len) > 0]
        temp_df['unique_hashtags'] = temp_df['hashtag_list'].apply(
            lambda x: set([j for i in x for j in i]))
        
        temp_df = temp_df.loc[temp_df['unique_hashtags'].map(len) >= 5]
        

        if len(temp_df) <= 1:
            continue

        df_time = pd.concat([df_time, temp_df])
        
    return df_time
        

    

def create_hashtag_sequence_network(df_time):
    '''
    Creates a network of users sharing same sequence of hashtags
    
    :param time_binned_df: tweet data that is binned according to time
    :param output_path: path where the hashtag graph is to be saved
    :param campaign_name: Name of campaigns
    '''
    all_users = []

    

    userid_combination = itertools.combinations(
        df_time.to_dict('records'), 2)

    for user in userid_combination:
        user[1]['hashtag_list'].sort()
        user[0]['hashtag_list'].sort()

        hashtag_1 = list(k for k, _ in itertools.groupby(user[0]['hashtag_list']))
        hashtag_2 = list(k for k, _ in itertools.groupby(user[1]['hashtag_list']))

        min_length, max_length = hashtag_2, hashtag_1

        if len(hashtag_1) < len(hashtag_2):
            min_length, max_length = hashtag_1, hashtag_2

        i = 0

        for hashtag in min_length:
            if hashtag in max_length:
                i = 1 + 1

        if i == 0:
            continue

        all_users.append([user[0]['userid'], user[1]['userid'], i])

    return pd.DataFrame(columns=['source', 'target', 'count'],data = all_users)
    
    
    
def create_graph(df_network,
                 output_path,
                 campaign_name,
                 source_column = 'source',
                 target_column = 'target',
                 weight_column=None, 
                 type_text=''):
    if len(df_network) == 0:
        print(f'\n This data has no users using same {type_text}')
              
        return
    
    G = nx.from_pandas_edgelist(df_network, 
                                source_column, 
                                target_column, 
                                [weight_column])
    Gcc = sorted(nx.connected_components(G), key=len, reverse=True)
    G0 = G.subgraph(Gcc[0])
              
    print(f'\n ------------- {type_text} network information ----------\n')
    print(nx.info(G))
    print('\n Number of connected components :', len(Gcc))          
    print('\n Largest connected component: ')
    print(nx.info(G0))
              
    nx.write_gml(G, f'{output_path}/{campaign_name}_{type_text}_network.gml')
              

def read_data(input_path, filename):
    '''
    Read the data file from the given location
    
    :param input_path: Location to input file
    :param filename: Name of the file
    :return pandas dataframe
    '''
                        
    parts = filename.split('.')
                        
    if parts[-1] == 'csv':
        return pd.read_csv(f'{input_path}/{filename}')
                        
    if parts[-1] == 'gz' and parts[-2] == 'pkl':
        return pd.read_pickle(f'{input_path}/{filename}')
    
    raise Exception('--filename : Invalid file format. \n Only pkl.gz and csv accepted')
        
        
def hashtag_coordination(df, args):
    '''
    Apply different hashtag filtering methods
    
    '''
    df = extract_hashtags(df)
    
    if len(df) == 0:
        print('No hashtag data \n')
        
        return
    
#     df = filter_tweet_count(df, False, args.filter_tweet)
    
#     if len(df) == 0:
#         print('No tweet greater than 5 \n')
        
#         return
        
#     df = filter_hastag_sequence(df, filter_hashtag_sequence=args.filter_hashtag_sequence)
    
#     if len(df) == 0:
#         print('No 5 hashtags \n')
        
#         return
    
    dict_df = bin_tweets_in_time(df)
    
    # print(
    
    if len(dict_df) == 0:
        print('No data in time dictionary \n')
        
        return
    df_network = create_hashtag_sequence_network(dict_df)
    create_graph(df_network,
                 output_path=args.output_path,
                 campaign_name=args.campaign_name,
                 source_column = 'source',
                 target_column = 'target',
                 weight_column='count', 
                 type_text='Hashtag_sequences')

def main():
    try:
        args = parse_args()
        df = read_data(args.input_path, args.tweet_filename)
        hashtag_coordination(df, args)
    except Exception as e:
        print(e)
        return e
    
if __name__ == "__main__":
    main()
                        
                        
# python hashtag-sequence-coordination.py --input=/geode2/home/u070/potem/Carbonate/Projects/infoOps-strategy/data/derived/MX_0621 --output=/geode2/home/u070/potem/Carbonate/Projects/infoOps-strategy/data/derived/ --campaign-name=MX_0621 --filename=MX_0621_all_tweet.pkl.gz --filter-tweet=5 --filter-hashtag-sequence=3
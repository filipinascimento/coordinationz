#Filter tweet: less than 8 tweets, 30 min time window

import pandas as pd
import sys
import argparse
sys.path.insert(0, 
                '/geode2/home/u070/potem/Carbonate/Projects/infoOps-strategy/script/helper')
from helper import *



def parse_args():
    '''
    Parses the arguments
    
    :return arguments passed in command
    '''
    parser = argparse.ArgumentParser(description='Create synchronizations coordination network')
    
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
    
    parser.add_argument('--time-bin',
                        dest='time_bin',
                        type=int,
                        help='Time to bin tweets')
                        
    return parser.parse_args()


def synchronized_action_coordination(df, args):
    '''
    Creates coordination network for synchronized action framework
    
    :param df: Tweet dataframe
    :param args: Arguments passed through command
    '''
    path = f'{args.output_path}/{args.campaign_name}_time_bin.pkl.gz'
    if os.path.isfile(path):
        df_time = pd.read_pickle(path)
    else:
        print('\n----- Start: Filtering tweets ---------')
        df = filter_tweet_count(df, retweet=None, 
                                filter_tweet = args.filter_tweet)
        print('----- End: Filtering tweets ---------\n')

        print(f'----- Start: Binning Tweets by {args.time_bin} min ---------')
        df_time = bin_time_in_time(df, time_bin=30, time_part='T')
        print(f'----- End: Binning Tweets by {args.time_bin} min ---------\n')
        
    if len(df_time) == 0:
        print('No data in time bin \n')
        
        return
    df_time.to_pickle(path)

    path = \
    f'{args.output_path}/{args.campaign_name}_synchronous_user_vector.pkl.gz'
    if os.path.isfile(path):
        df_vector = pd.read_pickle(path)
    else:
        print(f'----- Start: Calculating tfidf vector ---------')
        df = convert_tweetids_to_string(df_time, field='time_index')
        df_vector = calculate_tf_idf_vector(df)
        print(f'----- End: Calculating tfidf vector ---------\n')
        
    if len(df_vector) == 0:
        print('No data in vector \n')
        
        return
    
    df_vector.to_pickle(path)
    
    path = \
    f'{args.output_path}/{args.campaign_name}_synchronous_user_network.pkl.gz'
    if os.path.isfile(path):
        df_network = pd.read_pickle(path)
    else:
        print(f'----- Start: Creating user projection network ---------')
        df_network = create_user_projection(df_time,
                                            column_to_join='time_interval')
        print(f'----- End: Creating user projection network ---------')
        
    if len(df_network) == 0:
        print('No data in user projection \n')
        
        return
    
    df_network.to_pickle(path)

    path = \
f'{args.output_path}/{args.campaign_name}_synchronous_cosine_network.pkl.gz'
    if os.path.isfile(path):
        df_network = pd.read_pickle(path)
    else:
        print(f'----- Start: Calculating cosine similarity ---------')
        df_network = calculate_cosine_similarity(df_network, df_vector)
        print(f'----- End: Calculating cosine similarity ---------')
        
    if len(df_network) == 0:
        print('No data in cosine similarity \n')
        
        return
    
    df_network.to_pickle(path)
        
    print('\n ----- Start: Create synchronous user graph -----')
    create_graph(df_network, args.output_path,
                 campaign_name = args.campaign_name,
                 source_column='source',
                 target_column='target',
                 weight_column='cosine',
                 type_text='Synchronized action ')
    print(f'----- End: Creating synchronous user graph ---------\n')
    
    
def main():
    try:
        print('----- Start: Parsing the arguments ---------')
        args = parse_args()
        print('----- End: Parsing the arguments -----------\n')
        
        print('----- Start: Reading Tweet data ---------')
        df = read_data(args.input_path, args.tweet_filename)
        print('----- End: Reading Tweet data ---------\n')
        
        
        # print(df.info())
        
        print('----- Start: Synchronized Action Coordination ---------')
        synchronized_action_coordination(df, args)
        print('----- End: Synchronized Action Coordination ---------\n')
        
    except Exception as e:
        print(e)
        return e
    
if __name__ == "__main__":
    main()
    
    
    
# python sychronized_action_coordination.py --input=/geode2/home/u070/potem/Carbonate/Projects/infoOps-strategy/data/derived/MX_0621 --output=/geode2/home/u070/potem/Carbonate/Projects/infoOps-strategy/data/derived/MX_0621 --campaign-name=MX_0621 --filename=MX_0621_all_tweet.pkl.gz --filter-tweet=8 --time-bin=30
                    

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
    parser = argparse.ArgumentParser(description='Create mentions coordination network')
    
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
    
#     parser.add_argument('--time-bin',
#                         dest='time_bin',
#                         type=int,
#                         help='Time to bin tweets')
                        
    return parser.parse_args()


def mention_coordination(df, args):
    
    print('\n----- Start: Filtering tweets ---------')
    df = filter_tweet_count(df, retweet=False, 
                            filter_tweet = args.filter_tweet)
    print('----- End: Filtering tweets ---------\n')
    
    print('\n----- Start: Extract mentions in tweets ------')
    df = extract_mentions(df)
    print('\n----- End: Extract mentions in tweets --------')
    
    
#     print(df[['tweetid', 'retweet_tweetid']].head())
    
    print('\n----- Start: Create bi-partite network -----')
    df_network = create_user_projection(df, 'mentions_from_text')
    print('\n----- End: Create bi-partite network -----')
    
    # print(df_network.columns)
    
    print('\n----- Start: Create co-retweet strings  -----')
    df_string = convert_tweetids_to_string(df, field='mentions_from_text')
    print('\n----- End: Create co-retweet strings -----')
    
    print('\n----- Start: Calculate tfidf vector -----')
    df_string = calculate_tf_idf_vector(df_string)
    print('\n------ End: Calculate tfidf vector -----')
    
    print('\n----- Start: Retweet user projection network ------')
    df_network = calculate_cosine_similarity(df_network, df_string)
    print('\n----- End: Retweet user projection network -----')
    
    print('\n ----- Start: Create mention graph -----')
    create_graph(df_network, 
                 args.output_path,
                 campaign_name=args.campaign_name,
                 source_column='source',
                 target_column='target',
                 weight_column='cosine',
                 type_text='mention')
    print(f'----- End: Creating user mention graph ---------\n')


def main():
    try:
        print('----- Start: Parsing the arguments ---------')
        args = parse_args()
        print('----- End: Parsing the arguments -----------\n')
        
        print('----- Start: Reading Tweet data ---------')
        df = read_data(args.input_path, args.tweet_filename)
        print(df.head())
        print('----- End: Reading Tweet data ---------\n')
        
        
        # print(df.info())
        
        print('----- Start: Mention Coordination ---------')
        mention_coordination(df, args)
        print('----- End: Mention Coordination ---------\n')
        
    except Exception as e:
        print(e)
        return e
    
if __name__ == "__main__":
    main()
    
    
    
    
# python mention_coordination.py --input=/geode2/home/u070/potem/Carbonate/Projects/infoOps-strategy/data/derived/MX_0621/2011 --output=/geode2/home/u070/potem/Carbonate/Projects/infoOps-strategy/data/derived/MX_0621/2011 --campaign-name=MX_0621_2011 --filename=2011_tweets.pkl.gz --filter-tweet=0


# python mention_coordination.py --input=/geode2/home/u070/potem/Carbonate/Projects/infoOps-strategy/data/derived/uganda_0621 --output=/geode2/home/u070/potem/Carbonate/Projects/infoOps-strategy/data/derived/mentions --campaign-name=uganda_0621 --filename=uganda_0621_all_tweet.pkl.gz --filter-tweet=0
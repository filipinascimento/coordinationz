import pandas as pd
import numpy as np
import warnings
import datetime
from tqdm import tqdm
import gzip
import argparse
import orjson
import re
import sys

def parse_args():
    '''
    Parses the arguments
    
    :return arguments passed in command
    '''
    parser = argparse.ArgumentParser(description='Co-Retweet coordination network')
    
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
                        
    return parser.parse_args()


def parse_control_data(input_path, file_name, 
                       campaign_name, output_path):
    i = 0
    data = []
    with gzip.open(input_path + '/' + file_name, "r") as f:
        for line in tqdm(f):
            k = orjson.loads(line)

            hashtags = []
            if len(k['entities']['hashtags']) != 0:
                for h in k['entities']['hashtags']:
                    hashtags.append(h['text'])

            mentions = []
            if len(k['entities']['user_mentions']) != 0:
                for h in k['entities']['user_mentions']:
                    mentions.append(h['id'])

            urls = []
            if len(k['entities']['urls']) != 0:
                for h in k['entities']['urls']:
                    urls.append(h['expanded_url'])

            retweeted_id = '0'
            retweeted_user_id = '0'
            if 'retweeted_status' in list(k.keys()):
                retweeted_id = k['retweeted_status']['id']
                retweeted_user_id = k['retweeted_status']['user']['id']

            data.append([k['id'], k['full_text'], k['in_reply_to_status_id'], 
                         k['in_reply_to_user_id'], k['created_at'], 
                         hashtags, mentions, urls, retweeted_id, retweeted_user_id, k['user']['id']])

        
    df = pd.DataFrame(data=data, columns=['tweet_id', 'tweet_text', 'in_reply_to_status_id', 
                                          'in_reply_to_user_id', 'created_at', 'hashtags', 
                                          'mentions', 'urls', 'retweeted_id', 'retweeted_user_id', 'userid'])
    
    output_path = output_path + '/' + f'control_{campaign_name}.pkl.gz'
    
    df.to_pickle(f'{output_path}')

def main():
    try:
        print('----- Start: Parsing the arguments ---------')
        args = parse_args()
        print('----- End: Parsing the arguments -----------\n')
        
        print('----- Start: Parse the control data ---------')
        parse_control_data(args.input_path, args.tweet_filename,
                           args.campaign_name, args.output_path)
        print('----- End: Parse the control data ------- \n')
        
        
    except Exception as e:
        print(e)
        return e
    
if __name__ == "__main__":
    main()
    
    
    
# python control_parse.py --input=/geode2/home/u070/potem/Carbonate/Projects/infoOps-strategy/data/raw/control/MX_0621/2020 --output=/geode2/home/u070/potem/Carbonate/Projects/infoOps-strategy/data/derived/MX_0621/2020 --campaign-name=MX_0621 --filename=control_driver_tweets.jsonl.gz
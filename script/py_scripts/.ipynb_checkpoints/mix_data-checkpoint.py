import pandas as pd
import sys
import argparse
sys.path.insert(0, 
                '/geode2/home/u070/potem/Carbonate/Projects/infoOps-strategy/script/helper')
from helper import *

def parse_args():
    parser = argparse.ArgumentParser(description='Mix control and drivers')
    
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
                        
    return parser.parse_args()ssss
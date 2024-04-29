import pandas as pd
import sys
import argparse
sys.path.insert(0, 
                '/geode2/home/u070/potem/Carbonate/Projects/infoOps-strategy/script/helper')
from score_test import *

def parse_args():
    '''
    Parses the arguments
    
    :return arguments passed in command
    '''
    parser = argparse.ArgumentParser(description='Range of nodes in graph for test')
    
    parser.add_argument('--num-node',
                        dest='num_node',
                        type=int,
                        help='Input number of nodes in graph')
    
    parser.add_argument('--output-filepath',
                        dest='output_path',
                        help='Output file path')
    
    return parser.parse_args()


def start_test(num_node, output_path):
    data = generate_data(num_node, output_path)
    
    # calculate_score(data, args.output_path)

def main():
    args = parse_args()
    
    print(args.num_node, args.output_path)
    
    start_test(args.num_node, args.output_path)
    
    
if __name__ == "__main__":
    main()
    
    
    
# python coordination_score_test.py --num-node=20 --output-filepath=/geode2/home/u070/potem/Carbonate/Projects/infoOps-strategy/data/derived
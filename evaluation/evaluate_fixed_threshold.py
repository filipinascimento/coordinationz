'''
Author: Manita Pote
Description: This script evaluates the F1 score, Precision, Recall, True Positive Rate, False Positive Rate for the pvalue of 0.05, 0.01 and 0.001 in softunion merging strategy. 
Inputs: Indicator xnet files for cohashtag, coretweet, coword and courl.
'''

import pandas as pd
from xnetwork import load
import coordinationz.cohashtag_helper as cohp
import networkx as nx
import argparse
import os
import coordinationz.indicator_utilities as ind_utl
from sklearn.metrics import precision_score, recall_score, f1_score, classification_report

def parse_args():
    '''
    Parses the arguments
    
    :return arguments passed in command
    '''
    parser = argparse.ArgumentParser(
        description='Runs the evaluation of the IO and control data'
    )
    parser.add_argument('--input_path',
                        dest='input_path',
                        help='The path where all the xnet indicator result files are present.'
                       )
    parser.add_argument('--coretweet',
                        dest='coretweet',
                        help='The xnet file has result of coretweet coordination.'
                       )
    
    parser.add_argument('--courl',
                        dest='courl',
                        help='The xnet file has result of courl coordination.'
                       )
    parser.add_argument('--coword',
                        dest='coword',
                        help='The xnet file has result of coword coordination.'
                       )
    parser.add_argument('--cohashtag',
                        dest='cohashtag',
                        help='The xnet file has result of cohashtag coordination.'
                       )
    parser.add_argument('--output_path',
                        dest='output_path',
                        help='The path where all the result will be stored.'
                       )

    return parser.parse_args()


def check_file_exist(mapping=dict(), input_path=str):
    for key in ['coretweet', 'courl', 'coword', 'cohashtag']:
        if os.path.exists(
            input_path + os.sep + mapping[key]
        ):
            print(f" {key} file exists.")
        else:
            print(f" {key} file does not exist.")
            
            return False

    return True

def load_single_file(G_xnet, indicator):
    G = G_xnet.to_networkx()
        
    all_nodes = []
    for node, attrs in G.nodes(data=True):
        if indicator == 'type':
            all_nodes.append([attrs['Label'], 
                      attrs['category'],
                      attrs['_igraph_index']
                     ])
        else:
            all_nodes.append([attrs['Label'], 
                              attrs['category'],
                              attrs['left_degree'],
                              indicator,
                              attrs['_igraph_index']
                             ])
       
    all_edges = []
    for u, v, attrs in G.edges(data=True):
        if indicator == 'type':
            all_edges.append([u, v, attrs['weight'], 
                              attrs['pvalue'], attrs['Type'], 
                             ])
        else:
            all_edges.append([u, v, attrs['weight'], 
                              attrs['pvalue'], indicator
                             ])
    
    df_edge = pd.DataFrame(data = all_edges,
                           columns = ['source', 'target', 'weight',
                                      'pvalue', 'indicator'
                                     ]
                          )

    if indicator == 'type':
        df_node = pd.DataFrame(data = all_nodes,
                       columns = ['userid', 'category', 'index']
                      )
    else:
        df_node = pd.DataFrame(data = all_nodes,
                               columns = ['userid', 'category', 
                                          'degree', 'indicator',
                                          'index'
                                         ]
                              )

    return df_node, df_edge

def load_files(mapping, path):
    result_map = {}
    for indicator, file in mapping.items():
        G_xnet = load(path + os.sep + file)
            
        df_node, df_edge = load_single_file(G_xnet, indicator)

        print(df_edge.info())
        
        result_map[indicator] = {
            'node': df_node,
            'edge': df_edge
        }
        
    return result_map


def sanity_check(result_map=dict()):
    
    indicators = ['cohashtag', 'coretweet', 'courl', 'coword']

    for indicator in indicators:
        df = result_map[indicator]['node']
        print(f'Indicator, {indicator} 0 degree node:', len(df.loc[df['degree'] == 0]))
        df_edge = result_map[indicator]['edge']
        print(f'Indicator, {indicator} max pvalue :', df_edge['pvalue'].max())
        print(f'Indicator, {indicator} min pvalue :', df_edge['weight'].min())
        print('\n')


def number_of_io_control(result_map=dict()):
    indicators = ['cohashtag', 'coretweet', 'courl', 'coword']

    for indicator in indicators:
        df = result_map[indicator]['node']

        print('Category ', df['category'].unique())
        print('Number of nodes: ', df['index'].nunique())
        print(f'Indicator, {indicator} IO :',
              df.loc[df['category'] == 'io']['userid'].nunique()
             )
        print(f'Indicator, {indicator} Control:', 
              df.loc[df['category'] == 'control']['userid'].nunique()
             )
        print('\n')
    
def load_xnet_files(mapping, path):
    result_map = {}
    for indicator, file in mapping.items():
        G_xnet = load(f'{path}/{file}')
        result_map[indicator] = G_xnet

    return result_map

def check_singleton(graph_map):
    for indicator, G_xnet in graph_map.items():
        degrees = G_xnet.degree()
        singletons = [index for index, degree in enumerate(degrees) if degree == 0]
        
        print(f"Indicator, {indicator}, Singleton nodes:", len(singletons))

def add_remaining_nodes(indicators, result_map):
    all_nodes = []
    for indicator in indicators:
        df_nodes = result_map[indicator]['node']

        all_nodes.append(df_nodes[['userid', 'category']])

    df_all = pd.concat(all_nodes)
    
    df_all = df_all.groupby(['userid', 'category']).first().reset_index()

    return df_all
        

def node_min_edge_probability(df_edges, 
                              df_nodes, 
                              df_nodes_all
                             ):
    
    nodes_prob = pd.concat([
        df_edges[['source', 'pvalue']].rename(columns={'source': 'node'}),
        df_edges[['target', 'pvalue']].rename(columns={'target': 'node'})
    ])
    
    print('edge Intermediatory result:',
          nodes_prob['node'].nunique()
         )
    
    node_level_probs = nodes_prob.groupby('node')['pvalue'].min().reset_index()
    
    print('Intermediatory result (node):', 
          node_level_probs['node'].nunique()
         )

    node_level_probs = df_nodes.merge(node_level_probs,
                                      how='left',
                                      left_on='index',
                                      right_on='node'
                                     )
    df_nodes_all = df_nodes_all.merge(node_level_probs[['userid',
                                                        'category',
                                                        'pvalue']],
                                      on=['userid', 'category'],
                                      how='left'
                                     )
    df_nodes_all.loc[df_nodes_all['pvalue'].isnull(), 'pvalue'] = 1
    
    return df_nodes_all


def threshold_and_label(df_prob, indicator, threshold=0.05):
    print(f'\n Running for  {indicator} indicator')
    tpr = df_prob.loc[df_prob['category'] == 'io']['userid'].nunique()
    print('True positive:', 
          tpr
         )
    tnr = df_prob.loc[df_prob['category'] == 'control']['userid'].nunique()
    print('True negative:', 
          tnr
         )
    df_prob['y_true'] = 0
    df_prob.loc[df_prob['category'] == 'io', 'y_true'] = 1
    df_prob['y_pred'] = 0
    df_prob.loc[df_prob['pvalue'] <= threshold, 'y_pred'] = 1
    df_pred_1 = df_prob.loc[df_prob['y_pred'] == 1]

    true_positive = df_pred_1.loc[df_pred_1['category'] == 'io']['userid'].nunique()
    print('True positive  out of predicted positive:', true_positive)
    
    false_positive = df_pred_1.loc[df_pred_1['category'] == 'control']['userid'].nunique()
    print('False positive out of predicted positive:', false_positive)

    false_negative = df_prob.loc[df_prob['category'] == 'io']['userid'].nunique() - df_pred_1.loc[df_pred_1['category'] == 'io']['userid'].nunique()
    print('False negative out of predicted negative:', false_negative)

    true_negative = df_prob.loc[df_prob['category'] == 'control']['userid'].nunique() - df_pred_1.loc[df_pred_1['category'] == 'control']['userid'].nunique()
    print('True negative out of predicted negative :', true_negative)

    precision = precision_score(df_prob['y_true'],
                                df_prob['y_pred'],
                                pos_label=1,
                               )
    print('Precision :', precision)

    recall = recall_score(df_prob['y_true'],
                         df_prob['y_pred'],
                         pos_label=1,
                         )
    print(f"Recall: {recall}")
    
    # # Calculate F1-score
    f1 = f1_score(df_prob['y_true'],
                 df_prob['y_pred'],
                 pos_label=1,
                 )
    print(f"F1-Score: {f1}")

    return [tpr, tnr, true_positive, false_positive,
            false_negative, true_negative, precision,
            recall, f1, threshold, indicator
           ]


def threshold_range_result(df_prob, threshold_range, all_result, indicator):
    for threshold in threshold_range:
        print(f'\n Running for threshold {threshold} \n')
        all_result.append(
            threshold_and_label(df_prob, indicator, threshold=threshold)
        )
        print('\n')

    return all_result
        

def individual_indicator_result(result_map, 
                                all_result, 
                                thresholds_range,
                               ):
    for indicator, df in result_map.items():
        print('Indicator :', indicator)
        print('Userid :', result_map[indicator]['node']['userid'].nunique())
        print('Node :', result_map[indicator]['node']['index'].nunique())
        
        df_prob_test = node_min_edge_probability(
            result_map[indicator]['edge'],
            result_map[indicator]['node'],
            result_map[indicator]['node']
        )
        
        print(df_prob_test['userid'].nunique())
        print(df_prob_test['index'].nunique())
        print(df_prob_test['pvalue'].max())
        
        all_result = threshold_range_result(df_prob_test,
                                            thresholds_range,
                                            all_result,
                                            indicator
                                           )
        print('\n\n')
        
    return all_result

if __name__ == "__main__":
    args = parse_args()
    mapping = {}
    indicators = ['cohashtag', 'coretweet', 'courl', 'coword']

    # Strip leading/trailing spaces from all string arguments
    input_path = args.input_path.strip() if args.input_path else None
    output_path = args.output_path.strip() if args.output_path else None
    
    mapping['coretweet'] = args.coretweet.strip() if args.coretweet else None
    mapping['courl'] = args.courl.strip() if args.courl else None
    mapping['coword'] = args.coword.strip() if args.coword else None
    mapping['cohashtag'] = args.cohashtag.strip() if args.cohashtag else None

    print('\n')
    print(f"Input path: {input_path}")
    print(f"Coretweet: { mapping['coretweet']}")
    print(f"Courl: { mapping['courl']}")
    print(f"Coword: { mapping['coword']}")
    print(f"Cohashtag: { mapping['cohashtag']}")
    print(f"Output path: {output_path}")

    if check_file_exist(mapping, input_path) == False:
        exit()


    print('\n')

    result_map = load_files(mapping, input_path)
    print(result_map.keys())
    
    number_of_io_control(result_map)
    
    print('\n')
    sanity_check(result_map)
    print('\n')

    result_graph = load_xnet_files(mapping, input_path)
    check_singleton(result_graph)

    df_nodes_all = add_remaining_nodes(indicators, result_map)
    print(f'Indicator, merged IO count:',
          df_nodes_all.loc[df_nodes_all['category'] == 'io']['userid'].nunique()
    )
    print(f'Indicator, merged Control count:', 
          df_nodes_all.loc[df_nodes_all['category'] == 'control']['userid'].nunique()
         )

    
    merged_network = ind_utl.mergeNetworks(result_graph,
                                           shouldAggregate = True,
                                           method = "pvalue",
                                           weightAttribute="1-pvalue"
                                          )

    df_node, df_edge = load_single_file(merged_network, 'type')
    
    df_prob = node_min_edge_probability(df_edge, 
                                        df_node, 
                                        df_nodes_all
                                       )
    print(df_prob.info())
    
    print('Soft Union, IO merged network :',
          df_prob.loc[df_prob['category'] == 'io']['userid'].nunique()
         )
    print('Soft Union, Control merged network :',
          df_prob.loc[df_prob['category'] == 'control']['userid'].nunique(),
          '\n\n'
         )
    
    all_result = []
    threshold_range = [0.05, 0.01, 0.001]
    
    all_result = threshold_range_result(
        df_prob, threshold_range, all_result, 'softunion'
    )
    all_result = individual_indicator_result(result_map, 
                                             all_result, 
                                             threshold_range,
                                            )
    # [tpr, tnr, true_positive, false_positive,
    #         false_negative, true_negative, precision,
    #         recall, f1, threshold, indicator
    #        ]

    columns = ['total_io', 'total_control',
               'true_positive', 'false_positive',
               'false_negative', 'true_negative',
               'precision', 'recall', 'f1',
               'threshold', 'indicator'
              ]
    df = pd.DataFrame(data = all_result,
                      columns = columns
                     )
    x = round(result_map['cohashtag']['edge']['weight'].min(),
              2)
    df['minimum_similarity'] = x
    print(df)

    name =  mapping['coretweet'].split('_tweets_')[0]
    print(name)

    df['campaign'] = name

    df.to_pickle(
        output_path + os.sep + name + f'_evaluation_result_{x}.pkl.gz'
    )
    
# python evaluate_fixed_threshold.py \
# --input_path '/N/slate/potem/project/coordinationz/evaluation/data/uae/uae_new' \
# --coretweet 'uae_082019_1_tweets_softunion_null_manita_0.5_coretweet.xnet' \
# --courl 'uae_082019_1_tweets_softunion_null_manita_0.5_courl.xnet' \
# --cohashtag 'uae_082019_1_tweets_softunion_null_manita_0.5_cohashtag.xnet' \
# --coword 'uae_082019_1_tweets_softunion_null_manita_0.5_coword.xnet' \
# --output_path './data/eval_result'

# python evaluate_fixed_threshold.py \
# --input_path '/N/slate/potem/project/coordinationz/evaluation/data/uae/uae_new' \
# --coretweet 'uae_082019_1_tweets_softunion_null_manita_0.7_coretweet.xnet' \
# --courl 'uae_082019_1_tweets_softunion_null_manita_0.7_courl.xnet' \
# --cohashtag 'uae_082019_1_tweets_softunion_null_manita_0.7_cohashtag.xnet' \
# --coword 'uae_082019_1_tweets_softunion_null_manita_0.7_coword.xnet' \
# --output_path './data/eval_result'



# python evaluate_fixed_threshold.py \
# --input_path '/N/slate/potem/project/coordinationz/evaluation/data/uae' \
# --coretweet 'uae_082019_1_tweets_union_null_coretweet.xnet' \
# --courl 'uae_082019_1_tweets_union_null_courl.xnet' \
# --cohashtag 'uae_082019_1_tweets_union_null_cohashtag.xnet' \
# --coword 'uae_082019_1_tweets_union_null_coword.xnet' \
# --output_path './data/eval_result'

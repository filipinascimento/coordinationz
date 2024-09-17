import pandas as pd
import numpy as np
import os

def filter_user(df, min_activity=10):
    '''
    Filters the user based on number of activity
    :param df: Dataframe
    '''
    df_grp = (df
              .groupby(['userid'])['tweetid']
              .nunique()
              .to_frame('count')
              .reset_index()
             )
    
    df_grp = df_grp.loc[df_grp['count'] >= min_activity]
    
    df = df.loc[df['userid'].isin(df_grp['userid'])]
    
    print(f'After filtering user with min activity {min_activity}')
    
    df_1 = df.loc[df['label'] == 1]
    df_0 = df.loc[df['label'] == 0]
    
    print('Class 1: ', len(df_1))
    print('Class 2: ', len(df_0))
    
    return df
    
def get_original_tweet(df):
    '''
    Gets the original tweets only.
    :param df: DataFrame
    
    :return DataFrame
    '''
    is_retweet = df['is_retweet'] == False
    in_reply_to_tweetid = df['in_reply_to_tweetid'].isnull()
    # quoted_tweet_tweetid = df['quoted_tweet_tweetid'].isnull()
    
    return df.loc[is_retweet & ~in_reply_to_tweetid]

def filter_tweets_with_hashtag(df):
    '''
    Gets tweets with hashtag and convert the string hashtag to list
    :param df: DataFrame
    
    :return DataFrame
    '''
    import ast
    
    df_hashtag = df.loc[
        (~df['hashtags'].isnull()) & (df['hashtags'] != '[]')
    ]
    
    df_hashtag['list_hashtag'] = df_hashtag['hashtags'].apply(
        lambda x: ast.literal_eval(x)
    )
    
    df_0 = df_hashtag.loc[df_hashtag['label'] == 0]
    df_1 = df_hashtag.loc[df_hashtag['label'] == 1]
    
    print('After filtering: Total control users :', 
          df_0['userid'].nunique())
    print('After filtering: Total io users :', 
          df_1['userid'].nunique())
    print(df_hashtag['label'].nunique())
    
    return df_hashtag

def get_bipartite(df, columns):
    return list(zip(*[df[col] for col in columns]))

def load_file(io_path, control_path):
    '''
    Load io or control file or both
    :param io_path: Path to IO file
    :param control_path: Path to Control file
    
    :return DataFrame
    '''
    # columns = ['hashtags', 'userid', 'tweetid', 'label']
    
    if io_path == None:
        df = pd.read_pickle(control_path)
        df['hashtags'] = df['hashtags'].astype(str)
        df['label'] = 0
        
        print('Total control users :', df['userid'].nunique())
        
        return df
    elif control_path == None:
        df = pd.read_pickle(io_path)
        df_io = df_io.loc[
            ~df_io['quoted_tweet_tweetid'].isnull()
        ]
        df['label'] = 1
        
        print('Total io users :', df['userid'].nunique())

        return df
        
    if io_path != None and control_path != None:
        df_control = pd.read_pickle(control_path)
        df_control['hashtags'] = df_control['hashtags'].astype(str)
        df_control['label'] = 0
        
        df_io = pd.read_pickle(io_path)
        df_io = df_io.loc[
            ~df_io['quoted_tweet_tweetid'].isnull()
        ]
        df_io['label'] = 1
        
        print('Total control users :', df_control['userid'].nunique())
        print('Total io users :', df_io['userid'].nunique())
        print('Total control data: ', len(df_control))
        print('Total IO data: ', len(df_io))
        
        df = pd.concat([df_control, df_io],
                       ignore_index=True
                      )
        return df
    

def preprocess_data(io_path, control_path, min_activity):
    '''
    Loads, filter original tweets and filter tweets with
    hashtags
    :param io_path: Path to IO file
    :param control_path: Path to Control File
    
    :return DataFrame
    '''
    df = load_file(io_path, control_path)
    print('Loading data, class: ', df['label'].nunique())
    
    df = get_original_tweet(df)
    print('Filtering original tweet, class: ', 
          df['label'].nunique())
    
    df = filter_tweets_with_hashtag(df)
    print('Filtering tweet with hashtag, class: ', 
          df['label'].nunique())
    
    #filter user based on number of tweets with hashtags
    df = filter_user(df, min_activity=min_activity)
    print(f'Filtering the user with min activity {min_activity}, class: ', 
          df['label'].nunique())
    
    return df

def save_bipartite(tuple_list, save_path):
    '''
    Saves the bipartite graph as json
    :param tuple_list: List of tuple
    :param save_path: Path to save the file
    
    :return None
    '''
    import json

    with open(f'{save_path}', 'w') as f:
        json.dump(tuple_list, f)

    
def get_hashtags_bipartite(io_path, 
                           control_path, 
                           save_path, 
                           min_activity=10
                          ):
    '''
    Gets the bi-partite network of user and hashtags
    :param path: Path to the file
    
    :return list of tuple (user, hashtag)
    '''
    df = preprocess_data(io_path, control_path, min_activity)
    
    df_1 = df.loc[df['label'] == 1]
    df_0 = df.loc[df['label'] == 0]
    
    print('Total activity after filtering for IO:', len(df_1))
    print('Total activity after filtering for Control:', len(df_0))
    
    df = df.explode(['list_hashtag'])

    print(df.head())
    
    bipartite_graph = get_bipartite(df,
                                    ['userid','list_hashtag','label']
                                   )
#     print(df.head())
#     print(len(df))
#     print(bipartite_graph[:5])
    
    save_bipartite(bipartite_graph, save_path)
    
    return df



def save_xnet_to_gml(xnet_file, filename):
    '''
    Converts xnet graph to GML file
    :param xnet_file: Xnet graph
    :param filename: Name of file to save as gml
    
    :return networkx graph
    '''
    import networkx as nx
    
    G = xnet_file.to_networkx()
    
    for node, attrs in G.nodes(data=True):
        if '_igraph_index' in attrs:
            del attrs['_igraph_index']

    for u, v, attrs in G.edges(data=True):
        if '_igraph_index' in attrs:
            del attrs['_igraph_index']
            
    nx.write_gml(G, filename)
    
    return G

def save_edge_attributes(gml_graph, filename):
    '''
    Saves the edge attributes of the gml graph to DataFrame
    :param gml_graph: GML graph
    :param filename: Name of file to save
    
    :return None
    '''
    all_data = []
    for u, v, attrs in G.edges(data=True):
        all_data.append([attrs['weight'], attrs['pvalue']])

    (pd.DataFrame(data=all_data,
                  columns=['weight', 'pvalue']
                 )
    ).to_pickle(filename)

import pandas as pd
import numpy as np
import datetime
import warnings
import glob
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.colors as pltc
import sys
import os

import importlib

#### packages
import helper.strategy_helper as st
import helper.visualization as viz_hp
import helper.helper as hp
import helper.file_helper as file_hp
import config.config as config_hp
import helper.pandas_helper as pd_hp
import helper.twitter_helper as twitter_hp
import helper.clean_tweet as clean_hp



#Load files
importlib.reload(config_hp)

config = config_hp.config()
balanced = config['BALANCED']

positive_conv = balanced['balanced_pos_conversation']
df_pos = pd.read_pickle(positive_conv)

negative_conv = balanced['balanced_neg_conversation']
df_neg = pd.read_pickle(negative_conv)

df = df_pos.append(df_neg)


#Embedding function
def get_embedding(df, filename):
    import torch
    from transformers import BertModel, BertTokenizerFast

    tokenizer = BertTokenizerFast.from_pretrained("setu4993/LaBSE")
    model = BertModel.from_pretrained("setu4993/LaBSE")
    model = model.eval()

    all_replies = []
    total = len(df)
    print(f'\n *** Starting the embedding process: {total} *** \n')
    
    for row in df.iterrows():
        text = row[1]['tweet_text']
        data = row[1]
        
        tweet_clean = clean_hp.remove_mentions(text)
        tweet_clean = clean_hp.remove_hashtags(tweet_clean)
        tweet_clean = clean_hp.remove_URL(tweet_clean)

        inputs = tokenizer(tweet_clean, 
                           return_tensors="pt",
                           padding=True)

        with torch.no_grad():
            output = model(**inputs)

        embeddings = output.pooler_output
        
        all_replies.append([data['replier_tweetid'],
                            data['poster_userid'],
                            data['poster_tweetid'],
                            data['replier_userid'],
                            embeddings
                           ])
        
        if len(all_replies) % 1000 == 0:
            total = len(all_replies)
            
            print(f'{total} done!')
            
    (pd.DataFrame(data=all_replies,
                 columns=['replier_tweetid',
                          'poster_userid',
                          'poster_tweetid',
                          'replier_userid',
                          'embeddings'
                         ]
                )
    ).to_pickle(filename)
    
    print('\n *** Ending the embedding process *** \n')

    
#Run the function
importlib.reload(config_hp)

config = config_hp.config()
embedding_path = config['EMBEDDINGS_PATH']
filename = embedding_path['reply_multilanguage_embedding']

get_embedding(df, filename)

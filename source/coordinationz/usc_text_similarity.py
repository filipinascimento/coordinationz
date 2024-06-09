# based on Ashin's approach
# Will include credit to Ashin in the final version

try:
    from sentence_transformers import SentenceTransformer
except:
    # raise new exception instructing user to install sentence_transformers
    message = "Please install the sentence_transformers package"
    message+= " by running the following command:\n\n"
    message+= "pip install sentence-transformers"
    raise ImportError(message)

try:
    import faiss
except:
    # raise new exception instructing user to install faiss
    message = "Please install the faiss package"
    message+= " see:\nhttps://github.com/facebookresearch/faiss/blob/main/INSTALL.md"
    raise ImportError(message)

from pathlib import Path
import pandas as pd
import numpy as np
from nltk.corpus import stopwords
import nltk
import re
from scipy.stats import rankdata
from datetime import datetime
from datetime import timedelta

from tqdm.auto import tqdm
import networkx as nx
import igraph as ig
# import xnetwork as xn



def get_tweet_timestamp(datetimestring):
    # date in this format:
    # '2023-07-10 03:54:39'
    tid = datetime.strptime(datetimestring, '%Y-%m-%d %H:%M:%S').timestamp()

    try:
        # offset = 1288834974657
        # tstamp = (tid >> 22) + offset
        utcdttime = datetime.utcfromtimestamp(tid)
        return utcdttime
    except:
        return None  

#Downloading Stopwords
nltk.download('stopwords')

#Load English Stop Words
stopword = stopwords.words('english')

combined_tweets_df = None


def preprocess_text(df):
    # Cleaning tweets in en language
    # Removing RT Word from Messages
    df['tweet_text']=df['tweet_text'].str.lstrip('RT')
    # Removing selected punctuation marks from Messages
    df['tweet_text']=df['tweet_text'].str.replace( ":",'')
    df['tweet_text']=df['tweet_text'].str.replace( ";",'')
    df['tweet_text']=df['tweet_text'].str.replace( ".",'')
    df['tweet_text']=df['tweet_text'].str.replace( ",",'')
    df['tweet_text']=df['tweet_text'].str.replace( "!",'')
    df['tweet_text']=df['tweet_text'].str.replace( "&",'')
    df['tweet_text']=df['tweet_text'].str.replace( "-",'')
    df['tweet_text']=df['tweet_text'].str.replace( "_",'')
    df['tweet_text']=df['tweet_text'].str.replace( "$",'')
    df['tweet_text']=df['tweet_text'].str.replace( "/",'')
    df['tweet_text']=df['tweet_text'].str.replace( "?",'')
    df['tweet_text']=df['tweet_text'].str.replace( "''",'')
    # Lowercase
    df['tweet_text']=df['tweet_text'].str.lower()
    
    
    
    df = df[df['tweet_type'] != 'retweet']

    return df

def remove_emoji(string):
    emoji_pattern = re.compile("["
                           u"\U0001F600-\U0001F64F"  # emoticons
                           u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                           u"\U0001F680-\U0001F6FF"  # transport & map symbols
                           u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                           u"\U00002702-\U000027B0"
                           u"\U000024C2-\U0001F251"
                           "]+", flags=re.UNICODE)
    return emoji_pattern.sub(r'', string)

#Message Clean Function
def msg_clean(msg):
    #Remove URL
    msg = re.sub(r'https?://\S+|www\.\S+', " ", msg)

    #Remove Mentions
    msg = re.sub(r'@\w+',' ',msg)

    #Remove Digits
    msg = re.sub(r'\d+', ' ', msg)

    #Remove HTML tags
    msg = re.sub('r<.*?>',' ', msg)
    
    #Remove HTML tags
    msg = re.sub('r<.*?>',' ', msg)
    
    #Remove Emoji from text
    msg = remove_emoji(msg)

    # Remove Stop Words 
    msg = msg.split()
    
    msg = " ".join([word for word in msg if word not in stopword])

    return msg




def text_similarity(df,minSimilarity=0.0,init_threshold = 0.7):
    cum = df
    # warnings.warn(str(cum.columns))
    cum = cum[cum['tweet_type'] != 'retweet']
    
    filt = cum[['user_id', 'tweet_id']].groupby(['user_id'],as_index=False).count()
    filt = list(filt.loc[filt['tweet_id'] >= 10]['user_id'])
    cum = cum.loc[cum['user_id'].isin(filt)]

    # Changing colummns
    cum.rename(columns={'text':'tweet_text'},inplace=True)

    # Adding Timestamp
    #cum['tweet_time'] = cum['tweet_id'].apply(lambda x: get_tweet_timestamp(x))
    cum['tweet_time'] = cum['creation_date'].apply(lambda x:get_tweet_timestamp(x))
    # warnings.warn("calculated tweet_time")
    
    # Preprocess tweet texts
    cum_all = preprocess_text(cum)
    cum_all['tweet_text'] = cum['tweet_text'].replace(',','')
    cum_all['clean_tweet'] = cum['tweet_text'].astype(str).apply(lambda x:msg_clean(x))

    # Cleaning text
    cum_all = cum_all[cum_all['clean_tweet'].apply(lambda x: len(x.split(' ')) > 4)]
    
    #print(cum_all.shape)

    date = cum_all['tweet_time'].min().date()
    finalDate = cum_all['tweet_time'].max().date()
    
    allScores = []
    i = 1
    encoder = SentenceTransformer('stsb-xlm-r-multilingual')
    progressBar = tqdm(total=(finalDate-date).days)

    while date <= finalDate:
        progressBar.update(1)
        cum_all1 = cum_all.loc[(cum_all['tweet_time'].dt.date >=date)&(cum_all['tweet_time'].dt.date < date + timedelta(days=1))]
        if cum_all1.shape[0] == 0:
            date = date+timedelta(days=1)
            i += 1
            continue
        actual_user = cum_all.user_id.unique()

        combined_tweets_df = cum_all1.copy()
        combined_tweets_df.reset_index(inplace=True)
        combined_tweets_df = combined_tweets_df.loc[:, ~combined_tweets_df.columns.str.contains('index')]
    
        del cum_all1
    
        combined_tweets_df.reset_index(inplace=True)
        combined_tweets_df = combined_tweets_df.rename(columns = {'index':'my_idx'})
    
        sentences = combined_tweets_df.clean_tweet.tolist()
    
        plot_embeddings = encoder.encode(sentences)    

        try:
            dim = plot_embeddings.shape[1]  # vector dimension
        except:
            date = date+timedelta(days=1)
            continue
    
        db_vectors1 = plot_embeddings.copy().astype(np.float32)
        a = [mmm for mmm in range(plot_embeddings.shape[0])]
        db_ids1 = np.array(a, dtype=np.int64)
    
        faiss.normalize_L2(db_vectors1)
    
        index1 = faiss.IndexFlatIP(dim)
        index1 = faiss.IndexIDMap(index1)  # mapping df index as id
        index1.add_with_ids(db_vectors1, db_ids1)
    
        search_query1 = plot_embeddings.copy().astype(np.float32)
    
        faiss.normalize_L2(search_query1)

        result_plot_thres = []
        result_plot_score = []
        result_plot_metrics = []
    
    
        lims, D, I = index1.range_search(x=search_query1, thresh=init_threshold)
        # print('Retrieved results of index search')
        # warnings.warn('Retrieved results of index search')
    
        # sim_score_df = create_sim_score_df(lims,D,I,search_query1)

    
        source_idx = []
        target_idx = []
        sim_score = []

        for searchIndex in range(len(search_query1)):
            idx = I[lims[searchIndex]:lims[searchIndex+1]]
            sim = D[lims[searchIndex]:lims[searchIndex+1]]
            for j in range(len(idx)):
                source_idx.append(searchIndex)
                target_idx.append(idx[j])
                sim_score.append(sim[j])

        sim_score_df = pd.DataFrame(list(zip(source_idx, target_idx, sim_score)), columns=['source_idx', 'target_idx', 'sim_score'])
        del source_idx
        del target_idx
        del sim_score
        sim_score_df = sim_score_df.query("source_idx != target_idx").copy()
        sim_score_df['combined_idx'] = sim_score_df[['source_idx', 'target_idx']].apply(tuple, axis=1)
        sim_score_df['combined_idx'] = sim_score_df['combined_idx'].apply(sorted)
        sim_score_df['combined_idx'] = sim_score_df['combined_idx'].transform(lambda k: tuple(k))
        sim_score_df = sim_score_df.drop_duplicates(subset=['combined_idx'], keep='first')
        sim_score_df.reset_index(inplace=True)
        sim_score_df = sim_score_df.loc[:, ~sim_score_df.columns.str.contains('index')]
        sim_score_df.drop(['combined_idx'], inplace = True, axis=1)

        df_join = pd.merge(pd.merge(sim_score_df,combined_tweets_df, left_on='source_idx', right_on='my_idx', how='inner'),combined_tweets_df,left_on='target_idx',right_on='my_idx',how='inner')

        result = df_join[['user_id_x','user_id_y','clean_tweet_x','clean_tweet_y','sim_score']]
        result = result.rename(columns = {'user_id_x':'source_user',
                                        'user_id_y':'target_user',
                                        'clean_tweet_x':'source_text',
                                        'clean_tweet_y':'target_text'})
        sim_score_df = result

        # print('Generated Similarity Score DataFrame')
        # warnings.warn('Generated Similarity Score DataFrame')
    
        del combined_tweets_df
        
        # # for threshold in np.arange(0.7,1.01,0.05):
        # for threshold in [0.95]:    
        #     print("Threshold: ", threshold)
    
        #     sim_score_temp_df = sim_score_df[sim_score_df.sim_score >= threshold]
    
        #     text_sim_network = sim_score_temp_df[['source_user','target_user']]
        #     text_sim_network = text_sim_network.drop_duplicates(subset=['source_user','target_user'], keep='first')
    
        #     # outputfile = outputDir + '/threshold_' + str(threshold) + '_'+str(i)+'.csv'
        #     # text_sim_network.to_csv(outputfile)
        allScores.append(sim_score_df)
        date = date+timedelta(days=1)
        i += 1
        # warnings.warn(str(i))
        # print("Day: ",i)

    combined = pd.concat(allScores,ignore_index=True)
    combined['source_user'] = combined['source_user'].apply(lambda x: str(x).strip())
    combined['target_user'] = combined['target_user'].apply(lambda x: str(x).strip())
    # agg="max"

    # if(agg == 'max'):
    #     combined.sort_values(by='weight', ascending=False, inplace=True)
    #     combined.drop_duplicates(subset=['source_user', 'target_user'], inplace=True)
    # else:
    combined  = combined.groupby(['source_user','target_user'],as_index=False)['sim_score'].mean()
    # rename sim_score to weight
    combined.rename(columns={'sim_score':'weight'},inplace=True)
    G = nx.from_pandas_edgelist(combined, source='source_user', target='target_user', edge_attr=['weight'])
    g = ig.Graph.from_networkx(G)
    g.vs["Label"] = g.vs["_nx_name"]
    
    leftCount = len(df["user_id"].unique())
    pairsCount = leftCount*(leftCount-1)//2
    rank = rankdata(g.es["weight"], method="max")
    g.es["quantile"] = (pairsCount-len(rank)+rank)/pairsCount
    # filter edges by minSimilarity
    if(minSimilarity > 0):
        g.es.select(weight_lt=minSimilarity).delete()
        # delete singletons
        g.delete_vertices(g.vs.select(_degree=0))
    del g.vs["_nx_name"]
    return g

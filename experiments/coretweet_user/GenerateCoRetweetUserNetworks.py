# For HPC
import os
os.environ['OPENBLAS_NUM_THREADS']='1'

from pathlib import Path
import pandas as pd
import numpy as np
from tqdm.auto import tqdm
import coordinationz as cz
import xnetwork as xn
import coordinationz.experiment_utilities as czexp
from pathlib import Path


def obtainEvaluationBipartiteEdgescoRetweetUser(df):
    # Filter data which is retweeted
    retweet_df = df[df['is_retweet'] == True]

    retweeted_users = df[df['tweetid'].isin(retweet_df['retweeted_tweetid'].values.tolist())]
    users_hash = dict(zip(retweeted_users['tweetid'].values.tolist(),retweeted_users['user_screen_name'].values.tolist()))
    
    # Filtering the retweet_ids which are not present
    retweet_df = retweet_df[retweet_df['retweeted_tweetid'].isin(retweeted_users['tweetid'].values.tolist())]
    
    retweet_df['retweeted_user_screen_name'] = retweet_df['retweeted_tweetid'].apply(lambda x : users_hash[x])

    return retweet_df[['user_screen_name','retweeted_user_screen_name']].values.tolist()

 
if __name__ == "__main__":
    df_dir = "../../../cuba_082020_tweets_combined.pkl.gz"
    final_dir ="../../../output"

    dfEvaluation = pd.read_pickle(df_dir,compression='gzip')

    # Create a bipartite graph from the courl data
    bipartiteEdges = obtainEvaluationBipartiteEdgescoRetweetUser(dfEvaluation)

    # creates a null model output from the bipartite graph
    nullModelOutput = cz.nullmodel.bipartiteNullModelSimilarity(
        bipartiteEdges,
        scoreType=["zscore","pvalue-quantized"], # pvalue-quantized, pvalue, or zscore, 
        pvaluesQuantized=[0.0001,0.001,0.01,0.05,0.1,0.25,0.5],
        realizations=1000000,
        batchSize=1000,
        idf="smoothlog", # None, "none", "linear", "smoothlinear", "log", "smoothlog"
        workers=-1,
        minSimilarity = 0.4, # will only consider similarities above that
        returnDegreeValues=True, # will return the degrees of the nodes
    )

    # Create a network from the null model output with a pvalue threshold of 0.05
    g = cz.network.createNetworkFromNullModelOutput(
        nullModelOutput,
        # useZscoreWeights = True,
        # usePValueWeights = True,
        pvalueThreshold=0.01, # only keep edges with pvalue < 0.05
    )


    dataName = os.path.basename(df_dir).split('.')[0]
    final_dir = Path(final_dir)

    xn.save(g, final_dir/f"{dataName}_coRetweetUser.xnet")
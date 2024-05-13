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


def obtainEvaluationBipartiteEdgescoURL(df):
    bipartiteEdges = []
    urls_df = df[df['urls'].apply(lambda x: len(x)>1)]
    for index,row in urls_df.iterrows():
        bipartiteEdges += [[row['userid'],url['expanded_url']] for url in row['urls']]
    return bipartiteEdges

if __name__ == "__main__":

    df_dir = "/project/muric_789/ashwin/INCAS/processed_data/cuba_082020_tweets_combined.pkl.gz"
    final_dir ="/project/muric_789/ashwin/INCAS/outputs_cuba_082020"

    dfEvaluation = pd.read_pickle(df_dir,compression='gzip')

    # Create a bipartite graph from the courl data
    bipartiteEdges = obtainEvaluationBipartiteEdgescoURL(dfEvaluation)

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

    xn.save(g, final_dir/f"{dataName}_coURL.xnet")
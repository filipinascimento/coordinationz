from pathlib import Path
import pandas as pd
import numpy as np
from tqdm.auto import tqdm
import coordinationz as cz
import xnetwork as xn
import coordinationz.experiment_utilities as czexp


if __name__ == "__main__": # Needed for parallel processing
    config = cz.config
    tqdm.pandas()
    # config = cz.load_config("<path to config>")

    networksPath = Path(config["paths"]["NETWORKS"]).resolve()
    networksPath.mkdir(parents=True, exist_ok=True)
    

    dataName = "cuba_082020_tweets"

    # Loads data from the evaluation datasets as pandas dataframes
    dfIO = czexp.loadIODataset(dataName, config=config, flavor="io", minActivities=10)

    # Create a bipartite graph from the retweet data
    bipartiteEdges = czexp.obtainIOBipartiteEdgesRetweets(dfIO)

    # creates a null model output from the bipartite graph
    nullModelOutput = cz.nullmodel.bipartiteNullModelSimilarity(
        bipartiteEdges,
        scoreType=["zscore","pvalue-quantized"], # pvalue-quantized, pvalue, or zscore, 
        pvaluesQuantized=[0.001,0.01,0.05,0.1,0.25,0.5],
        realizations=1000,
        idf="smoothlog", # None, "none", "linear", "smoothlinear", "log", "smoothlog"
        batchSize=100,
        workers=-1,
        minSimilarity = 0.5, # will only consider similarities above 0.5
    )

    g = cz.network.createNetworkFromNullModelOutput(
        nullModelOutput,
        pvalueThreshold=0.05, # only keep edges with pvalue < 0.05
    )

    xn.save(g, networksPath/f"{dataName}_coretweet.xnet")


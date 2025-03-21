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
    dfIO = czexp.loadIODataset(dataName, config=config, flavor="both", minActivities=10)

    # Create a bipartite graph from the retweet data
    bipartiteEdges = czexp.obtainIOBipartiteEdgesRetweets(dfIO)

    # creates a null model output from the bipartite graph
    nullModelOutput = cz.nullmodel.bipartiteNullModelSimilarity(
        bipartiteEdges,
        scoreType=["pvalue"], # pvalue, 
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

import matplotlib.pyplot as plt
# plot nullModelOutput["similarities"] vs nullModelOutput["pvalues"]
fig, ax = plt.subplots()
ax.scatter(nullModelOutput["similarities"], nullModelOutput["pvalues"])
ax.set_xlabel("Similarity")
ax.set_ylabel("P-value")

plt.savefig(Path("Data")/"Figures"/f"{dataName}_similarity_vs_pvalue.png")
plt.close()

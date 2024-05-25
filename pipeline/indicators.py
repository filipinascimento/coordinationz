from pathlib import Path
import pandas as pd
import numpy as np
from tqdm.auto import tqdm
import coordinationz as cz
import xnetwork as xn
import coordinationz.experiment_utilities as czexp
import coordinationz.preprocess_utilities as czpre
import coordinationz.indicator_utilities as czind
import sys


dataName = "sampled_20240226"
# dataName = "challenge_filipinos_5DEC"
# dataName = "challenge_problem_two_21NOV"

realizations=10000
idf="smoothlog"
minSimilarity = 0.2
minActivity = 10
pvalueThreshold=0.25

indicators = ["coretweet","cohashtag","courl"]


networkParameters = f"r{realizations}_idf{idf}_minSim{minSimilarity}_active{minActivity}_pval{pvalueThreshold}"

if len(sys.argv) > 1:
    dataName = sys.argv[1]
if len(sys.argv) > 2:
    indicators = sys.argv[2:]

    
if __name__ == "__main__": # Needed for parallel processing
    config = cz.config
    tqdm.pandas()
    # config = cz.load_config("<path to config>")
    
    networksPath = Path(config["paths"]["NETWORKS"]).resolve()
    networksPath.mkdir(parents=True, exist_ok=True)


    # Loads data from the evaluation datasets as pandas dataframes
    df = czpre.loadPreprocessedData(dataName, config=config)

    bipartiteData = {}
    bipartiteData["coretweet"] = czind.obtainBipartiteEdgesRetweets(df, minActivities=minActivity)
    bipartiteData["cohashtag"] = czind.obtainBipartiteEdgesHashtags(df, minActivities=minActivity)
    bipartiteData["courl"] = czind.obtainBipartiteEdgesURLs(df, minActivities=minActivity)


    # creates a null model output from the bipartite graph
 
    for networkName in indicators:
        bipartiteEdges = bipartiteData[networkName]
        # creates a null model output from the bipartite graph
        nullModelOutput = cz.nullmodel.bipartiteNullModelSimilarity(
            bipartiteEdges,
            scoreType=["zscore","pvalue-quantized"], # pvalue-quantized, pvalue, or zscore, 
            pvaluesQuantized=[0.0001,0.0002,0.0005,0.001,0.005,0.002,0.01,0.02,0.05,0.1,0.25,0.5],
            realizations=realizations,
            batchSize=100,
            idf=idf, # None, "none", "linear", "smoothlinear", "log", "smoothlog"
            workers=-1,
            minSimilarity = minSimilarity, # will only consider similarities above that
            returnDegreeSimilarities=False, # will return the similarities of the nodes
            returnDegreeValues=True, # will return the degrees of the nodes
        )

        # Create a network from the null model output with a pvalue threshold of 0.05
        g = cz.network.createNetworkFromNullModelOutput(
            nullModelOutput,
            # useZscoreWeights = True,
            # usePValueWeights = True,
            pvalueThreshold=pvalueThreshold, # only keep edges with pvalue < 0.05
        )

        xn.save(g, networksPath/f"{dataName}_{networkName}_{networkParameters}.xnet")

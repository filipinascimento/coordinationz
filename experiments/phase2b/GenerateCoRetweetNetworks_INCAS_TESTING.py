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


    # dataName = "challenge_filipinos_5DEC"
    dataName = "sampled_20240226"
    # dataName = "challenge_problem_two_21NOV"

    # Loads data from the evaluation datasets as pandas dataframes
    dfINCAS = czexp.loadINCASDataset(dataName, config=config)

    # Create a bipartite graph from the retweet data
    bipartiteEdges = czexp.obtainINCASBipartiteEdgesRetweets(dfINCAS, minActivities=10)

    # creates a null model output from the bipartite graph
    # creates a null model output from the bipartite graph
    nullModelOutput = cz.nullmodel.bipartiteNullModelSimilarity(
        bipartiteEdges,
        scoreType=["zscore","pvalue-quantized"], # pvalue-quantized, pvalue, or zscore, 
        pvaluesQuantized=[0.0001,0.0005,0.001,0.005,0.01,0.05,0.1,0.5],
        realizations=10000,
        batchSize=1000,
        idf="smoothlog", # None, "none", "linear", "smoothlinear", "log", "smoothlog"
        workers=-1,
        minSimilarity = 0.4, # will only consider similarities above that
        returnDegreeSimilarities=True, # will return the similarities of the nodes
        returnDegreeValues=True, # will return the degrees of the nodes
    )

    onlyNullModelOutput = cz.nullmodel.bipartiteNullModelSimilarity(
        bipartiteEdges,
        scoreType="onlynullmodel", # pvalue-quantized, pvalue, or zscore, 
        realizations=10000,
        batchSize=1000,
        idf="smoothlog", # None, "none", "linear", "smoothlinear", "log", "smoothlog"
        workers=-1,
        returnDegreeSimilarities=True, # will return the similarities of the nodes
        returnDegreeValues=True, # will return the degrees of the nodes
    )

    # Create a network from the null model output with a pvalue threshold of 0.05
    g = cz.network.createNetworkFromNullModelOutput(
        nullModelOutput,
        # useZscoreWeights = True,
        # usePValueWeights = True,
        pvalueThreshold=0.01, # only keep edges with pvalue < 0.05
    )
    
    xn.save(g, networksPath/f"{dataName}_coretweet.xnet")

    nullsim = nullModelOutput["nullmodelDegreePairsSimilarities"]
    nullsim2 = onlyNullModelOutput["nullmodelDegreePairsSimilarities"]
    # plot nullModelOutput["similarities"]
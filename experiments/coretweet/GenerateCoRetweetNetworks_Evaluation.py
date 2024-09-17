from pathlib import Path
import pandas as pd
import numpy as np
from tqdm.auto import tqdm
import coordinationz as cz
import xnetwork as xn
import coordinationz.experiment_utilities as czexp
from collections import Counter

if __name__ == "__main__": # Needed for parallel processing
    config = cz.config
    tqdm.pandas()
    # config = cz.load_config("<path to config>")
    
    networksPath = Path(config["paths"]["NETWORKS"]).resolve()
    networksPath.mkdir(parents=True, exist_ok=True)


    # dataName = "challenge_filipinos_5DEC"
    # dataName = "challenge_problem_two_21NOV_activeusers"
    dataName = "challenge_problem_two_21NOV"

    # Loads data from the evaluation datasets as pandas dataframes
    dfEvaluation = czexp.loadEvaluationDataset(dataName, config=config, minActivities=1)

    # Create a bipartite graph from the retweet data
    bipartiteEdges = czexp.obtainEvaluationBipartiteEdgesRetweets(dfEvaluation,minActivities=1)
    bipartiteEdges = czexp.filterRightNodes(bipartiteEdges, minDegree=10,minActivities=10)


    # plot distribution of right degrees
    import matplotlib.pyplot as plt
    rightDegrees = Counter(bipartiteEdges[:,1])
    plt.figure()
    logbins = np.logspace(np.log10(1),np.log10(max(rightDegrees.values())),50)
    values,_ = np.histogram(list(rightDegrees.values()), bins=logbins)
    plt.scatter(logbins[:-1],values)
    plt.xscale("log")
    plt.yscale("log")
    plt.savefig("Figures/right_degrees.png")
    plt.close()
    print("Unique right nodes:",len(rightDegrees))
    print("Total edges:",len(bipartiteEdges))
    x = set([node for node,degree in tqdm(rightDegrees.items()) if degree >= 10])
    print("Right nodes with degree >= 10:",len(x))
    # singleshot right nodes count
    singleshot = set([node for node,degree in rightDegrees.items() if degree == 1])
    print("Right nodes with degree == 1:",len(singleshot))


    # creates a null model output from the bipartite graph
    nullModelOutputJustSimilarities = cz.nullmodel.bipartiteNullModelSimilarity(
        bipartiteEdges,
        scoreType=["pvalue"], # pvalue, 
        realizations=0,
        batchSize=10,
        idf="smoothlog", # None, "none", "linear", "smoothlinear", "log", "smoothlog"
        workers=-1,
        minSimilarity = 0.1, # will only consider similarities above that
        returnDegreeSimilarities=True, # will return the similarities of the nodes
        returnDegreeValues=True, # will return the degrees of the nodes
    )

    # onlyNullModelOutput = cz.nullmodel.bipartiteNullModelSimilarity(
    #     bipartiteEdges,
    #     scoreType="onlynullmodel", pvalue, 
    #     realizations=1000,
    #     batchSize=10,
    #     idf="none", # None, "none", "linear", "smoothlinear", "log", "smoothlog"
    #     workers=1,
    #     returnDegreeSimilarities=True, # will return the similarities of the nodes
    #     returnDegreeValues=True, # will return the degrees of the nodes
    # )

    # # Create a network from the null model output with a pvalue threshold of 0.05
    # g = cz.network.createNetworkFromNullModelOutput(
    #     nullModelOutput,
    #     # usePValueWeights = True,
    #     pvalueThreshold=0.01, # only keep edges with pvalue < 0.05
    # )
    
    # xn.save(g, networksPath/f"{dataName}_coretweet.xnet")

    # nullsim = nullModelOutput["nullmodelDegreePairsSimilarities"]
    # nullsim2 = onlyNullModelOutput["nullmodelDegreePairsSimilarities"]
    # plot nullModelOutput["similarities"]
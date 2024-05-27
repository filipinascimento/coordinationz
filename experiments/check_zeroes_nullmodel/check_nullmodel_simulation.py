from pathlib import Path
from tqdm.auto import tqdm
import coordinationz as cz
import xnetwork as xn
import coordinationz.indicator_utilities as czind
import coordinationz.preprocess_utilities as czpre
import numpy as np

"""
This script checks the null model similarities for a given bipartite graph using
a simulated dataset.
It calculates the similarities between pairs of nodes in the graph and compares
them to the null model similarities. The null model is created by generating random
realizations of the graph and calculating the similarities for each realization.
The script prints the indexed edges, their similarities, and optionally the p-values
and z-scores.
"""

if __name__ == "__main__": # Needed for parallel processing
    config = cz.config
    tqdm.pandas()
    # config = cz.load_config("<path to config>")

    bipartiteEdges =[
        # e.g., user, hashtag or user, coretweet_id,
        # numeric indices can also be used instead of strings
        ("userA", "a"),
        ("userA", "b"),
        ("userA", "c"),
        ("userA", "d"),
        ("userA", "e"),

        ("userB", "c"),
        ("userB", "d"),
        ("userB", "e"),
        ("userB", "f"),

        ("userC", "a"),
        ("userC", "b"),
        ("userC", "c"),
        ("userC", "d"),
        ("userC", "f"),
        ("userC", "f"),
        ("userC", "f"),
        
        ("userD", "c"),
        ("userD", "d"),
        ("userD", "g"),
        ("userD", "h"),

        ("userE", "c"),
        ("userE", "d"),
        ("userE", "g"),
        ("userE", "i"),

        ("userF", "j"),
        ("userF", "k"),
        ("userF", "l"),
        ("userF", "m"),

        ("userG", "j"),
        ("userG", "k"),
        ("userG", "l"),
        ("userG", "o"),
    ]



    randomEdgesSize = 100000
    hashtagsCount = 10000
    usersCount = 1000

    def uniformProbabilities(size):
        return np.ones(size)/size
    
    def linearProbabilities(size):
        probabilities = np.arange(size)
        return probabilities/np.sum(probabilities)
    
    def powelawProbabilities(size,exponent=3):
        probabilities = 1.0/np.arange(1,size+1)**exponent
        return probabilities/np.sum(probabilities)
    
    hashtagProbabilities = linearProbabilities(hashtagsCount)
    userProbabilities = uniformProbabilities(usersCount)

    randomUsers = np.random.choice(usersCount, size=randomEdgesSize, p=userProbabilities)
    randomHashtags = np.random.choice(hashtagsCount, size=randomEdgesSize, p=hashtagProbabilities)
    randomEdges = [(f"simulated{user}", f"hashtag{hashtag}") for user, hashtag in zip(randomUsers, randomHashtags)]    

        
    bipartiteEdges+=randomEdges

    # scoreType = ["zscore","pvalue"]
    scoreType = ["zscore","pvalue-quantized"]
    # pvalue-quantized is faster than pvalue
    # but pvalues are quantized

    # creates a null model output from the bipartite graph
    nullModelOutput = cz.nullmodel.bipartiteNullModelSimilarity(
        bipartiteEdges,
        scoreType=scoreType,
        realizations=1000, # number of realizations of the null model, use 0 for no null model
        batchSize=10, # number of realizations to calculate in each process
        minSimilarity = 0.1, # will first filter out similarities below this value
        pvaluesQuantized=[0.001,0.005,0.01,0.05,0.1,0.25,0.5],
        # for pvalue-quantized, you can define the p-values of interest
        idf="none", # None, "none", "linear", "smoothlinear", "log", "smoothlog"
        workers=-1, # -1 to use all available cores, 0 or 1 to use a single core
        returnDegreeSimilarities=True, # will also return the similarities by degree pair
        returnDegreeValues=True, # will return the degrees of the nodes
    )

    # similarity network edges indexed
    nullModelIndexedEdges = nullModelOutput["indexedEdges"]

    # similarities same order as indexedEdges 
    nullModelSimilarities = nullModelOutput["similarities"]

    nullModelPvalues = None
    # pvalues same order as indexedEdges
    if("pvalues" in nullModelOutput):
        nullModelPvalues = nullModelOutput["pvalues"]

    nullModelZScores = None
    # zscores same order as indexedEdges (if zcores was requested)
    if("zscores" in nullModelOutput):
        nullModelZScores = nullModelOutput["zscores"]

    # labels for the nodes
    labels = nullModelOutput["labels"]

    # Print the indexed edges and their similarities together with pvalues and zscores
    for i, (edge, similarity) in enumerate(zip(nullModelIndexedEdges, nullModelSimilarities)):
        labelledEdge = (labels[edge[0]],labels[edge[1]])
        if(not labelledEdge[0].startswith("user") or not labelledEdge[1].startswith("user")):
            continue
        printStringList = []
        printStringList.append(f"{labelledEdge[0]}, {labelledEdge[1]} -> {similarity:.3}")

        if(nullModelPvalues):
            pvalue = nullModelPvalues[i]
            if("pvalue-quantized" in scoreType):
                pvalueString = f"(p<{pvalue:.2g})"
            else:
                pvalueString = f"(p={pvalue:.2g})"
            printStringList.append(pvalueString)
        
        if(nullModelZScores):
            zscore = nullModelZScores[i]
            zscoreString = f"(z={zscore:.2g})"
            printStringList.append(zscoreString)
        print(' '.join(printStringList))
    print("")
    

    # You can access the similarities
    degrees2Similarity = nullModelOutput["nullmodelDegreePairsSimilarities"]
    degrees = nullModelOutput["degrees"]

    
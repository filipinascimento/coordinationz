from pathlib import Path
from tqdm.auto import tqdm
import coordinationz as cz
import xnetwork as xn
import coordinationz.experiment_utilities as czexp
import numpy as np

if __name__ == "__main__": # Needed for parallel processing
    # config = cz.config
    # config = cz.load_config("<path to config>")

    # networksPath = Path(config["paths"]["NETWORKS"]).resolve()
    # networksPath.mkdir(parents=True, exist_ok=True)

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

        ("userC", "a"),
        ("userC", "b"),
        ("userC", "c"),
        ("userC", "d"),
        ("userC", "f"),

        ("userD", "i"),
        ("userD", "j"),
        ("userD", "k"),
        ("userD", "l"),

        ("userE", "m"),
        ("userE", "n"),
        ("userE", "k"),
        ("userE", "l"),

        ("userF", "j"),
        ("userF", "k"),
        ("userF", "l"),
        ("userF", "o"),

        ("userG", "j"),
        ("userG", "k"),
        ("userG", "l"),
        ("userG", "o"),

        ("userH", "a"),
        ("userH", "b"),
        ("userH", "c"),
        ("userH", "d"),
        ("userH", "e"),
    ]

    scoreType = ["pvalue"]

    # creates a null model output from the bipartite graph
    nullModelOutput = cz.nullmodel.bipartiteNullModelSimilarity(
        bipartiteEdges,
        scoreType=scoreType,
        realizations=1_000_000, # number of realizations of the null model, use 0 for no null model
        batchSize=100, # number of realizations to calculate in each process
        minSimilarity = 0.0, # will first filter out similarities below this value
        idf="none", # None, "none", "linear", "smoothlinear", "log", "smoothlog"
        workers=10, # -1 to use all available cores, 0 or 1 to use a single core
        returnDegreeSimilarities=False, # will also return the similarities by degree pair
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

    # labels for the nodes
    labels = nullModelOutput["labels"]

    # Print the indexed edges and their similarities together with pvalues
    for i, (edge, similarity) in enumerate(zip(nullModelIndexedEdges, nullModelSimilarities)):
        labelledEdge = (labels[edge[0]],labels[edge[1]])
        
        printStringList = []
        printStringList.append(f"{labelledEdge[0]}, {labelledEdge[1]} -> {similarity:.3}")

        if(nullModelPvalues is not None):
            pvalue = nullModelPvalues[i]
            if("pvalue" in scoreType):
                pvalueString = f"(p<{pvalue:.2g})"
            printStringList.append(pvalueString)
        
        print(' '.join(printStringList))
    print("")
    


    # You can access the null model similarities
    if("nullmodelDegreePairsSimilarities" in nullModelOutput):
        degrees2Similarity = nullModelOutput["nullmodelDegreePairsSimilarities"]
    degrees = nullModelOutput["degrees"]

    
    # Create a network from the null model output with a pvalue threshold of 0.05
    g = cz.network.createNetworkFromNullModelOutput(
        nullModelOutput,
        pvalueThreshold=0.25, # only keep edges with pvalue < 0.5
    )

    # Save the network
    xn.save(g, f"sample_network.xnet")


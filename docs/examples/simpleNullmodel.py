from pathlib import Path
from tqdm.auto import tqdm
import coordinationz as cz
import xnetwork as xn
import coordinationz.experiment_utilities as czexp


if __name__ == "__main__": # Needed for parallel processing
    config = cz.config
    # config = cz.load_config("<path to config>")

    networksPath = Path(config["paths"]["NETWORKS"]).resolve()
    networksPath.mkdir(parents=True, exist_ok=True)

    bipartiteEdges =[
        ("A", "a"),
        ("A", "b"),
        ("A", "c"),
        ("A", "d"),

        ("B", "c"),
        ("B", "d"),

        ("C", "b"),
        ("C", "b"),
        ("C", "e"),
        ("C", "f"),
        ("C", "g"),
        ("C", "h"),
        ("C", "h"),

        ("D", "e"),
        ("D", "f"),
        ("D", "g"),
        ("D", "h"),
    ]
    
    # scoreType = ["zscore","pvalue"]
    scoreType = ["zscore","pvalue-quantized"]
    # pvalue-quantized is faster than pvalue
    # but pvalues are quantized

    # creates a null model output from the bipartite graph
    nullModelOutput = cz.nullmodel.bipartiteNullModelSimilarity(
        bipartiteEdges,
        scoreType=scoreType,
        realizations=1000000,
        batchSize=1000, # number of realizations to calculate in each process
        minSimilarity = 0.0, # will first filter out similarities below this value
        # pvaluesQuantized=[0.01,0.05,0.1,0.25,0.5],
        # for pvalue-quantized, you can define the p-values of interest
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
    # print("\nIndexed Edges -> similarities (p:pvalue) (z:zscore)")
    for i, (edge, similarity) in enumerate(zip(nullModelIndexedEdges, nullModelSimilarities)):
        reconstructedEdge = [(labels[edge[0]],labels[edge[1]])]
        
        printStringList = []
        printStringList.append(f"{reconstructedEdge} -> {similarity:.3}")

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

    
    # Create a network from the null model output with a pvalue threshold of 0.05
    g = cz.network.createNetworkFromNullModelOutput(
        nullModelOutput,
        pvalueThreshold=0.5, # only keep edges with pvalue < 0.5
    )

    # Save the network
    xn.save(g, networksPath/f"sample_network.xnet")


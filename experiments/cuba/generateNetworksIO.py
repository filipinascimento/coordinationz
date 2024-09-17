from pathlib import Path
import pandas as pd
import numpy as np
from tqdm.auto import tqdm
import coordinationz as cz
import xnetwork as xn
import coordinationz.experiment_utilities as czexp
import igraph as ig

if __name__ == "__main__": # Needed for parallel processing
    config = cz.config
    tqdm.pandas()
    # config = cz.load_config("<path to config>")

    networksPath = Path(config["paths"]["NETWORKS"]).resolve()
    networksPath.mkdir(parents=True, exist_ok=True)
    

    dataName = "cuba_082020_tweets"

    # Loads data from the evaluation datasets as pandas dataframes
    dfIO = czexp.loadIODataset(dataName, config=config, flavor="all", minActivities=10)

    bipartiteData = {}
    bipartiteData["coretweet"] = czexp.obtainIOBipartiteEdgesRetweets(dfIO, minActivities=10)
    bipartiteData["cohashtag"] = czexp.obtainIOBipartiteEdgesHashtags(dfIO, minActivities=10)
    bipartiteData["courl"] = czexp.obtainIOBipartiteEdgesURLs(dfIO, minActivities=10)

    for networkName,bipartiteEdges in bipartiteData.items():
        # creates a null model output from the bipartite graph
        nullModelOutput = cz.nullmodel.bipartiteNullModelSimilarity(
            bipartiteEdges,
            scoreType=["pvalue"], # pvalue, 
            realizations=10000,
            batchSize=100,
            idf="smoothlog", # None, "none", "linear", "smoothlinear", "log", "smoothlog"
            workers=-1,
            minSimilarity = 0.2, # will only consider similarities above that
            returnDegreeSimilarities=False, # will return the similarities of the nodes
            returnDegreeValues=True, # will return the degrees of the nodes
        )

        # Create a network from the null model output with a pvalue threshold of 0.05
        g = cz.network.createNetworkFromNullModelOutput(
            nullModelOutput,
            # usePValueWeights = True,
            pvalueThreshold=0.25, # only keep edges with pvalue < 0.05
        )
        
        xn.save(g, networksPath/f"{dataName}_{networkName}.xnet")


    networks = {}
    for networkType in bipartiteData.keys():
        networks[networkType] = xn.load(networksPath/f"{dataName}_{networkType}.xnet")
    
    # merge the networks via property Label
    label2Index = {}
    index2Label = {}
    edges = []
    for networkType, network in networks.items():
        labels = network.vs["Label"]
        for fromIndex, toIndex in network.get_edgelist():
            fromLabel = labels[fromIndex]
            toLabel = labels[toIndex]
            if fromLabel not in label2Index:
                label2Index[fromLabel] = len(label2Index)
                index2Label[len(index2Label)] = fromLabel
            if toLabel not in label2Index:
                label2Index[toLabel] = len(label2Index)
                index2Label[len(index2Label)] = toLabel
            edges.append((label2Index[fromLabel], label2Index[toLabel]))
        
    mergedNetwork = ig.Graph(len(label2Index), edges=edges, directed=False)
    mergedNetwork.vs["Label"] = [index2Label[i] for i in range(len(index2Label))]
    xn.save(mergedNetwork,networksPath/f"{dataName}_merged.xnet")

    # get components of size >1 mark the cluster membership and save as pandas Label->Cluster
    components = mergedNetwork.components()
    # only keep components of size > 1
    components = [component for component in components if len(component) > 1]
    label2Cluster = {}
    for clusterIndex, component in enumerate(components):
        for labelIndex in component:
            label2Cluster[index2Label[labelIndex]] = clusterIndex
    label2Cluster = pd.Series(label2Cluster, name="Cluster")
    label2Cluster.index.name = "User"
    label2Cluster.to_csv(networksPath/f"{dataName}_merged_clusters.csv")

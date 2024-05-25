from pathlib import Path
import pandas as pd
import numpy as np
from tqdm.auto import tqdm
import coordinationz as cz
import xnetwork as xn
import coordinationz.experiment_utilities as czexp
import igraph as ig
import sys

dataName = "sampled_20240226"

realizations=10000
idf="smoothlog"
minSimilarity = 0.2
minActivity = 10
pvalueThreshold=0.25

indicators = ["coretweet","cohashtag","courl"]

if len(sys.argv) > 1:
    dataName = sys.argv[1]
if len(sys.argv) > 2:
    indicators = sys.argv[2:]

    

networkParameters = f"r{realizations}_idf{idf}_minSim{minSimilarity}_active{minActivity}_pval{pvalueThreshold}"

if __name__ == "__main__": # Needed for parallel processing
    config = cz.config
    tqdm.pandas()
    # config = cz.load_config("<path to config>")
    
    networksPath = Path(config["paths"]["NETWORKS"]).resolve()
    networksPath.mkdir(parents=True, exist_ok=True)


    networkTypes = indicators

    # load xnet networks which are igraph objects
    networks = {}
    for networkType in networkTypes:
        networks[networkType] = xn.load(networksPath/f"{dataName}_{networkType}_{networkParameters}.xnet")
    
    # merge the networks via property Label
    label2Index = {}
    index2Label = {}
    edges = []
    edgeType = []
    pvalues = []
    zscores = []
    similarities = []
    for networkType, network in networks.items():
        labels = network.vs["Label"]
        edgePValues = network.es["pvalue"]
        edgeZScores = network.es["zscore"]
        edgeSimilarities = network.es["weight"]
        for edgeIndex,(fromIndex, toIndex) in enumerate(network.get_edgelist()):
            fromLabel = labels[fromIndex]
            toLabel = labels[toIndex]
            if fromLabel not in label2Index:
                label2Index[fromLabel] = len(label2Index)
                index2Label[len(index2Label)] = fromLabel
            if toLabel not in label2Index:
                label2Index[toLabel] = len(label2Index)
                index2Label[len(index2Label)] = toLabel
            edges.append((label2Index[fromLabel], label2Index[toLabel]))
            edgeType.append(networkType)
            pvalues.append(edgePValues[edgeIndex])
            zscores.append(edgeZScores[edgeIndex])
            similarities.append(edgeSimilarities[edgeIndex])
            
        
    mergedNetwork = ig.Graph(len(label2Index), edges=edges, directed=False)
    mergedNetwork.vs["Label"] = [index2Label[i] for i in range(len(index2Label))]
    mergedNetwork.es["Type"] = edgeType
    mergedNetwork.es["pvalue"] = pvalues
    mergedNetwork.es["zscore"] = zscores
    mergedNetwork.es["similarity"] = similarities
    
    xn.save(mergedNetwork,networksPath/f"{dataName}_{networkParameters}_merged.xnet")

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
    label2Cluster.to_csv(networksPath/f"{dataName}_{networkParameters}_merged_clusters.csv")

    # also save the edges
    edgesData = []
    labels = mergedNetwork.vs["Label"]

    for fromIndex, toIndex in mergedNetwork.get_edgelist():
        fromLabel = labels[fromIndex]
        toLabel = labels[toIndex]
        edgesData.append((fromLabel, toLabel))
    dfEdges = pd.DataFrame(edgesData, columns=["From","To"])
    
    dfEdges.to_csv(networksPath/f"{dataName}_{networkParameters}_merged_edges.csv",index=False)
# Path: experiments/phase2b/MergeINCASpvalue.py


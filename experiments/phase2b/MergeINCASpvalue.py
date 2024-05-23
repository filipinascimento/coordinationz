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

    dataName = "sampled_20240226"

    networkTypes = ["coretweet","cohashtag","courl"]

    # load xnet networks which are igraph objects
    networks = {}
    for networkType in networkTypes:
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


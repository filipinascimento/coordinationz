from . import config

from pathlib import Path
import pandas as pd
import igraph as ig

# ast evaluates strings that are python expressions
import ast
import numpy as np
from collections import Counter

def obtainBipartiteEdgesRetweets(df):
    # keep only tweet_type == "retweet"
    # if linked_tweet or tweet_type or user_id are not in the dataframe, return an empty list
    if "linked_tweet" not in df or "tweet_type" not in df or "user_id" not in df:
        return []
    df = df[df["tweet_type"] == "retweet"]
    bipartiteEdges = df[["user_id","linked_tweet"]].values
    return bipartiteEdges


def obtainBipartiteEdgesRetweetsUsers(df):
    # keep only tweet_type == "retweet"
    # if linked_tweet or tweet_type or user_id are not in the dataframe, return an empty list
    if "linked_tweet_userid" not in df or "tweet_type" not in df or "user_id" not in df:
        return []
    df = df[df["tweet_type"] == "retweet"]
    bipartiteEdges = df[["user_id","linked_tweet_userid"]].values
    return bipartiteEdges


def obtainBipartiteEdgesURLs(df,removeRetweets=True, removeQuotes=False, removeReplies=False):
    if "urls" not in df or "tweet_type" not in df or "user_id" not in df:
        return []
    if(removeRetweets):
        df = df[df["tweet_type"] != "retweet"]
    if(removeQuotes):
        df = df[df["tweet_type"] != "quote"]
    if(removeReplies):
        df = df[df["tweet_type"] != "reply"]
    # convert url strings that looks like lists to actual lists
    urls = df["urls"]
    users = df["user_id"]
    # keep only non-empty lists
    mask = urls.apply(lambda x: len(x) > 0)
    urls = urls[mask]
    users = users[mask]
    # create edges list users -> urls
    edges = [(user,url) for user,urlList in zip(users,urls) for url in urlList]
    return edges

def obtainBipartiteEdgesHashtags(df,removeRetweets=True,removeQuotes=False,removeReplies=False):
    if "hashtags" not in df or "tweet_type" not in df or "user_id" not in df:
        return []
    
    if(removeRetweets):
        df = df[df["tweet_type"] != "retweet"]
    if(removeQuotes):
        df = df[df["tweet_type"] != "quote"]
    if(removeReplies):
        df = df[df["tweet_type"] != "reply"]

    # convert url strings that looks like lists to actual lists
    users = df["user_id"]
    hashtags = df["hashtags"]
    # keep only non-empty lists
    mask = hashtags.apply(lambda x: len(x) > 0)
    hashtags = hashtags[mask]
    users = users[mask]
    # create edges list users -> hashtags
    edges = [(user,hashtag) for user,hashtag_list in zip(users,hashtags) for hashtag in hashtag_list]
    return edges




def filterNodes(bipartiteEdges, minRightDegree=1, minRightStrength=1, minLeftDegree=1, minLeftStrength=1):
    # goes from right to left
    bipartiteEdges = np.array(bipartiteEdges)
    mask = np.ones(len(bipartiteEdges),dtype=bool)
    if(minRightDegree>1):
        uniqueEdges = set(tuple(edge) for edge in bipartiteEdges)
        uniqueEdges = np.array(list(uniqueEdges))
        rightDegrees = Counter(uniqueEdges[:,1])
        mask &= np.array([rightDegrees[rightNode]>=minRightDegree for _,rightNode in bipartiteEdges])
    if(minRightStrength>1):
        rightStrengths = Counter(bipartiteEdges[:,1])
        mask &= np.array([rightStrengths[rightNode]>=minRightStrength for _,rightNode in bipartiteEdges])
    bipartiteEdges = bipartiteEdges[mask]
    
    # goes from left to right
    mask = np.ones(len(bipartiteEdges),dtype=bool)
    if(minLeftDegree>1):
        uniqueEdges = set(tuple(edge) for edge in bipartiteEdges)
        uniqueEdges = np.array(list(uniqueEdges))
        leftDegrees = Counter(uniqueEdges[:,0])
        mask &= np.array([leftDegrees[leftNode]>=minLeftDegree for leftNode,_ in bipartiteEdges])
    if(minLeftStrength>1):
        leftStrengths = Counter(bipartiteEdges[:,0])
        mask &= np.array([leftStrengths[leftNode]>=minLeftStrength for leftNode,_ in bipartiteEdges])
    bipartiteEdges = bipartiteEdges[mask]

    return bipartiteEdges


def parseParameters(config,indicators):
    indicatorConfig = {}
    if("indicator" in config):
        indicatorConfig = config["indicator"]

    nullModelConfig = {}
    if("nullmodel" in config):
        nullModelConfig = config["nullmodel"]
    
    networkConfig = {}
    if("network" in config):
        networkConfig = config["network"]
    
    mergingConfig = {}
    if "merging" in config:
        mergingConfig = config["merging"]
    
    # name to (key, default value)
    nodeFilterParametersMap = {
        "minItemDegree":("minRightDegree",1),
        "minItemStrength":("minRightStrength",1),
        "minUserDegree":("minLeftDegree",1),
        "minUserStrength":("minLeftStrength",1),
    }

    generalFilterOptions = {}
    for key, (param, default) in nodeFilterParametersMap.items():
        if key in indicatorConfig:
            generalFilterOptions[param] = indicatorConfig[key]
        else:
            generalFilterOptions[param] = default
    
    # create a version for each indicator for when indicator is not in the config
    specificFilterOptions = {}
    for indicator in indicators:
        specificConfig = {}
        if indicator in indicatorConfig:
            for key, (param, default) in nodeFilterParametersMap.items():
                if key in indicatorConfig[indicator]:
                    specificConfig[param] = indicatorConfig[indicator][key]
        specificFilterOptions[indicator] = {**generalFilterOptions, **specificConfig}
    

    networkParametersMap = {
        "similarityThreshold":("similarityThreshold",0.0),
        "zscoreThreshold":("zscoreThreshold",0.0),
        "pvalueThreshold":("pvalueThreshold",1.0),
    }

    generalNetworkOptions = {}
    for key, (param, default) in networkParametersMap.items():
        if key in networkConfig:
            generalNetworkOptions[param] = networkConfig[key]
        else:
            generalNetworkOptions[param] = default

    specificNetworkOptions = {}
    for indicator in indicators:
        specificConfig = {}
        if indicator in networkConfig:
            for key, (param, default) in networkParametersMap.items():
                if key in networkConfig[indicator]:
                    specificConfig[param] = networkConfig[indicator][key]
        specificNetworkOptions[indicator] = {**generalNetworkOptions, **specificConfig}

    nullModelOptions = {
        "scoreType": ("scoreType",["zscore","pvalue-quantized"]),
        "realizations":("realizations",10000),
        "pvaluesQuantized":("pvaluesQuantized",None),
        "idf":("idf","smoothlog"), # None, "none", "linear", "smoothlinear", "log", "smoothlog"
        "minSimilarity":("minSimilarity",0.1),
        "batchSize":("batchSize",10),
        "workers":("workers",10),
    }

    generalNullModelOptions = {}
    for key, (param, default) in nullModelOptions.items():
        if key in nullModelConfig:
            generalNullModelOptions[param] = nullModelConfig[key]
        else:
            generalNullModelOptions[param] = default
        
    specificNullModelOptions = {}
    for indicator in indicators:
        specificConfig = {}
        if indicator in nullModelConfig:
            for key, (param, default) in nullModelOptions.items():
                if key in nullModelConfig[indicator]:
                    specificConfig[param] = nullModelConfig[indicator][key]
        specificNullModelOptions[indicator] = {**generalNullModelOptions, **specificConfig}
    
    mergingOptions = {
        "method": "pvalue" # This is the only option for now
    }

    if "merging" in mergingConfig:
        mergingOptions = {**mergingOptions, **mergingConfig}

    returnValue = {}
    returnValue["filter"] = specificFilterOptions
    returnValue["network"] = specificNetworkOptions
    returnValue["nullmodel"] = specificNullModelOptions
    returnValue["merging"] = mergingOptions
    return returnValue


def timestamp():
    import datetime
    return datetime.datetime.now().strftime("%Y%m%d%H%M%S")




def mergeNetworks(networksDictionary):
    # merge the networks via property Label
    label2Index = {}
    index2Label = {}
    nodeAttributes = {}
    edges = []
    edgeType = []
    edgeAttributes = {}

    for networkType, network in networksDictionary.items():
        # if network is empty continue
        if len(network.vs) == 0:
            continue
        labels = network.vs["Label"]
        # rename weight to similarity
        if("weight" in network.es.attributes()):
            network.es["similarity"] = network.es["weight"]
            del network.es["weight"]
        if not nodeAttributes:
            nodeAttributes = {key:[] for key in network.vs.attributes()}
        if not edgeAttributes:
            edgeAttributes = {key:[] for key in network.es.attributes()}

        for edgeIndex,(fromIndex, toIndex) in enumerate(network.get_edgelist()):
            fromLabel = labels[fromIndex]
            toLabel = labels[toIndex]
            if fromLabel not in label2Index:
                label2Index[fromLabel] = len(label2Index)
                index2Label[len(index2Label)] = fromLabel
                for key in nodeAttributes:
                    nodeAttributes[key].append(network.vs[fromIndex][key])
            if toLabel not in label2Index:
                label2Index[toLabel] = len(label2Index)
                index2Label[len(index2Label)] = toLabel
                for key in nodeAttributes:
                    nodeAttributes[key].append(network.vs[toIndex][key])
            edges.append((label2Index[fromLabel], label2Index[toLabel]))
            edgeType.append(networkType)
            for key in edgeAttributes:
                edgeAttributes[key].append(network.es[edgeIndex][key])
    
    edgeAttributes["Type"] = edgeType
    mergedNetwork = ig.Graph(len(label2Index), edges=edges, directed=False,
        vertex_attrs=nodeAttributes, edge_attrs=edgeAttributes)
    return mergedNetwork

def mergedSuspiciousClusters(mergedNetwork):
    # get components of size >1 mark the cluster membership and save as pandas Label->Cluster
    components = mergedNetwork.components()
    # only keep components of size > 1
    components = [component for component in components if len(component) > 1]
    labels = mergedNetwork.vs["Label"]
    label2Cluster = {}
    for clusterIndex, component in enumerate(components):
        for labelIndex in component:
            label2Cluster[labels[labelIndex]] = clusterIndex
    label2Cluster = pd.Series(label2Cluster, name="Cluster")
    label2Cluster.index.name = "User"
    # label2Cluster.to_csv(networksPath/f"{dataName}_{networkParameters}_merged_clusters.csv")
    return label2Cluster

def mergedSuspiciousEdges(mergedNetwork):
    edgesData = []
    labels = mergedNetwork.vs["Label"]
    for fromIndex, toIndex in mergedNetwork.get_edgelist():
        fromLabel = labels[fromIndex]
        toLabel = labels[toIndex]
        edgesData.append((fromLabel, toLabel))
    dfEdges = pd.DataFrame(edgesData, columns=["From","To"])
    # dfEdges.to_csv(networksPath/f"{dataName}_{networkParameters}_merged_edges.csv",index=False)
    return dfEdges
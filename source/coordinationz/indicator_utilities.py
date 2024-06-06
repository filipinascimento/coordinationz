from . import config

from pathlib import Path
import pandas as pd
import igraph as ig

# ast evaluates strings that are python expressions
import ast
import numpy as np
from collections import Counter

from nltk.corpus import stopwords
import nltk
import re


def filterUsersByMinActivities(df, minUserActivities=1, activityType="any"):
    if minUserActivities > 0:
        if(activityType == "any"):
            userActivityCount = df["user_id"].value_counts()
            usersWithMinActivities = set(userActivityCount[userActivityCount >= minUserActivities].index)
        elif("retweet" in activityType.lower()):
            userActivityCount = df[df["tweet_type"]=="retweet"]["user_id"].value_counts()
            usersWithMinActivities = set(userActivityCount[userActivityCount >= minUserActivities].index)
        elif("hashtag" in activityType.lower()):
            # len(hashtags) should be >0
            userActivityCount = df[df["hashtags"].apply(lambda x: len(x) > 0)]["user_id"].value_counts()
            usersWithMinActivities = set(userActivityCount[userActivityCount >= minUserActivities].index)
        elif("url" in activityType.lower()):
            # len(urls) should be >0
            userActivityCount = df[df["urls"].apply(lambda x: len(x) > 0)]["user_id"].value_counts()
            usersWithMinActivities = set(userActivityCount[userActivityCount >= minUserActivities].index)
        else:
            # activity not retweet
            userActivityCount = df[df["tweet_type"]!="retweet"]["user_id"].value_counts()
            usersWithMinActivities = set(userActivityCount[userActivityCount >= minUserActivities].index)
        df = df[df["user_id"].isin(usersWithMinActivities)]
    return df


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



def obtainBipartiteEdgesWords(df,removeRetweets=True,removeQuotes=False,removeReplies=False, ngramSize = 1):
    if "text" not in df or "tweet_type" not in df or "user_id" not in df:
        return []
    
    if(removeRetweets):
        df = df[df["tweet_type"] != "retweet"]
    if(removeQuotes):
        df = df[df["tweet_type"] != "quote"]
    if(removeReplies):
        df = df[df["tweet_type"] != "reply"]

    # convert url strings that looks like lists to actual lists
    users = df["user_id"]
    tokens = df["text"].apply(lambda x: tokenizeTweet(x,ngram_range=(1,ngramSize)))
    # keep only non-empty lists
    mask = tokens.apply(lambda x: len(x) > 0)
    tokens = tokens[mask]
    users = users[mask]
    # create edges list users -> hashtags
    edges = [(user,token) for user,token_list in zip(users,tokens) for token in token_list]
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
    
    outputConfig = {}
    if "output" in config:
        outputConfig = config["output"]
    
    userFilterParametersMap = {
        "minUserActivities":("minUserActivities",1),
    }
    
    generalUserFilterOptions = {}
    for key, (param, default) in userFilterParametersMap.items():
        if key in indicatorConfig:
            generalUserFilterOptions[param] = indicatorConfig[key]
        else:
            generalUserFilterOptions[param] = default

    specificUserFilterOptions = {}
    for indicator in indicators:
        specificConfig = {}
        if indicator in indicatorConfig:
            for key, (param, default) in userFilterParametersMap.items():
                if key in indicatorConfig[indicator]:
                    specificConfig[param] = indicatorConfig[indicator][key]
        specificUserFilterOptions[indicator] = {**generalUserFilterOptions, **specificConfig}

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
        "method": ("method","pvalue"), # This is the only option for now
        # shouldAggregate = true
        # weightAttribute = "similarity"
        # quantileThreshold = 0.0
        # pvalueThreshold = 1.0
        # zscoreThreshold = 0.0
        # similarityThreshold = 0.0
        "shouldAggregate": ("shouldAggregate",True),
        "weightAttribute": ("weightAttribute","similarity"),
        "quantileThreshold": ("quantileThreshold",0.0),
        "pvalueThreshold": ("pvalueThreshold",1.0),
        "zscoreThreshold": ("zscoreThreshold",0.0),
        "similarityThreshold": ("similarityThreshold",0.0),
    }

    generalMergingOptions = {}
    for key, (param, default) in mergingOptions.items():
        if key in mergingConfig:
            generalMergingOptions[param] = mergingConfig[key]
        else:
            generalMergingOptions[param] = default
    
    # thresholdAttribute = "quantile"
    # thresholds = [0.95,0.99]
    outputOptions = {
        "thresholdAttribute": ("thresholdAttribute","quantile"),
        "thresholds": ("thresholds",[0.95,0.99]),
    }

    generalOutputOptions = {}
    for key, (param, default) in outputOptions.items():
        if key in outputConfig:
            generalOutputOptions[param] = outputConfig[key]
        else:
            generalOutputOptions[param] = default

    returnValue = {}
    returnValue["user"] = specificUserFilterOptions
    returnValue["filter"] = specificFilterOptions
    returnValue["network"] = specificNetworkOptions
    returnValue["nullmodel"] = specificNullModelOptions
    returnValue["merging"] = generalMergingOptions
    returnValue["output"] = generalOutputOptions
    return returnValue


def timestamp():
    import datetime
    return datetime.datetime.now().strftime("%Y%m%d%H%M%S")




def mergeNetworks(networksDictionary,
                  shouldAggregate = True,
                  weightAttribute="similarity",
                  quantileThreshold=0.0,
                  pvalueThreshold=1.0,
                  zscoreThreshold=0.0,
                  similarityThreshold=0.0):
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
        
        if("left_degree" in network.vs.attributes()):
            # rename to networkType_left_degree
            network.vs[f"{networkType}_left_degree"] = network.vs["left_degree"]
            del network.vs["left_degree"]
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
                    if key in network.vs.attributes():
                        nodeAttributes[key].append(network.vs[fromIndex][key])
            if toLabel not in label2Index:
                label2Index[toLabel] = len(label2Index)
                index2Label[len(index2Label)] = toLabel
                for key in nodeAttributes:
                    if key in network.vs.attributes():
                        nodeAttributes[key].append(network.vs[toIndex][key])
            edges.append((label2Index[fromLabel], label2Index[toLabel]))
            edgeType.append(networkType)
            for key in edgeAttributes:
                edgeAttributes[key].append(network.es[edgeIndex][key])
    
    edgeAttributes["Type"] = edgeType
    mergedNetwork = ig.Graph(len(label2Index), edges=edges, directed=False,
        vertex_attrs=nodeAttributes, edge_attrs=edgeAttributes)
    if(shouldAggregate):
        combineEdges = {}
        if("similarity" in mergedNetwork.es.attributes()):
            combineEdges["similarity"] = "mean"
        if("zscore" in mergedNetwork.es.attributes()):
            combineEdges["zscore"] = "mean"
        if("pvalue" in mergedNetwork.es.attributes()):
            # use product of (1-pvalue)
            pvalueTransformed = 1-np.array(mergedNetwork.es["pvalue"])
            mergedNetwork.es["1-pvalue"] = pvalueTransformed
            combineEdges["1-pvalue"] = "product"
        if("quantile" in mergedNetwork.es.attributes()):
            quantileTransformed = 1-np.array(mergedNetwork.es["quantile"])
            mergedNetwork.es["1-quantile"] = quantileTransformed
            combineEdges["1-quantile"] = "product"
        # type concatenate
        # sort and concatenate
        combineEdges["Type"] = lambda x: "-".join(sorted(x))
        mergedNetwork = mergedNetwork.simplify(combine_edges=combineEdges)
        if("1-quantile" in mergedNetwork.es.attributes()):
            mergedNetwork.es["quantile"] = 1-np.array(mergedNetwork.es["1-quantile"])
            del mergedNetwork.es["1-quantile"]
        # print("--------")
        # print(f"Using {weightAttribute} as the weight attribute")
        # print("--------")
        if(weightAttribute in mergedNetwork.es.attributes()):
            mergedNetwork.es["weight"] = np.nan_to_num(mergedNetwork.es[weightAttribute], nan=0.0)
    mask = np.ones(mergedNetwork.ecount(),dtype=bool)
    if(similarityThreshold > 0.0):
        mask &= np.array(mergedNetwork.es["similarity"]) > similarityThreshold
    if(zscoreThreshold > 0.0):
        mask &= np.array(mergedNetwork.es["zscore"]) > zscoreThreshold
    if(pvalueThreshold < 1.0):
        mask &= np.array(mergedNetwork.es["pvalue"]) < pvalueThreshold
    if(quantileThreshold > 0.0):
        mask &= np.array(mergedNetwork.es["quantile"]) > quantileThreshold
    mergedNetwork.delete_edges(np.where(mask == False)[0])
    return mergedNetwork

def mergedSuspiciousClusters(mergedNetwork, ):
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



def generateEdgesINCASOutput(mergedNetwork, allUsers,
                             thresholdAttribute = "quantile",
                             thresholds = [0.95,0.99]):
    outputs = {}
    for threshold in thresholds:
        edgesData = []
        labels = mergedNetwork.vs["Label"]
        quantiles = mergedNetwork.es[thresholdAttribute]
        edgeList = mergedNetwork.get_edgelist()
        # sort edgeList and quantiles by quantiles
        edgeList,quantiles = zip(*sorted(zip(edgeList,quantiles), key=lambda x: x[1], reverse=True))

        for edgeIndex,(fromIndex, toIndex) in enumerate(edgeList):
            fromLabel = labels[fromIndex]
            toLabel = labels[toIndex]
            if(thresholdAttribute=="pvalue"):
                if quantiles[edgeIndex] < threshold:
                    edgesData.append((fromLabel, toLabel))
            else:
                if quantiles[edgeIndex] > threshold:
                    edgesData.append((fromLabel, toLabel))
        uniqueUsers = set([user for edge in edgesData for user in edge])
        coordinated = {
            'confidence':1,
            'description':"coordinated pairs based on unified indicator, sorted by quantile",
            'name':"coordinated users pairs",
            'pairs':edgesData,
            'text':f'edges:{len(edgesData)},users:{len(uniqueUsers)}'
        }
        nonCoordinatedUsers = set(allUsers) - uniqueUsers
        non_coordinated = {
            'confidence':0,
            'description':"non coordinated users based on unified indicator",
            'name':"non coordinated users",
            'text':f'users:{len(nonCoordinatedUsers)}',
            'actors':list(nonCoordinatedUsers)
        }

        users = [coordinated,non_coordinated]
        outputs[f"{threshold}"] = {"segments":users}
    return outputs




def suspiciousTables(df,mergedNetwork,
                thresholdAttribute = "quantile",
                thresholds = [0.95,0.99]):
    outputs = {}
    for threshold in thresholds:
        edgesData = []
        labels = mergedNetwork.vs["Label"]
        if "similarity" in mergedNetwork.es.attributes():
            similarities = mergedNetwork.es["similarity"]
        else:
            similarities = mergedNetwork.es["weight"]
        if "Type" in mergedNetwork.es.attributes():
            edgeTypes = mergedNetwork.es["Type"]
        else:
            edgeTypes = ["NA"] * len(similarities)
        quantiles = mergedNetwork.es[thresholdAttribute]
        edgeList = mergedNetwork.get_edgelist()
        # sort edgeList and quantiles by quantiles
        edgeList,quantiles,similarities,edgeTypes = zip(*sorted(zip(edgeList,quantiles,similarities,edgeTypes), key=lambda x: x[1], reverse=True))
        
        for edgeIndex,(fromIndex, toIndex) in enumerate(edgeList):
            fromLabel = labels[fromIndex]
            toLabel = labels[toIndex]
            if(thresholdAttribute=="pvalue"):
                if quantiles[edgeIndex] < threshold:
                    edgesData.append((fromLabel, toLabel,quantiles[edgeIndex],similarities[edgeIndex],edgeTypes[edgeIndex]))
            else:
                if quantiles[edgeIndex] > threshold:
                    edgesData.append((fromLabel, toLabel,quantiles[edgeIndex],similarities[edgeIndex],edgeTypes[edgeIndex]))
        uniqueUsers = set([user for edge in edgesData for user in edge[:2]])
        dfEdges = pd.DataFrame(edgesData, columns=["From","To","Quantile","Similarity","Type"])
        dfFiltered = df[df["user_id"].isin(uniqueUsers)]
        outputs[f"{threshold}"] = {"edges":dfEdges,"filtered":dfFiltered}
    return outputs


def remove_emoji(string):
    emoji_pattern = re.compile("["
                           u"\U0001F600-\U0001F64F"  # emoticons
                           u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                           u"\U0001F680-\U0001F6FF"  # transport & map symbols
                           u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                           u"\U00002702-\U000027B0"
                           u"\U000024C2-\U0001F251"
                           "]+", flags=re.UNICODE)
    return emoji_pattern.sub(r'', string)

def tokenizeTweet(text, ngram_range=(1,2)):
    # check if nltk stopwords are available, if not download
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('stopwords')
    #Load English Stop Words
    stopword = stopwords.words('english')
    stopword = set(stopword)
    stopword.add("https")
    stopword.add("co")
    # Cleaning tweets in en language
    # Removing RT Word from Messages
    text=text.lstrip('RT')
    text=text.replace( ":",' ')
    text=text.replace( ";",' ')
    text=text.replace( ".",' ')
    text=text.replace( ",",' ')
    text=text.replace( "!",' ')
    text=text.replace( "&",' ')
    text=text.replace( "-",' ')
    text=text.replace( "_",' ')
    text=text.replace( "$",' ')
    text=text.replace( "/",' ')
    text=text.replace( "?",' ')
    text=text.replace( "''",' ')
    text=text.lower()
    #Remove URL
    text = re.sub(r'https?://\S+|www\.\S+', " ", text)
    #Remove Mentions
    text = re.sub(r'@\w+',' ',text)
    #Remove Digits
    text = re.sub(r'\d+', ' ', text)
    #Remove HTML tags
    text = re.sub('r<.*?>',' ', text)
    #Remove HTML tags
    text = re.sub('r<.*?>',' ', text)
    #Remove Emoji from text
    text = remove_emoji(text)
    #Remove hashtags
    text = re.sub(r'#\w+', ' ', text)
    #Remove Punctuation
    text = re.sub(r'[^\w\s]', ' ', text)
    #Remove Extra Spaces
    text = re.sub(r'\s+', ' ', text)
    # remove stopwords
    text = ' '.join([word for word in text.split() if word not in stopword])
    # tokenize
    tokens = text.split()
    # also include n-grams of size defined by ngram_range
    ngrams = []
    for n in range(ngram_range[0],ngram_range[1]+1):
        ngrams.extend([" ".join(tokens[i:i+n]) for i in range(len(tokens)-n+1)])
    return ngrams





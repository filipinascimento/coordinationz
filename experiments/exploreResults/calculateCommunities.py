#!/usr/bin/env python

from pathlib import Path
import pandas as pd
import numpy as np
from tqdm.auto import tqdm
import coordinationz as cz
import xnetwork as xn
import coordinationz.experiment_utilities as czexp
import coordinationz.preprocess_utilities as czpre
import coordinationz.indicator_utilities as czind
import coordinationz.network as cznet
import sys
import argparse
import pickle
import shutil
import json
from collections import Counter
# import partial
from functools import partial
import matplotlib.pyplot as plt

dataName = "TA2_full_eval_NO_GT_nat+synth_2024-06-03"
configPath = None
tqdm.pandas()

if(configPath is not None):
    config = cz.load_config(configPath)
    # print("------")
    print("Loading config from",configPath,"...")
    # print("------")
else:
    config = cz.config
    # print("------")
    print("Loading config from default location...")
    # print("------")


def filterNgramParts(tokens,maxTokens=6):
    # suppose it receives a list of tokens that can be ngrams
    # ngrams are separated by space
    # from the first to the last, if the ngram parts are repeated, remove them
    # will return at most 6 valid tokens
    prohibitedTokens = set()
    filteredTokens = []
    for token in tokens:
        if token in prohibitedTokens:
            continue
        tokenParts = token.split()
        for i in range(0,len(tokenParts)):
            prohibitedTokens.add(tokenParts[i])
        # also add all the possible ngrams
        for i in range(2,len(tokenParts)+1):
            for j in range(0,len(tokenParts)-i+1):
                prohibitedTokens.add(" ".join(tokenParts[j:j+i]))
        filteredTokens.append(token)
        # print(prohibitedTokens)
        if(len(filteredTokens) == maxTokens):
            break
    return filteredTokens

networksPath = Path(config["paths"]["NETWORKS"]).resolve()
tablesOutputPath = Path(config["paths"]["TABLES"]).resolve()
figuresOutputPath = Path(config["paths"]["FIGURES"]).resolve()

df = czpre.loadPreprocessedData(dataName, config=config)
df["contentText"] = df["text"]
if("data_translatedContentText" in df):
    df["contentText"] = df["data_translatedContentText"]
    # for the nans, use the original text
    mask = df["contentText"].isna()
    df.loc[mask,"contentText"] = df["text"][mask]


typeToSuffix = {
    ("merged","all"),
    # ("coretweet","all"),
    # ("cohashtag","all"),
    # ("courl","all"),
    # ("coword","all"),
    # ("merged","alltextsim"),
    # ("textsimilarity","alltextsim"),
    # ("merged","allusc"),
    # ("usctextsimilarity","allusc"),
}



preprocessPath = Path(config["paths"]["PREPROCESSED_DATASETS"])
onlyInSynthPath = preprocessPath/f"{dataName}_onlyInSynth.pkl"
if(onlyInSynthPath.exists()):
    with open(onlyInSynthPath, "rb") as f:
        onlyInSynth = pickle.load(f)
        newUsersInSynth = onlyInSynth["newUsers"]
        usersWithNewTweetsInSynth = onlyInSynth["usersWithNewTweets"]
else:
    newUsersInSynth = set()
    usersWithNewTweetsInSynth = set()


tweetID2Tokens = {}
def getTokens(tweetID,text):
    if(tweetID in tweetID2Tokens):
        return tweetID2Tokens[tweetID]
    tokens = czind.tokenizeTweet(text,ngram_range=(1,3))
    tweetID2Tokens[tweetID] = tokens
    return tokens




for networkType,suffix in typeToSuffix:
    networkPath = networksPath/f"{dataName}_{suffix}_{networkType}.xnet"
    g = xn.load(networkPath)



    if newUsersInSynth:
        g.vs["ExtraField"] = ["Natural"]*g.vcount()
        # newUsersInSynt
        # usersWithNewTweetsInSynth
        for userIndex in range(g.vcount()):
            user = g.vs[userIndex]
            if user["Label"] in newUsersInSynth:
                user["ExtraField"] = "New Synth User"
            elif user["Label"] in usersWithNewTweetsInSynth:
                user["ExtraField"] = "New Synth Tweets"

    quantiles = np.array(g.es["quantile"])
    # if("similarity" in g.es.attributes()):
    #     similarities = np.array(g.es["similarity"])
    if("weight" in g.es.attributes()):
        similarities = np.array(g.es["weight"])

    quantileMarkers = [0.90,0.95,0.99,0.995,0.999,0.9995,0.9999,0.99995,0.99999]
    # two histograms one on top of the other same x axis
    fig, axes = plt.subplots(2, 1, sharex=True,figsize=(4,5))

    # cumulative
    axes[0].hist(similarities, bins=50, cumulative=True, density=True,color="#AAAAAA")
    # density with log y axis
    axes[1].hist(similarities, bins=50,color="#AAAAAA")
    axes[1].set_yscale('log')
    # create vertical bars for quantiles markers
    previousQuantilePosition = -1
    differenceMaxMin = np.max(similarities) - np.min(similarities)
    for marker in sorted(quantileMarkers,reverse=True):
        # use quantile variable first point above or equal quantile
        quantilePosition = similarities[np.argmax(quantiles >= marker)]
        if(np.abs(previousQuantilePosition - quantilePosition)<0.04*differenceMaxMin):
            continue
        previousQuantilePosition = quantilePosition
        # quantilePosition = np.quantile(similarities, marker)
        axes[0].axvline(quantilePosition,color="#CC5555", linestyle="--")
        axes[1].axvline(quantilePosition,color="#CC5555", linestyle="--")
        # also add text
        # number of links
        remainingLinks = np.sum(similarities >= quantilePosition)
        print(f"Number of links above {marker}: {remainingLinks}")
        # number of remaining nodes in the network
        gthreshold = g.copy()
        # if("similarity" in g.es.attributes()):
        #     gthreshold.delete_edges(gthreshold.es.select(similarity_lt=quantilePosition))
        if("weight" in g.es.attributes()):
            gthreshold.delete_edges(gthreshold.es.select(weight_lt=quantilePosition))
        # remove singletons
        gthreshold.delete_vertices(gthreshold.vs.select(_degree=0))
        remainingNodes = len(gthreshold.vs)
        print(f"Number of nodes in the network above {marker}: {remainingNodes}")
        axes[0].text(quantilePosition+0.02*differenceMaxMin, 0.5, f"{marker}: {remainingLinks}e {remainingNodes}u", color="#220000", rotation=90,va="center")
        # get axes1 mid y log scale
        ytopLim = axes[1].get_ylim()[1]
        ybottomLim = axes[1].get_ylim()[0]
        midy = np.sqrt(ytopLim*ybottomLim)
        axes[1].text(quantilePosition+0.01*differenceMaxMin, midy, f"{marker}: {remainingLinks}e {remainingNodes}u", color="#220000", rotation=90,va="center")

    # axes[0].set_xlabel("Weight")
    axes[1].set_xlabel("Weight")
    axes[0].set_ylabel("Density (sim. > 0.2)")
    axes[1].set_ylabel("Counts")
    axes[0].set_title(f"Sim. ({suffix},{networkType})")
    # remove right and top frame
    axes[0].spines['right'].set_visible(False)
    axes[0].spines['top'].set_visible(False)
    axes[1].spines['right'].set_visible(False)
    axes[1].spines['top'].set_visible(False)
    plt.tight_layout()
    plt.savefig(figuresOutputPath/f"sim_distribution_{dataName}_{suffix}_{networkType}.png")
    plt.savefig(figuresOutputPath/f"sim_distribution_{dataName}_{suffix}_{networkType}.pdf")
    plt.close()


    # for threshold in [0.990, 0.999, 0.9995, 0.9999, 0.99995, 0.99999]:
    for threshold in [0.9999, 0.99999]:
        thresholdAttribute = "quantile"
        

        gThresholded = g.copy()
        mask = np.ones(gThresholded.ecount(),dtype=bool)
        if(thresholdAttribute=="pvalue"):
            attributeArray = np.array(gThresholded.es["pvalue"])
            mask &= attributeArray < threshold
        else:
            attributeArray = np.array(gThresholded.es[thresholdAttribute])
            mask &= attributeArray > threshold
        gThresholded.delete_edges(np.where(mask == False)[0])
        # remove degree 0 nodes
        gThresholded.delete_vertices(gThresholded.vs.select(_degree=0))
        gThresholded.vs["CommunityIndex"] = gThresholded.community_leiden(objective_function = "modularity",
                                                                        weights = "weight",
                                                                        ).membership
        gThresholded.vs["CommunityLabel"] = [f"{i}" for i in gThresholded.vs["CommunityIndex"]]
        thresholdedNetworkPath = networksPath/f"{dataName}_{suffix}_{networkType}_thres_{thresholdAttribute}_{threshold}.xnet"

        # all users in gThresholded (Label)
        allUsers = set(gThresholded.vs["Label"])

        # onlyEntries in set
        dfFiltered = df[df["user_id"].isin(allUsers)]
        dfRetweets = dfFiltered[dfFiltered["tweet_type"]=="retweet"]
        dfOriginal = dfFiltered[dfFiltered["tweet_type"]!="retweet"]
        dfInNetworkURLs = dfOriginal.dropna(subset=["urls"])
        dfInNetworkHashtags = dfOriginal.dropna(subset=["hashtags"])
        dfInNetworkRetweets = dfRetweets.dropna(subset=["linked_tweet"])
        # for tokens use czind.tokenizeTweet(string)

        dfInNetworkTokens = dfOriginal.dropna(subset=["contentText"])
        # use translated 
        # apply getTokens to text, tweet_id
        dfInNetworkTokens["tokens"] = dfInNetworkTokens[["tweet_id","contentText"]].progress_apply(lambda x: getTokens(*x),axis=1)
        dfInNetworkRetweetTokens = dfInNetworkRetweets.dropna(subset=["contentText"])
        if(dfInNetworkRetweetTokens.empty):
            dfInNetworkRetweetTokens["tokens"] = []
        else:
            dfInNetworkRetweetTokens["tokens"] = dfInNetworkRetweetTokens[["tweet_id","contentText"]].progress_apply(lambda x: getTokens(*x),axis=1)

        user2urlCounter = {}
        user2hashtagsCounter = {}
        user2retweetsCounter = {}
        user2tokensCounter = {}
        user2RetweetTokensCounter = {}

        hashtag2TotalCounts = Counter()
        url2TotalCounts = Counter()
        retweet2TotalCounts = Counter()
        token2TotalCounts = Counter()
        retweetToken2TotalCounts = Counter()

        hashtagSumCount = 0
        urlSumCount = 0
        retweetSumCount = 0
        tokenSumCount = 0
        retweetTokenSumCount = 0

        penaltyFactor = 4
        transform = lambda x: x
        # transform = lambda x: np.log(x+1)

        for user,urls in tqdm(dfInNetworkURLs[["user_id","urls"]].values):
            urlsCounter = Counter(urls)
            # normalize by size of urls
            # urlsCounter/=len(urls)
            for url in urlsCounter:
                urlsCounter[url] /= len(urls)
            if user in user2urlCounter:
                user2urlCounter[user].update(urlsCounter)
            else:
                user2urlCounter[user] = Counter(urlsCounter)
            url2TotalCounts.update(urlsCounter)
            urlSumCount += sum(urlsCounter.values())

        for user,hashtags in tqdm(dfInNetworkHashtags[["user_id","hashtags"]].values):
            hashtagsCounter = Counter(hashtags)
            # normalize by size of hashtags
            # hashtagsCounter/=len(hashtags)
            for hashtag in hashtagsCounter:
                hashtagsCounter[hashtag] /= len(hashtags)
            if user in user2hashtagsCounter:
                user2hashtagsCounter[user].update(hashtagsCounter)
            else:
                user2hashtagsCounter[user] = Counter(hashtagsCounter)
            hashtag2TotalCounts.update(hashtagsCounter)
            hashtagSumCount += sum(hashtagsCounter.values())

        for user,linked_tweet in tqdm(dfInNetworkRetweets[["user_id","linked_tweet"]].values):
            retweetsCounter = Counter(linked_tweet)
            # normalize by size of retweets
            # retweetsCounter/=len(linked_tweet)
            for retweet in retweetsCounter:
                retweetsCounter[retweet] /= len(linked_tweet)
            if user in user2retweetsCounter:
                user2retweetsCounter[user].update(retweetsCounter)
            else:
                user2retweetsCounter[user] = Counter(retweetsCounter)
            retweet2TotalCounts.update(retweetsCounter)
            retweetSumCount += sum(retweetsCounter.values())

        for user,tokens in tqdm(dfInNetworkTokens[["user_id","tokens"]].values):
            tokensCounter = Counter(tokens)
            # normalize by size of tokens
            # tokensCounter/=len(tokens)
            for token in tokensCounter:
                tokensCounter[token] /= len(tokens)
            if user in user2tokensCounter:
                user2tokensCounter[user].update(tokensCounter)
            else:
                user2tokensCounter[user] = Counter(tokensCounter)
            token2TotalCounts.update(tokensCounter)
            tokenSumCount += sum(tokensCounter.values())

        for user,tokens in tqdm(dfInNetworkRetweetTokens[["user_id","tokens"]].values):
            tokensCounter = Counter(tokens)
            # normalize by size of tokens
            # tokensCounter/=len(tokens)
            for token in tokensCounter:
                tokensCounter[token] /= len(tokens)
            if user in user2RetweetTokensCounter:
                user2RetweetTokensCounter[user].update(tokensCounter)
            else:
                user2RetweetTokensCounter[user] = Counter(tokensCounter)
            retweetToken2TotalCounts.update(tokensCounter)
            retweetTokenSumCount += sum(tokensCounter.values())

        community2urlCounter = {}
        community2hashtagsCounter = {}
        community2retweetsCounter = {}
        community2tokensCounter = {}
        community2RetweetTokensCounter = {}

        community2urlSumTotal = {}
        community2hashtagsSumTotal = {}
        community2retweetsSumTotal = {}
        community2tokensSumTotal = {}
        community2RetweetTokensSumTotal = {}

        for community in tqdm(set(gThresholded.vs["CommunityIndex"])):
            community2urlCounter[community] = Counter()
            community2hashtagsCounter[community] = Counter()
            community2retweetsCounter[community] = Counter()
            community2tokensCounter[community] = Counter()
            community2RetweetTokensCounter[community] = Counter()

            community2urlSumTotal[community] = 0
            community2hashtagsSumTotal[community] = 0
            community2retweetsSumTotal[community] = 0
            community2tokensSumTotal[community] = 0
            community2RetweetTokensSumTotal[community] = 0

        labels = gThresholded.vs["Label"]
        communities = gThresholded.vs["CommunityIndex"]
        for userIndex in tqdm(range(len(gThresholded.vs))):
            user = labels[userIndex]
            community = communities[userIndex]
            if(user in user2urlCounter):
                community2urlCounter[community].update(user2urlCounter[user])
                community2urlSumTotal[community] += sum(user2urlCounter[user].values())
            if(user in user2hashtagsCounter):
                community2hashtagsCounter[community].update(user2hashtagsCounter[user])
                community2hashtagsSumTotal[community] += sum(user2hashtagsCounter[user].values())
            if(user in user2retweetsCounter):
                community2retweetsCounter[community].update(user2retweetsCounter[user])
                community2retweetsSumTotal[community] += sum(user2retweetsCounter[user].values())
            if(user in user2tokensCounter):
                community2tokensCounter[community].update(user2tokensCounter[user])
                community2tokensSumTotal[community] += sum(user2tokensCounter[user].values())
            if(user in user2RetweetTokensCounter):
                community2RetweetTokensCounter[community].update(user2RetweetTokensCounter[user])
                community2RetweetTokensSumTotal[community] += sum(user2RetweetTokensCounter[user].values())


        community2urlRelativeDifferenceCounter = {}
        community2hashtagsRelativeDifferenceCounter = {}
        community2retweetsRelativeDifferenceCounter = {}
        community2tokensRelativeDifferenceCounter = {}
        community2RetweetTokensRelativeDifferenceCounter = {}

        # incommunity/incommunitytotal - (all-incommunity)/(all-incommunitytotal)
        for community in tqdm(set(gThresholded.vs["CommunityIndex"])):
            community2urlRelativeDifferenceCounter[community] = Counter()
            community2hashtagsRelativeDifferenceCounter[community] = Counter()
            community2retweetsRelativeDifferenceCounter[community] = Counter()
            community2tokensRelativeDifferenceCounter[community] = Counter()
            community2RetweetTokensRelativeDifferenceCounter[community] = Counter()
            for url in community2urlCounter[community]:
                incommunityRelativeFrequency = community2urlCounter[community][url]/community2urlSumTotal[community]
                if (urlSumCount - community2urlSumTotal[community]) == 0:
                    outcommunityRelativeFrequency = url2TotalCounts[url]
                else:
                    outcommunityRelativeFrequency = (url2TotalCounts[url] - community2urlCounter[community][url])/(urlSumCount - community2urlSumTotal[community])
                community2urlRelativeDifferenceCounter[community][url] = transform(incommunityRelativeFrequency) - penaltyFactor*transform(outcommunityRelativeFrequency)
            for hashtag in community2hashtagsCounter[community]:
                incommunityRelativeFrequency = community2hashtagsCounter[community][hashtag]/community2hashtagsSumTotal[community]
                if (hashtagSumCount - community2hashtagsSumTotal[community]) == 0:
                    outcommunityRelativeFrequency = hashtag2TotalCounts[hashtag]
                else:
                    outcommunityRelativeFrequency = (hashtag2TotalCounts[hashtag] - community2hashtagsCounter[community][hashtag])/(hashtagSumCount - community2hashtagsSumTotal[community])
                community2hashtagsRelativeDifferenceCounter[community][hashtag] = transform(incommunityRelativeFrequency) - penaltyFactor*transform(outcommunityRelativeFrequency)
            for retweet in community2retweetsCounter[community]:
                incommunityRelativeFrequency = community2retweetsCounter[community][retweet]/community2retweetsSumTotal[community]
                if (retweetSumCount - community2retweetsSumTotal[community]) == 0:
                    outcommunityRelativeFrequency = retweet2TotalCounts[retweet]
                else:
                    outcommunityRelativeFrequency = (retweet2TotalCounts[retweet] - community2retweetsCounter[community][retweet])/(retweetSumCount - community2retweetsSumTotal[community])
                community2retweetsRelativeDifferenceCounter[community][retweet] = transform(incommunityRelativeFrequency) - penaltyFactor*transform(outcommunityRelativeFrequency)
            for token in community2tokensCounter[community]:
                incommunityRelativeFrequency = community2tokensCounter[community][token]/community2tokensSumTotal[community]
                if (tokenSumCount - community2tokensSumTotal[community]) == 0:
                    outcommunityRelativeFrequency = token2TotalCounts[token]
                else:
                    outcommunityRelativeFrequency = (token2TotalCounts[token] - community2tokensCounter[community][token])/(tokenSumCount - community2tokensSumTotal[community])
                # if(community==1 and token=="china"):
                #     print("incommunityCount:", community2tokensCounter[community][token])
                #     print("incommunityTotal:", community2tokensSumTotal[community])
                #     print("outcommunityCount:", token2TotalCounts[token] - community2tokensCounter[community][token])
                #     print("outcommunityTotal:", tokenSumCount - community2tokensSumTotal[community])
                #     print("incommunityRelativeFrequency:", incommunityRelativeFrequency)
                #     print("outcommunityRelativeFrequency:", outcommunityRelativeFrequency)
                #     print("relativeDifference:", transform(incommunityRelativeFrequency) - penaltyFactor*transform(outcommunityRelativeFrequency))
                community2tokensRelativeDifferenceCounter[community][token] = transform(incommunityRelativeFrequency) - penaltyFactor*transform(outcommunityRelativeFrequency)
            for token in community2RetweetTokensCounter[community]:
                incommunityRelativeFrequency = community2RetweetTokensCounter[community][token]/community2RetweetTokensSumTotal[community]
                if (retweetTokenSumCount - community2RetweetTokensSumTotal[community]) == 0:
                    outcommunityRelativeFrequency = retweetToken2TotalCounts[token]
                else:
                    outcommunityRelativeFrequency = (retweetToken2TotalCounts[token] - community2RetweetTokensCounter[community][token])/(retweetTokenSumCount - community2RetweetTokensSumTotal[community])

                community2RetweetTokensRelativeDifferenceCounter[community][token] = transform(incommunityRelativeFrequency) - penaltyFactor*transform(outcommunityRelativeFrequency)

        # 0.05517002081887578-0.21999104936469946
        mostCommonLimit = 8
        community2topURLs = {}
        community2topHashtags = {}
        community2topRetweets = {}
        community2topTokens = {}
        community2topRetweetTokens = {}

        community2topSurprisingURLs = {}
        community2topSurprisingHashtags = {}
        community2topSurprisingRetweets = {}
        community2topSurprisingTokens = {}
        community2topSurprisingRetweetTokens = {}

        for community in tqdm(set(gThresholded.vs["CommunityIndex"])):
            community2topURLs[community] = community2urlCounter[community].most_common(mostCommonLimit)
            community2topHashtags[community] = community2hashtagsCounter[community].most_common(mostCommonLimit)
            community2topRetweets[community] = community2retweetsCounter[community].most_common(mostCommonLimit)
            community2topTokens[community] = community2tokensCounter[community].most_common(mostCommonLimit*3)
            community2topRetweetTokens[community] = community2RetweetTokensCounter[community].most_common(mostCommonLimit*3)

            community2topSurprisingURLs[community] = community2urlRelativeDifferenceCounter[community].most_common(mostCommonLimit)
            community2topSurprisingHashtags[community] = community2hashtagsRelativeDifferenceCounter[community].most_common(mostCommonLimit)
            community2topSurprisingRetweets[community] = community2retweetsRelativeDifferenceCounter[community].most_common(mostCommonLimit)
            community2topSurprisingTokens[community] = community2tokensRelativeDifferenceCounter[community].most_common(mostCommonLimit*3)
            community2topSurprisingRetweetTokens[community] = community2RetweetTokensRelativeDifferenceCounter[community].most_common(mostCommonLimit*3)

        community2topSummaryURLs = {}
        community2topSummaryHashtags = {}
        community2topSummaryRetweets = {}
        community2topSummaryTokens = {}
        community2topSummaryRetweetTokens = {}

        community2SurprisingSummaryURLs = {}
        community2SurprisingSummaryHashtags = {}
        community2SurprisingSummaryRetweets = {}
        community2SurprisingSummaryTokens = {}
        community2SurprisingSummaryRetweetTokens = {}

        for community in tqdm(set(gThresholded.vs["CommunityIndex"])):
            # each summary should be a string of semi comma separated values
            community2topSummaryURLs[community] = "; ".join([f"{entry}" for entry,_ in community2topURLs[community]])
            community2topSummaryHashtags[community] = "; ".join([f"{entry}" for entry,_ in community2topHashtags[community]])
            community2topSummaryRetweets[community] = "; ".join([f"{entry}" for entry,_ in community2topRetweets[community]])
            community2topSummaryTokens[community] = "; ".join(filterNgramParts([f"{entry}" for entry,_ in community2topTokens[community]],maxTokens=mostCommonLimit))
            community2topSummaryRetweetTokens[community] = "; ".join(filterNgramParts([f"{entry}" for entry,_ in community2topRetweetTokens[community]],maxTokens=mostCommonLimit))

            community2SurprisingSummaryURLs[community] = "; ".join([f"{entry}" for entry,_ in community2topSurprisingURLs[community]])
            community2SurprisingSummaryHashtags[community] = "; ".join([f"{entry}" for entry,_ in community2topSurprisingHashtags[community]])
            community2SurprisingSummaryRetweets[community] = "; ".join([f"{entry}" for entry,_ in community2topSurprisingRetweets[community]])
            community2SurprisingSummaryTokens[community] = "; ".join(filterNgramParts([f"{entry}" for entry,_ in community2topSurprisingTokens[community]],maxTokens=mostCommonLimit))
            community2SurprisingSummaryRetweetTokens[community] = "; ".join(filterNgramParts([f"{entry}" for entry,_ in community2topSurprisingRetweetTokens[community]],maxTokens=mostCommonLimit))

        # now set the attributes of nodes based on the community
        # "Top URLs","Top Hashtags","Top Retweets","Top Tokens","Top Retweet Tokens"
        # "Surprising URLs","Surprising Hashtags","Surprising Retweets","Surprising Tokens","Surprising Retweet Tokens"
        for nodeIndex in tqdm(range(len(gThresholded.vs))):
            node = gThresholded.vs[nodeIndex]
            community = node["CommunityIndex"]
            node["Top URLs"] = community2topSummaryURLs[community]
            if(not node["Top URLs"]):
                node["Top URLs"] = "None"
            node["Top Hashtags"] = community2topSummaryHashtags[community]
            if(not node["Top Hashtags"]):
                node["Top Hashtags"] = "None"
            # node["Top Retweets"] = community2topSummaryRetweets[community]
            # if(not node["Top Retweets"]):
            #     node["Top Retweets"] = "None
            node["Top Tokens"] = community2topSummaryTokens[community]
            if(not node["Top Tokens"]):
                node["Top Tokens"] = "None"

            node["Top Retweet Tokens"] = community2topSummaryRetweetTokens[community]
            if(not node["Top Retweet Tokens"]):
                node["Top Retweet Tokens"] = "None"
            node["Surprising URLs"] = community2SurprisingSummaryURLs[community]
            if(not node["Surprising URLs"]):
                node["Surprising URLs"] = "None"
            node["Surprising Hashtags"] = community2SurprisingSummaryHashtags[community]
            if(not node["Surprising Hashtags"]):
                node["Surprising Hashtags"] = "None"
            # node["Surprising Retweets"] = community2SurprisingSummaryRetweets[community]
            # if(not node["Surprising Retweets"]):
            #     node["Surprising Retweets"] = "None"
            node["Surprising Tokens"] = community2SurprisingSummaryTokens[community]
            if(not node["Surprising Tokens"]):
                node["Surprising Tokens"] = "None"
            node["Surprising Retweet Tokens"] = community2SurprisingSummaryRetweetTokens[community]
            if(not node["Surprising Retweet Tokens"]):
                node["Surprising Retweet Tokens"] = "None"

        xn.save(gThresholded,thresholdedNetworkPath)
        # save a txt file with the community information
        with open(tablesOutputPath/f"{dataName}_{suffix}_{networkType}_thres_{thresholdAttribute}_{threshold}_communities.txt","w") as f:
            for attribute in gThresholded.vertex_attributes():
                if(attribute.startswith("Top") or attribute.startswith("Surprising")):
                    attributeArray = gThresholded.vs[attribute]
                    # order by most common (ignore None)
                    attributeCounter = Counter(attributeArray)
                    attributeCounter.pop("None",None)
                    f.write(f"{attribute}:\n")
                    for entry,count in attributeCounter.most_common(15):
                        f.write(f"\t{entry}\n")
                    f.write("\n")

        





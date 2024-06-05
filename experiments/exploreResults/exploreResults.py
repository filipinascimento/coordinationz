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
import shutil
import json
from collections import Counter

dataName = "sampled_20240226"
networkType = "coretweet"
suffix = "coretweet"

configPath = None

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

networksPath = Path(config["paths"]["NETWORKS"]).resolve()
networkPath = networksPath/f"{dataName}_{suffix}_{networkType}.xnet"

# df = czpre.loadPreprocessedData(dataName, config=config)

g = xn.load(networkPath)

# get top 20 edges by quantile
topEdges = []
labels = g.vs["Label"]
scores = g.es["quantile"]
if("similarity" in g.es.attributes()):
    similarities = np.array(g.es["similarity"])
elif("weight" in g.es.attributes()):
    similarities = np.array(g.es["weight"])

# plot similarity vs score
import matplotlib.pyplot as plt
plt.figure()
plt.scatter(similarities, scores)
plt.xlabel("Similarity")
plt.ylabel("Quantile")
plt.title("Similarity vs Quantile")
plt.savefig(f"Outputs/Figures/sim_quantile_{dataName}_{suffix}_{networkType}.png")
plt.close()


# get top 20 edges by quantile
# similarities = g.es["similarity"]
quantiles = np.array(g.es["quantile"])
# plot similarity distribution cummulative

quantileMarkers = [0.99,0.95,0.90]
plt.figure()
plt.hist(similarities, bins=100, cumulative=True, density=True)
# create vertical bars for quantiles markers
for marker in quantileMarkers:
    # use quantile variable first point above or equal quantile
    quantilePosition = similarities[np.argmax(quantiles >= marker)]
    # quantilePosition = np.quantile(similarities, marker)
    plt.axvline(quantilePosition,color="red", linestyle="--")
    # also add text
    # number of links
    remainingLinks = np.sum(similarities >= quantilePosition)
    print(f"Number of links above {marker}: {remainingLinks}")
    # number of remaining nodes in the network
    gthreshold = g.copy()
    if("similarity" in g.es.attributes()):
        gthreshold.delete_edges(gthreshold.es.select(similarity_lt=quantilePosition))
    elif("weight" in g.es.attributes()):
        gthreshold.delete_edges(gthreshold.es.select(weight_lt=quantilePosition))
    # remove singletons
    gthreshold.delete_vertices(gthreshold.vs.select(_degree=0))
    remainingNodes = len(gthreshold.vs)
    print(f"Number of nodes in the network above {marker}: {remainingNodes}")
    plt.text(quantilePosition, 0.5, f"{marker} ({remainingLinks}, {remainingNodes})", rotation=90)

plt.xlabel("Similarity")
plt.ylabel("Density")
plt.title("Similarity Distribution")

plt.savefig(f"Outputs/Figures/sim_distribution_{dataName}_{suffix}_{networkType}.png")
plt.close()


topEdgeIndices = np.argsort(scores)[-200:]
# reverse the order to get the top 20
topEdgeIndices = topEdgeIndices[::-1]
allEdges = g.get_edgelist()
if("Type" in g.es.attributes()):
    edgeTypes = g.es["Type"]
else:
    edgeTypes = [networkType] * len(allEdges)

for edgeIndex in topEdgeIndices:
    edge = allEdges[edgeIndex]
    score = scores[edgeIndex]
    edgeType = edgeTypes[edgeIndex]
    similarity = similarities[edgeIndex]
    topEdges.append((labels[edge[0]], labels[edge[1]], score, edgeType, similarity))
    

def printEdge(edge):
    user1 = edge[0]
    user2 = edge[1]
    score = edge[2]
    edgeType = edge[3]
    similarity = edge[4]
    print(f"{user1} - {user2}")
    print(f"({edgeType}) : {score} (sim. {similarity})")
    # [(user,hashtag) for user,hashtag_list in zip(users,hashtags) for hashtag in hashtag_list]
    # pandas has user_id, tweet_type, hashtags, urls, text
    user1Data = df[df["user_id"] == user1]
    user2Data = df[df["user_id"] == user2]
    hashtagsLists1 = user1Data[user1Data.tweet_type!="retweet"]["hashtags"].values
    hashtagsLists2 = user2Data[user2Data.tweet_type!="retweet"]["hashtags"].values
    
    hashtags1 = Counter([hashtag for hashtagsList in hashtagsLists1 for hashtag in hashtagsList])
    hashtags2 = Counter([hashtag for hashtagsList in hashtagsLists2 for hashtag in hashtagsList])
    # calculate cosine similarity between two users via hashtag
    numerator = sum([hashtags1[hashtag] * hashtags2[hashtag] for hashtag in hashtags1.keys() if hashtag in hashtags2.keys()])
    denominator = np.sqrt(sum([hashtags1[hashtag]**2 for hashtag in hashtags1.keys()]) * sum([hashtags2[hashtag]**2 for hashtag in hashtags2.keys()]))
    cosineSimilarity = numerator / denominator
    print("\t Hashtags cosine similarity:", cosineSimilarity)

    linkedIDsList1 = user1Data[user1Data.tweet_type=="retweet"]["linked_tweet"].values
    linkedIDsList2 = user2Data[user2Data.tweet_type=="retweet"]["linked_tweet"].values
    linkedIDs1 = Counter(linkedIDsList1)
    linkedIDs2 = Counter(linkedIDsList2)
    # calculate cosine similarity between two users via retweet
    numerator = sum([linkedIDs1[linkedID] * linkedIDs2[linkedID] for linkedID in linkedIDs1.keys() if linkedID in linkedIDs2.keys()])
    denominator = np.sqrt(sum([linkedIDs1[linkedID]**2 for linkedID in linkedIDs1.keys()]) * sum([linkedIDs2[linkedID]**2 for linkedID in linkedIDs2.keys()]))
    cosineSimilarity = numerator / denominator
    print("\t Retweet cosine similarity:", cosineSimilarity)
    urlsLists1 = user1Data[user1Data.tweet_type!="retweet"]["urls"].values
    urlsLists2 = user2Data[user2Data.tweet_type!="retweet"]["urls"].values

    urls1 = Counter([url for urlsList in urlsLists1 for url in urlsList])
    urls2 = Counter([url for urlsList in urlsLists2 for url in urlsList])

    print("\t Hashtags:")
    print("\t\t User 1:", [entry for entry,_ in hashtags1.most_common()])
    print("\t\t User 2:", [entry for entry,_ in hashtags2.most_common()])
    print("\t URLs:")
    print("\t\t User 1:", [f"{entry}:{counts}" for entry,counts in urls1.most_common()])
    print("\t\t User 2:", [f"{entry}:{counts}" for entry,counts in urls2.most_common()])
    # two columns for text, one for each user
    text1 = sorted(user1Data["text"].values)
    text2 = sorted(user2Data["text"].values)
    print("\t Text: \n\tUser1:")
    for text in text1:
        # print("\t\t", text)
        # break text into lines and add padding
        text = text.split()
        lines = []
        line = ""
        for word in text:
            if len(line) + len(word) > 80:
                lines.append(line)
                line = ""
            line += word + " "
        lines.append(line)
        for line in lines:
            print("\t\t ", line)
    print("\tUser2:")
    # print("\t\t", text2)
    for text in text2:
        # break text into lines and add padding
        text = text.split()
        lines = []
        line = ""
        for word in text:
            if len(line) + len(word) > 80:
                lines.append(line)
                line = ""
            line += word + " "
        lines.append(line)
        for line in lines:
            print("\t\t ", line)

# print similarity of the top 10 edges
# print("Top 10 edges:")
# for edge in topEdges[0:10]:
#     print(edge[4])

# for edge in topEdges[0:1]:
#     printEdge(edge)

printEdge(topEdges[1])


# # create dictionary of user (user_id) to all retweets as list (linked_tweet) 
# user2retweetSets = {}
# allUsers = set(labels)
# # onlyEntries in set
# dfInNetwork = df[df["user_id"].isin(allUsers)].dropna(subset=["linked_tweet"])

# for user,linkedTweet in tqdm(dfInNetwork[["user_id","linked_tweet"]].values):
#     if user in user2retweetSets:
#         user2retweetSets[user].add(linkedTweet)
#     else:
#         user2retweetSets[user] = set([linkedTweet])

# overlaps = []
# for edgeIndex, edge in enumerate(tqdm(allEdges)):
#     user1 = labels[edge[0]]
#     user2 = labels[edge[1]]
#     quantile = quantiles[edgeIndex]
#     if quantile > 0.99999:
#         if user1 in user2retweetSets and user2 in user2retweetSets:
#             overlap = len(user2retweetSets[user1].intersection(user2retweetSets[user2]))
#             overlaps.append(overlap)

# print("\n".join([f"{overlaps}:{overlapCount}" for overlaps,overlapCount in sorted(Counter(overlaps).items())]))
# # printEdge(topEdges[100])



# create dictionary of user (user_id) to all retweets as list (linked_tweet) 
user2urlSets = {}
allUsers = set(labels)
# onlyEntries in set
dfInNetwork = df[df["user_id"].isin(allUsers)].dropna(subset=["urls"])

for user,urls in tqdm(dfInNetwork[["user_id","urls"]].values):
    if user in user2urlSets:
        user2urlSets[user].update(urls)
    else:
        user2urlSets[user] = set(urls)

topOverlaps = []
topSimilarities = []
for edgeIndex in tqdm(topEdgeIndices[0:10]):
    edge = allEdges[edgeIndex]
    user1 = labels[edge[0]]
    user2 = labels[edge[1]]
    quantile = quantiles[edgeIndex]
    if quantile > 0.0:
        if user1 in user2urlSets and user2 in user2urlSets:
            overlap = len(user2urlSets[user1].intersection(user2urlSets[user2]))
            topOverlaps.append(overlap)
            topSimilarities.append(similarities[edgeIndex])

print("\n".join(map(str,list(zip(topOverlaps,topSimilarities)))))
print("\n".join([f"{overlaps}:{overlapCount}" for overlaps,overlapCount in sorted(Counter(topOverlaps).items())]))
# printEdge(topEdges[100])

dfOutput = pd.DataFrame(topEdges[0:10], columns=["user1","user2","score","type","similarity"])
dfOutput.drop(columns=["type","similarity","score"], inplace=True)
# save networkType_suspicious.csv
dfOutput.to_csv(f"Outputs/Tables/{dataName}_{suffix}_{networkType}_suspicious.csv", index=False)


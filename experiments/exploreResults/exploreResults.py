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

dataName = "TA2_full_eval_NO_GT_nat+synth_2024-05-29-cleaned"
networkType = "merged"
suffix = "nonullmodel"

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

df = czpre.loadPreprocessedData(dataName, config=config)

g = xn.load(networkPath)

# get top 20 edges by quantile
topEdges = []
labels = g.vs["Label"]
scores = g.es["quantile"]
if("similarity" in g.es.attributes()):
    similarities = g.es["similarity"]
if("weight" in g.es.attributes()):
    similarities = g.es["weight"]

# plot similarity vs score
import matplotlib.pyplot as plt
plt.figure()
plt.scatter(similarities, scores)
plt.xlabel("Similarity")
plt.ylabel("Score")
plt.title("Similarity vs Score")
plt.savefig("Outputs/Figures/similarity_vs_score.png")
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
    urlsLists1 = user1Data[user1Data.tweet_type!="retweet"]["urls"].values
    urlsLists2 = user2Data[user2Data.tweet_type!="retweet"]["urls"].values

    urls1 = Counter([url for urlsList in urlsLists1 for url in urlsList])
    urls2 = Counter([url for urlsList in urlsLists2 for url in urlsList])

    print("\t Hashtags:")
    print("\t\t User 1:", [entry for entry,_ in hashtags1.most_common()])
    print("\t\t User 2:", [entry for entry,_ in hashtags2.most_common()])
    print("\t URLs:")
    print("\t\t User 1:", [entry for entry,_ in urls1.most_common()])
    print("\t\t User 2:", [entry for entry,_ in urls2.most_common()])
    # two columns for text, one for each user
    text1 = sorted(user1Data["text"].values)
    text2 = sorted(user2Data["text"].values)
    print("\t Text: \n\tUser1:")
    for text in text1:
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


    

for edge in topEdges[1:2]:
    print(edge)
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

dataName = "TA2_full_eval_NO_GT_nat_2024-06-03"
networkType = "merged"
suffix = "all_coword_min0.5_v5"
# suffix = "all_union_coword_0.8_0.85"
# scoresAttribute = "weight" # UNION
scoresAttribute = "quantile" # SOFT
threshold = 0.95
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
networkPath = networksPath/f"{dataName}_{suffix}_{networkType}_{threshold}.xnet"

g = xn.load(networkPath)

df = czpre.loadPreprocessedData(dataName, config=config)


# get top 20 edges by quantile
topEdges = []
labels = g.vs["Label"]
if("similarity" in g.es.attributes()):
    similarities = np.array(g.es["similarity"])
elif("weight" in g.es.attributes()):
    similarities = np.array(g.es["weight"])

scores = g.es[scoresAttribute]

# plot similarity vs score
# import matplotlib.pyplot as plt
# plt.figure()
# plt.scatter(similarities, scores)
# plt.xlabel("Similarity")
# plt.ylabel("Quantile")
# plt.title("Similarity vs Quantile")
# plt.savefig(f"Outputs/Figures/sim_quantile_{dataName}_{suffix}_{networkType}.png")
# plt.close()


# get top 20 edges by quantile
# similarities = g.es["similarity"]
quantiles = np.array(g.es["quantile"])
# plot similarity distribution cummulative

# quantileMarkers = [0.99,0.95,0.90]
# plt.figure()
# plt.hist(similarities, bins=100, cumulative=True, density=True)
# # create vertical bars for quantiles markers
# for marker in quantileMarkers:
#     # use quantile variable first point above or equal quantile
#     quantilePosition = similarities[np.argmax(quantiles >= marker)]
#     # quantilePosition = np.quantile(similarities, marker)
#     plt.axvline(quantilePosition,color="red", linestyle="--")
#     # also add text
#     # number of links
#     remainingLinks = np.sum(similarities >= quantilePosition)
#     print(f"Number of links above {marker}: {remainingLinks}")
#     # number of remaining nodes in the network
#     gthreshold = g.copy()
#     if("similarity" in g.es.attributes()):
#         gthreshold.delete_edges(gthreshold.es.select(similarity_lt=quantilePosition))
#     elif("weight" in g.es.attributes()):
#         gthreshold.delete_edges(gthreshold.es.select(weight_lt=quantilePosition))
#     # remove singletons
#     gthreshold.delete_vertices(gthreshold.vs.select(_degree=0))
#     remainingNodes = len(gthreshold.vs)
#     print(f"Number of nodes in the network above {marker}: {remainingNodes}")
#     plt.text(quantilePosition, 0.5, f"{marker} ({remainingLinks}, {remainingNodes})", rotation=90)

# plt.xlabel("Similarity")
# plt.ylabel("Density")
# plt.title("Similarity Distribution")

# plt.savefig(f"Outputs/Figures/sim_distribution_{dataName}_{suffix}_{networkType}.png")
# plt.close()


topEdgeIndices = np.argsort(scores)
# reverse the order to get the top 20
topEdgeIndices = topEdgeIndices[::-1]
allEdges = g.get_edgelist()
if("Type" in g.es.attributes()):
    edgeTypes = g.es["Type"]
else:
    edgeTypes = [networkType] * len(allEdges)

topEdges = []
for edgeIndex in topEdgeIndices:
    edge = allEdges[edgeIndex]
    score = scores[edgeIndex]
    edgeType = edgeTypes[edgeIndex]
    similarity = similarities[edgeIndex]
    topEdges.append((labels[edge[0]], labels[edge[1]], score, edgeType, similarity))
    

user2Community = {user:int(community) for user,community in zip(g.vs["Label"],g.vs["CommunityIndex"])}

user2CommunityDescriptions = {}
for attribute in g.vs.attributes():
    if(attribute.startswith("Surprising")):
        user2CommunityDescriptions[attribute] = {user:community for user,community in zip(g.vs["Label"],g.vs[attribute])}

user2Synthetic = {}
if "Synthetic" in g.vs.attributes():
    user2Synthetic = {user:synthetic for user,synthetic in zip(g.vs["Label"],g.vs["Synthetic"])}

def printEdge(edge,file=sys.stdout):
    user1 = edge[0]
    user2 = edge[1]
    score = edge[2]
    if(user1 in user2Community):
        community1 = user2Community[user1]
    else:
        community1 = "None"
    if(user2 in user2Community):
        community2 = user2Community[user2]
    else:
        community2 = "None"
    edgeType = edge[3]
    similarity = edge[4]
    file.write(f"{user1} - {user2}"+"\n")
    file.write(f"({edgeType}) : {score} (sim. {similarity})"+"\n")
    file.write("Communities: "+str(community1)+" "+str(community2)+"\n")
    file.write("Synthetic: "+str(user2Synthetic.get(user1,"None"))+" "+str(user2Synthetic.get(user2,"None"))+"\n")
    file.write("Community descriptions:\n")
    for attribute, user2description in user2CommunityDescriptions.items():
        file.write(f"\t{attribute}:\n\t\tuser1: {user2description[user1]}\n\t\tuser2:{user2description[user2]}")

    # [(user,hashtag) for user,hashtag_list in zip(users,hashtags) for hashtag in hashtag_list]
    # pandas has user_id, tweet_type, hashtags, urls, text
    user1Data = df[df["user_id"] == user1]
    user2Data = df[df["user_id"] == user2]
    hashtagsLists1 = user1Data[user1Data.tweet_type!="retweet"]["hashtags"].values
    hashtagsLists2 = user2Data[user2Data.tweet_type!="retweet"]["hashtags"].values
    
    hashtags1 = Counter([hashtag for hashtagsList in hashtagsLists1 for hashtag in hashtagsList])
    hashtags2 = Counter([hashtag for hashtagsList in hashtagsLists2 for hashtag in hashtagsList])
    # calculate cosine similarity between two users via hashtag
    # numerator = sum([hashtags1[hashtag] * hashtags2[hashtag] for hashtag in hashtags1.keys() if hashtag in hashtags2.keys()])
    # denominator = np.sqrt(sum([hashtags1[hashtag]**2 for hashtag in hashtags1.keys()]) * sum([hashtags2[hashtag]**2 for hashtag in hashtags2.keys()]))
    # cosineSimilarity = numerator / denominator
    # print("\t Hashtags cosine similarity:", cosineSimilarity)

    linkedIDsList1 = user1Data[user1Data.tweet_type=="retweet"]["linked_tweet"].values
    linkedIDsList2 = user2Data[user2Data.tweet_type=="retweet"]["linked_tweet"].values
    linkedIDs1 = Counter(linkedIDsList1)
    linkedIDs2 = Counter(linkedIDsList2)
    # calculate cosine similarity between two users via retweet
    numerator = sum([linkedIDs1[linkedID] * linkedIDs2[linkedID] for linkedID in linkedIDs1.keys() if linkedID in linkedIDs2.keys()])
    denominator = np.sqrt(sum([linkedIDs1[linkedID]**2 for linkedID in linkedIDs1.keys()]) * sum([linkedIDs2[linkedID]**2 for linkedID in linkedIDs2.keys()]))
    cosineSimilarity = numerator / denominator
    file.write("\t Retweet cosine similarity: "+str(cosineSimilarity)+"\n")
    urlsLists1 = user1Data[user1Data.tweet_type!="retweet"]["urls"].values
    urlsLists2 = user2Data[user2Data.tweet_type!="retweet"]["urls"].values

    urls1 = Counter([url for urlsList in urlsLists1 for url in urlsList])
    urls2 = Counter([url for urlsList in urlsLists2 for url in urlsList])

    file.write("\t Hashtags:"+"\n")
    file.write("\t\t User 1: "+str([f"{entry}:{count}" for entry,count in hashtags1.most_common()])+"\n")
    file.write("\t\t User 2: "+str([f"{entry}:{count}" for entry,count in hashtags2.most_common()])+"\n")
    file.write("\t URLs:"+"\n")
    file.write("\t\t User 1: "+str([f"{entry}:{counts}" for entry,counts in urls1.most_common()])+"\n")
    file.write("\t\t User 2: "+str([f"{entry}:{counts}" for entry,counts in urls2.most_common()])+"\n")
    # two columns for text, one for each user
    # concatenate text and creation_date as text (creation_date)
    user1ParendIDs = user1Data["linked_tweet"].replace(np.nan, "None")
    user2ParendIDs = user2Data["linked_tweet"].replace(np.nan, "None") 
    textDate1 = user1Data["text"] + " (" + user1Data["creation_date"] + ") -- " + user1ParendIDs
    textDate2 = user2Data["text"] + " (" + user2Data["creation_date"] + ") -- " + user2ParendIDs
    text1 = sorted(textDate1.values)
    text2 = sorted(textDate2.values)
    file.write("\t Text: \n\tUser1:"+"\n")
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
            file.write("\t\t "+line+"\n")
    file.write("\tUser2:\n")
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
            file.write("\t\t "+line+"\n")

# print similarity of the top 10 edges
# print("Top 10 edges:")
# for edge in topEdges[0:10]:
#     print(edge[4])

# for edge in topEdges[0:1]:
#     printEdge(edge)

selectedCommunityIndex = 265
topEdgesFiltered = [edges for edges in topEdges if user2Community[edges[0]] == selectedCommunityIndex and user2Community[edges[1]] == selectedCommunityIndex]
print("Community filtered edges:")
print(Counter([edge[3] for edge in topEdgesFiltered]))
printEdge(topEdgesFiltered[0])


selectedCommunityIndex = 6
topEdgesFiltered = [edges for edges in topEdges if user2Community[edges[0]] != selectedCommunityIndex and user2Community[edges[1]] != selectedCommunityIndex]
printEdge(topEdgesFiltered[0])
# 
findEdge = ("af5c7bfe2c638ee0af8ba1dbe18060272d924a8848013d64205d1438cc3d66b2" , "e4e6dfc75e2912a8aba8a1614322b994ba1df858348bfa907b6d66e3a8355e70")
# find edge in topEdges
edgeTest = None
for edge in topEdges:
    if edge[0] == findEdge[0] and edge[1] == findEdge[1]:
        edgeTest = edge
        break
    if edge[0] == findEdge[1] and edge[1] == findEdge[0]:
        edgeTest = edge
        break
if(edgeTest is None):
    edgeTest = (findEdge[0], findEdge[1], 0.0, "None", 0.0)

with open(f"Outputs/Tables/{dataName}_{suffix}_{networkType}_edgeTest.txt", "w") as file:
    # printEdge(edgeTest, file=file)
    printEdge(topEdgesFiltered[3],file)


with open(f"Outputs/Tables/{dataName}_{suffix}_{networkType}_edgeTest.txt", "w") as file:
    # printEdge(edgeTest, file=file)
    printEdge(topEdges[-100],file)


printEdge(topEdges[0])

# user2Community = {user:int(community) for user,community in zip(g.vs["Label"],g.vs["CommunityIndex"])}
# topEdgesFilteredSynth = [edges for edges in topEdges if user2Community[edges[0]] == selectedCommunityIndex and user2Community[edges[1]] == selectedCommunityIndex]
# print("Synthetic filtered edges:")
# print(Counter([edge[3] for edge in topEdgesFiltered]))
# printEdge(topEdgesFiltered[0])

# create dictionary of user (user_id) to all retweets as list (linked_tweet) 
user2retweetSets = {}
allUsers = set(labels)
# onlyEntries in set
dfInNetwork = df[df["user_id"].isin(allUsers)].dropna(subset=["linked_tweet"])

for user,linkedTweet in tqdm(dfInNetwork[["user_id","linked_tweet"]].values):
    if user in user2retweetSets:
        user2retweetSets[user].add(linkedTweet)
    else:
        user2retweetSets[user] = set([linkedTweet])

overlaps = []
for edgeIndex, edge in enumerate(tqdm(allEdges)):
    user1 = labels[edge[0]]
    user2 = labels[edge[1]]
    quantile = quantiles[edgeIndex]
    if quantile > 0.99999:
        if user1 in user2retweetSets and user2 in user2retweetSets:
            overlap = len(user2retweetSets[user1].intersection(user2retweetSets[user2]))
            overlaps.append(overlap)

print("\n".join([f"{overlaps}:{overlapCount}" for overlaps,overlapCount in sorted(Counter(overlaps).items())]))
# printEdge(topEdges[100])



# # create dictionary of user (user_id) to all retweets as list (linked_tweet) 
# user2urlSets = {}
# allUsers = set(labels)
# # onlyEntries in set
# dfInNetwork = df[df["user_id"].isin(allUsers)].dropna(subset=["urls"])

# for user,urls in tqdm(dfInNetwork[["user_id","urls"]].values):
#     if user in user2urlSets:
#         user2urlSets[user].update(urls)
#     else:
#         user2urlSets[user] = set(urls)

# topOverlaps = []
# topSimilarities = []
# for edgeIndex in tqdm(topEdgeIndices[0:10]):
#     edge = allEdges[edgeIndex]
#     user1 = labels[edge[0]]
#     user2 = labels[edge[1]]
#     quantile = quantiles[edgeIndex]
#     if quantile > 0.0:
#         if user1 in user2urlSets and user2 in user2urlSets:
#             overlap = len(user2urlSets[user1].intersection(user2urlSets[user2]))
#             topOverlaps.append(overlap)
#             topSimilarities.append(similarities[edgeIndex])

# print("\n".join(map(str,list(zip(topOverlaps,topSimilarities)))))
# print("\n".join([f"{overlaps}:{overlapCount}" for overlaps,overlapCount in sorted(Counter(topOverlaps).items())]))
# printEdge(topEdges[100])

# dfOutput = pd.DataFrame(topEdges[0:10], columns=["user1","user2","score","type","similarity"])
# dfOutput.drop(columns=["type","similarity","score"], inplace=True)
# # save networkType_suspicious.csv
# dfOutput.to_csv(f"Outputs/Tables/{dataName}_{suffix}_{networkType}_suspicious.csv", index=False)


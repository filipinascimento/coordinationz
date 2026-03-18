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

dataName = "iran_202012_tweets"
networkType = "merged"
suffix = "softunion_noretweetusers_null_zero"
thresholdAttribute = "quantile"
similarityMeasure = "pvalue"
threshold = 1
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
# "cuba_082020_tweets_softunion_null_zero_merged_1.xnet"

df = czpre.loadPreprocessedData(dataName, config=config)
# "cuba_082020_tweets_softunion_null_zero_merged_1.xnet"

g = xn.load(networkPath)

# get top 20 edges by quantile
labels = g.vs["Label"]
scores = g.es[similarityMeasure]
if("similarity" in g.es.attributes()):
    similarities = np.array(g.es["similarity"])
elif("weight" in g.es.attributes()):
    similarities = np.array(g.es["weight"])

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
# quantiles = np.array(g.es["quantile"])
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


# topEdgeIndices = np.argsort(scores)
# filter edges based on >0.9999 threshold
topEdgeIndices = np.where(np.array(scores) >= 0.9999)[0]
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


user2Category = {user:category for user,category in zip(df["user_id"],df["category"])}



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
            # should have at least 2 hashtags in the considered tweets.
            userActivityCount = df[df["hashtags"].apply(lambda x: len(x) > 1)]["user_id"].value_counts()
            usersWithMinActivities = set(userActivityCount[userActivityCount >= minUserActivities].index)
        elif("url" in activityType.lower()):
            # len(urls) should be >0
            userActivityCount = df[df["urls"].apply(lambda x: len(x) > 0)]["user_id"].value_counts()
            usersWithMinActivities = set(userActivityCount[userActivityCount >= minUserActivities].index)
        # TODO: include retweet users
        else:
            # activity not retweet
            userActivityCount = df["user_id"].value_counts()
            usersWithMinActivities = set(userActivityCount[userActivityCount >= minUserActivities].index)
        df = df[df["user_id"].isin(usersWithMinActivities)]
    return df
  


def obtainBipartiteEdgesRetweetsUsers(df):
    # keep only tweet_type == "retweet"
    # if linked_tweet or tweet_type or user_id are not in the dataframe, return an empty list
    if "linked_tweet_user_id" not in df or "tweet_type" not in df or "user_id" not in df:
        return []
    df = df[df["tweet_type"] == "retweet"]
    bipartiteEdges = df[["user_id","linked_tweet_user_id"]].apply(tuple, axis=1).tolist()
    return bipartiteEdges


df = filterUsersByMinActivities(df, minUserActivities=10, activityType="any")
df = df[df["tweet_type"] == "retweet"]
# bipartiteEdges = obtainBipartiteEdgesRetweetsUsers(df)
# bipartiteEdges = czind.filterNodes(bipartiteEdges, minRightDegree=10, minLeftStrength=10)

# allowedUsers = set([userID for userID,_ in bipartiteEdges])
# allowedLinkedUsers = set([linkedUser for _,linkedUser in bipartiteEdges])

# filter by the allowed users (user_id) and linked_tweet_user_id
# dfFiltered = df[df["user_id"].isin(allowedUsers) & df["linked_tweet_user_id"].isin(allowedLinkedUsers)]
# dfFiltered = 


def printEdge(edge,printTweets=True):
    user1 = edge[0]
    user2 = edge[1]
    score = edge[2]
    community1 = user2Community[user1]
    community2 = user2Community[user2]
    edgeType = edge[3]
    similarity = edge[4]
    print(f"{user1} - {user2}")
    print(f"({edgeType}) : {score} (sim. {similarity})")
    # print("Communities: ", community1, community2)
    # print("Community descriptions:")
    # for attribute, user2description in user2CommunityDescriptions.items():
    #     print(f"\t{attribute}:\n\t\tuser1: {user2description[user1]}\n\t\tuser2: {user2description[user2]}")

    # [(user,hashtag) for user,hashtag_list in zip(users,hashtags) for hashtag in hashtag_list]
    # pandas has user_id, tweet_type, hashtags, urls, text
    user1Data = df[df["user_id"] == user1]
    user2Data = df[df["user_id"] == user2]

    hashtagsLists1 = user1Data[user1Data.tweet_type!="retweet"]["hashtags"].values
    hashtagsLists2 = user2Data[user2Data.tweet_type!="retweet"]["hashtags"].values

    hashtags1 = Counter([hashtag for hashtagsList in hashtagsLists1 for hashtag in hashtagsList])
    hashtags2 = Counter([hashtag for hashtagsList in hashtagsLists2 for hashtag in hashtagsList])

    # shared hashtags
    sharedHashtags = set(hashtags1.keys()).intersection(set(hashtags2.keys()))
    print("\t Hashtags in common:", sharedHashtags)
    print("\t Hashtags in common count:", len(sharedHashtags))
    print("\t Hashtags shared by User1:", len(hashtags1))
    print("\t Hashtags shared by User2:", len(hashtags2))

    # print("\t User1 hashtags:", hashtags1)
    # print("\t User2 hashtags:", hashtags2)
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
    # numerator = sum([linkedIDs1[linkedID] * linkedIDs2[linkedID] for linkedID in linkedIDs1.keys() if linkedID in linkedIDs2.keys()])
    # denominator = np.sqrt(sum([linkedIDs1[linkedID]**2 for linkedID in linkedIDs1.keys()]) * sum([linkedIDs2[linkedID]**2 for linkedID in linkedIDs2.keys()]))
    # cosineSimilarity = numerator / denominator
    # print("\t Retweet cosine similarity:", cosineSimilarity)
    urlsLists1 = user1Data[user1Data.tweet_type!="retweet"]["urls"].values
    urlsLists2 = user2Data[user2Data.tweet_type!="retweet"]["urls"].values

    urls1 = Counter([url for urlsList in urlsLists1 for url in urlsList])
    urls2 = Counter([url for urlsList in urlsLists2 for url in urlsList])

    print("\t\t --------")
    print("\t Hashtags:")
    print("\t\t User 1:", [entry for entry,_ in hashtags1.most_common()])
    print("\t\t --------")
    print("\t\t User 2:", [entry for entry,_ in hashtags2.most_common()])
    print("\t\t --------")


    # Now print the linked_tweet_user_id (users who they retweeted) and the shared users
    print("\t Retweet Users:")
    linkedUserIDs1 = user1Data[user1Data.tweet_type=="retweet"]["linked_tweet_user_id"].values
    linkedUserIDs2 = user2Data[user2Data.tweet_type=="retweet"]["linked_tweet_user_id"].values
    print("\t\t User 1:", [entry for entry,_ in Counter(linkedUserIDs1).most_common()])
    print("\t\t User 2:", [entry for entry,_ in Counter(linkedUserIDs2).most_common()])
    sharedRetweetUsers = set(linkedUserIDs1).intersection(set(linkedUserIDs2))
    print("\t Retweet Users in common:", sharedRetweetUsers)
    print("\t Retweet Users in common count:", len(sharedRetweetUsers))
    print("\t Retweet Users shared by User1:", len(linkedUserIDs1))
    print("\t Retweet Users shared by User2:", len(linkedUserIDs2))

    # print("\t URLs:")
    # print("\t\t User 1:", [f"{entry}:{counts}" for entry,counts in urls1.most_common()])
    # print("\t\t User 2:", [f"{entry}:{counts}" for entry,counts in urls2.most_common()])
    # two columns for text, one for each user
    # concatenate text and creation_date as text (creation_date)
    textDate1 = user1Data[user1Data.tweet_type!="retweet"]["text"] + " (" + user1Data["created_at"] + ")"
    textDate2 = user2Data[user2Data.tweet_type!="retweet"]["text"] + " (" + user2Data["created_at"] + ")"
    text1 = sorted(textDate1.dropna().values)
    text2 = sorted(textDate2.dropna().values)

    if(printTweets):
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


    
    retweets1 = user1Data[user1Data.tweet_type=="retweet"]
    retweets2 = user2Data[user2Data.tweet_type=="retweet"]
    retweetsInCommon = set(retweets1["linked_tweet"].values).intersection(set(retweets2["linked_tweet"].values))
    retweets1shared = retweets1[retweets1["linked_tweet"].isin(retweetsInCommon)]
    retweets2shared = retweets2[retweets2["linked_tweet"].isin(retweetsInCommon)]
    print("\t Retweets:")
    print("\t\t Retweets in common:", len(retweetsInCommon))
    # print("\t\t Common RTs by User1:", len(retweets1shared))
    # print("\t\t Common RTs by User2:", len(retweets2shared))
    print("\t\t Total RTs by User1:", len(retweets1))
    print("\t\t Total RTs by User2:", len(retweets2))
    print("\t\t Retweets User1:") #Text
    print(f"{user1} - {user2}")
    print(f"({edgeType}) : {score} (sim. {similarity})")
    print(f"Edge type: {edgeType}")
    category1 = user1Data["category"].iloc[0]
    category2 = user2Data["category"].iloc[0]
    print(f"\t User1 category: {category1}")
    print(f"\t User2 category: {category2}")
    print(f"-----")
    # allRetweetsShared = pd.concat([retweets1shared,retweets2shared])
    # # aggregate on liked_tweet first: text, but creation date create a list
    # allRetweetsShared = allRetweetsShared.groupby("linked_tweet").agg({"text":"first","creation_date":list})
    # allRetweetsShared["creation_date"] = allRetweetsShared["creation_date"].apply(lambda x: ", ".join(x))
    # allRetweetsShared = allRetweetsShared.reset_index()
    # allRetweetsSharedText = allRetweetsShared["text"] + " (" + allRetweetsShared["creation_date"] + ")"
    # print("\t\t Text:") #Text
    # for text in allRetweetsSharedText:
    #     # break text into lines and add padding
    #     text = text.split()
    #     lines = []
    #     line = ""
    #     for word in text:
    #         if len(line) + len(word) > 80:
    #             lines.append(line)
    #             line = ""
    #         line += word + " "
    #     lines.append(line)
    #     for line in lines:
    #         print("\t\t ", line)
    #     # retweet content

printEdge(topEdges[100])
# random edge in topEdge
import random
topEdgesAllUsers = set([edge[0] for edge in topEdges]).union(set([edge[1] for edge in topEdges]))
topEdgesCategories = [user2Category[user] for user in topEdgesAllUsers]

topEdgesControl = [edge for edge in topEdges if user2Category[edge[0]] == "control" and user2Category[edge[1]] == "control"]

randomEdgeData = random.choice(topEdgesControl)
printEdge(randomEdgeData,printTweets=False)

# print similarity of the top 10 edges
# print("Top 10 edges:")
# for edge in topEdges[0:10]:
#     print(edge[4])

# for edge in topEdges[0:1]:
#     printEdge(edge)

# selectedCommunityIndex = 265
# topEdgesFiltered = [edges for edges in topEdges if user2Community[edges[0]] == selectedCommunityIndex and user2Community[edges[1]] == selectedCommunityIndex]
# print("Community filtered edges:")
# print(Counter([edge[3] for edge in topEdgesFiltered]))
# printEdge(topEdgesFiltered[0])


# create dictionary of user (user_id) to all retweets as list (linked_tweet) 
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



# # # create dictionary of user (user_id) to all retweets as list (linked_tweet) 
# # user2urlSets = {}
# # allUsers = set(labels)
# # # onlyEntries in set
# # dfInNetwork = df[df["user_id"].isin(allUsers)].dropna(subset=["urls"])

# # for user,urls in tqdm(dfInNetwork[["user_id","urls"]].values):
# #     if user in user2urlSets:
# #         user2urlSets[user].update(urls)
# #     else:
# #         user2urlSets[user] = set(urls)

# # topOverlaps = []
# # topSimilarities = []
# # for edgeIndex in tqdm(topEdgeIndices[0:10]):
# #     edge = allEdges[edgeIndex]
# #     user1 = labels[edge[0]]
# #     user2 = labels[edge[1]]
# #     quantile = quantiles[edgeIndex]
# #     if quantile > 0.0:
# #         if user1 in user2urlSets and user2 in user2urlSets:
# #             overlap = len(user2urlSets[user1].intersection(user2urlSets[user2]))
# #             topOverlaps.append(overlap)
# #             topSimilarities.append(similarities[edgeIndex])

# # print("\n".join(map(str,list(zip(topOverlaps,topSimilarities)))))
# # print("\n".join([f"{overlaps}:{overlapCount}" for overlaps,overlapCount in sorted(Counter(topOverlaps).items())]))
# # printEdge(topEdges[100])

# # dfOutput = pd.DataFrame(topEdges[0:10], columns=["user1","user2","score","type","similarity"])
# # dfOutput.drop(columns=["type","similarity","score"], inplace=True)
# # # save networkType_suspicious.csv
# # dfOutput.to_csv(f"Outputs/Tables/{dataName}_{suffix}_{networkType}_suspicious.csv", index=False)

# 

linkedUsersCounts = Counter(dfFiltered["linked_tweet_user_id"].dropna().values)

import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))

linearBins = np.arange(1, max(linkedUsersCounts.values()) + 1, 1)
counts,bins = np.histogram(list(linkedUsersCounts.values()), bins=linearBins)
counts= counts*bins[1:] # normalize by bin width
counts = counts / np.sum(counts)*100  # normalize to get a probability distribution
# inverse cummulative distribution
counts_inverse = np.cumsum(counts[::-1])[::-1]
plt.bar(bins[:-1], counts_inverse, width=1, color='blue', alpha=0.7, align='edge')
plt.title('Inverse Cumulative Distribution of Linked Tweet User Counts')
plt.xlabel('Number of Linked Tweet Users')
plt.ylabel('Frequency')
plt.grid(axis='y', alpha=0.75)
plt.savefig(f"Outputs/Figures/linked_tweet_user_counts_inverse_{dataName}_{suffix}_{networkType}_linear.png")
plt.close()



# inverse cummulative distribution
plt.figure(figsize=(10, 6))
plt.hist(list(linkedUsersCounts.values()), bins=logbins, color='blue', alpha=0.7, cumulative=-1)
plt.xscale('log')
plt.yscale('log')
plt.title('Inverse Cumulative Distribution of Linked Tweet User Counts')
plt.xlabel('Number of Linked Tweet Users')
plt.ylabel('Frequency')
plt.grid(axis='y', alpha=0.75)
plt.savefig(f"Outputs/Figures/linked_tweet_user_counts_inverse_{dataName}_{suffix}_{networkType}.png")
plt.close()



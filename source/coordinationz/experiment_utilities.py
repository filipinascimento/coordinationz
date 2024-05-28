from . import config

from pathlib import Path
import pandas as pd

# ast evaluates strings that are python expressions
import ast
import numpy as np
from collections import Counter



def loadEvaluationDataset(dataName, config=config, minActivities=0):
    # config = cz.load_config("<path to config>")
    
    # dataName = "challenge_filipinos_5DEC"
    # dataName = "challenge_problem_two_21NOV_activeusers"
    # dataName = "challenge_problem_two_21NOV"

    dataPath = Path(config["paths"]["EVALUATION_DATASETS"])

    df = pd.read_csv(dataPath/f'{dataName}.csv',
                    dtype={
                        'screen_name':str,
                        'linked_tweet':str
                        })

    if minActivities > 0:
        userActivityCount = df["screen_name"].value_counts()
        usersWithMinActivities = set(userActivityCount[userActivityCount >= minActivities].index)
        df = df[df["screen_name"].isin(usersWithMinActivities)]

    return df


def obtainEvaluationBipartiteEdgesRetweets(df,minActivities=1):
    # keep only tweet_type == "retweet"
    df = df[df["tweet_type"] == "retweet"]
    if minActivities > 0:
        userActivityCount = df["screen_name"].value_counts()
        usersWithMinActivities = set(userActivityCount[userActivityCount >= minActivities].index)
        df = df[df["screen_name"].isin(usersWithMinActivities)]
    bipartiteEdges = df[["screen_name","linked_tweet"]].values
    return bipartiteEdges

def loadINCASDataset(dataName, config=config, minActivities=0):
    # config = cz.load_config("<path to config>")
    
    # dataName = "challenge_filipinos_5DEC"
    # dataName = "challenge_problem_two_21NOV_activeusers"
    # dataName = "challenge_problem_two_21NOV"

    dataPath = Path(config["paths"]["INCAS_DATASETS"])

    df = pd.read_csv(dataPath/f'{dataName}.csv',
                    dtype={
                        'screen_name':str,
                        'linked_tweet':str
                        })

    if minActivities > 0:
        userActivityCount = df["screen_name"].value_counts()
        usersWithMinActivities = set(userActivityCount[userActivityCount >= minActivities].index)
        df = df[df["screen_name"].isin(usersWithMinActivities)]

    return df


def obtainINCASBipartiteEdgesRetweets(df,minActivities=1):
    # keep only tweet_type == "retweet"
    df = df[df["tweet_type"] == "retweet"]
    if minActivities > 0:
        userActivityCount = df["screen_name"].value_counts()
        usersWithMinActivities = set(userActivityCount[userActivityCount >= minActivities].index)
        df = df[df["screen_name"].isin(usersWithMinActivities)]
    bipartiteEdges = df[["screen_name","linked_tweet"]].values
    return bipartiteEdges

def obtainINCASBipartiteEdgesURLs(df,removeRetweets=True,minActivities=1,):
    if(removeRetweets):
        df = df[df["tweet_type"] != "retweet"]
    # convert url strings that looks like lists to actual lists
    urls = df["urls"].apply(ast.literal_eval)
    users = df["screen_name"]
    # keep only non-empty lists
    mask = urls.apply(lambda x: len(x) > 0)
    urls = urls[mask]
    users = users[mask]
    # only keep users with at least minActivities
    userActivityCount = users.value_counts()
    usersWithMinActivities = set(userActivityCount[userActivityCount >= minActivities].index)
    mask = users.isin(usersWithMinActivities)
    users = users[mask]
    urls = urls[mask]
    # create edges list users -> urls
    edges = [(user,url) for user,urlList in zip(users,urls) for url in urlList]
    return edges

def obtainINCASBipartiteEdgesHashtags(df,removeRetweets=True,minActivities=1):
    # get hashtags from text content
    if(removeRetweets):
        df = df[df["tweet_type"] != "retweet"]
    # detect hashtags from text #<word>
    hashtags = df["text"].str.findall(r"#\w+")
    users = df["screen_name"]
    # keep only non-empty lists
    mask = hashtags.apply(lambda x: len(x) > 0 if x==x else False)
    hashtags = hashtags[mask]
    users = users[mask]
    # only keep users with at least minActivities
    userActivityCount = users.value_counts()
    usersWithMinActivities = set(userActivityCount[userActivityCount >= minActivities].index)
    mask = users.isin(usersWithMinActivities)
    users = users[mask]
    hashtags = hashtags[mask]

    # create edges list users -> hashtags
    edges = [(user,hashtag) for user,hashtagList in zip(users,hashtags) for hashtag in hashtagList]
    return edges







def loadIODataset(dataName, config=config, flavor="all", minActivities=0):
    dataPath = Path(config["paths"]["IO_DATASETS"])

    flavors = [flavor]
    if(flavor == "all"):
        flavors = ["io","control"]
    
    datasets = {}
    for flavor in flavors:
        datasets[flavor] = pd.read_pickle(dataPath/flavor/f'{dataName}_{flavor}.pkl.gz',
                                          compression='gzip')
        datasets[flavor]["flavor"] = flavor

    # concatenate 
    df = pd.concat(datasets.values(), ignore_index=True)

    if minActivities > 0:
        userActivityCount = df["userid"].value_counts()
        usersWithMinActivities = set(userActivityCount[userActivityCount >= minActivities].index)
        df = df[df["userid"].isin(usersWithMinActivities)]
    return df



def obtainIOBipartiteEdgesRetweets(df,minActivities=1):
    # only retweets
    df = df[df.is_retweet]
    if minActivities > 0:
        userActivityCount = df["userid"].value_counts()
        usersWithMinActivities = set(userActivityCount[userActivityCount >= minActivities].index)
        df = df[df["userid"].isin(usersWithMinActivities)]
    bipartiteEdges = df[["userid","retweet_tweetid"]].values
    return bipartiteEdges

def obtainIOBipartiteEdgesURLs(df,removeRetweets=True,minActivities=1):
    # row['userid'] url['expanded_url']
    if(removeRetweets):
        df = df[~df.is_retweet]
    # only keep rows with urls
    mask = df.urls.apply(lambda x: len(x) > 0 if x==x else False)
    df = df[mask]
    if minActivities > 0:
        userActivityCount = df["userid"].value_counts()
        usersWithMinActivities = set(userActivityCount[userActivityCount >= minActivities].index)
        df = df[df["userid"].isin(usersWithMinActivities)]
    bipartiteEdges = [(row["userid"],url) for _,row in df.iterrows() for url in row["urls"]]
    return bipartiteEdges

def obtainIOBipartiteEdgesHashtags(df,removeRetweets=True,minActivities=1):
    # "userid" and "hashtags" (list)
    if(removeRetweets):
        df = df[~df.is_retweet]
    # only keep rows with hashtags
    mask = df.hashtags.apply(lambda x: len(x) > 0 if x==x else False)
    df = df[mask]
    if minActivities > 0:
        userActivityCount = df["userid"].value_counts()
        usersWithMinActivities = set(userActivityCount[userActivityCount >= minActivities].index)
        df = df[df["userid"].isin(usersWithMinActivities)]
    bipartiteEdges = [(row["userid"],hashtag) for _,row in df.iterrows() for hashtag in row["hashtags"]]
    return bipartiteEdges




def filterRightNodes(bipartiteEdges, minDegree=1,minStrength=1,minActivities=1):
    # filter right nodes with at least minDegree
    if(minDegree>1):
        rightDegrees = Counter(bipartiteEdges[:,1])
        mask = np.array([rightDegrees[rightNode]>=minDegree for _,rightNode in bipartiteEdges])
        bipartiteEdges = bipartiteEdges[mask]
    # filter left nodes with at least minActivities
    if(minActivities>1):
        leftDegrees = Counter(bipartiteEdges[:,0])
        mask = np.array([leftDegrees[leftNode]>=minActivities for leftNode,_ in bipartiteEdges])
        bipartiteEdges = bipartiteEdges[mask]
    return bipartiteEdges
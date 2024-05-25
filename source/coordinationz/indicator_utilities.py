from . import config

from pathlib import Path
import pandas as pd

# ast evaluates strings that are python expressions
import ast

def obtainBipartiteEdgesRetweets(df,minActivities=1):
    # keep only tweet_type == "retweet"
    df = df[df["tweet_type"] == "retweet"]
    if minActivities > 0:
        userActivityCount = df["user_id"].value_counts()
        usersWithMinActivities = set(userActivityCount[userActivityCount >= minActivities].index)
        df = df[df["user_id"].isin(usersWithMinActivities)]
    bipartiteEdges = df[["user_id","linked_tweet"]].values
    return bipartiteEdges



def obtainBipartiteEdgesURLs(df,removeRetweets=True,removeQuotes=False,removeReplies=False,minActivities=1,):
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
    # only keep users with at least minActivities
    userActivityCount = users.value_counts()
    usersWithMinActivities = set(userActivityCount[userActivityCount >= minActivities].index)
    mask = users.isin(usersWithMinActivities)
    users = users[mask]
    urls = urls[mask]
    # create edges list users -> urls
    edges = [(user,url) for user,urlList in zip(users,urls) for url in urlList]
    return edges

def obtainBipartiteEdgesHashtags(df,removeRetweets=True,removeQuotes=False,removeReplies=False,minActivities=1):
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
    # only keep users with at least minActivities
    userActivityCount = users.value_counts()
    usersWithMinActivities = set(userActivityCount[userActivityCount >= minActivities].index)
    mask = users.isin(usersWithMinActivities)
    users = users[mask]
    hashtags = hashtags[mask]
    # create edges list users -> hashtags
    edges = [(user,hashtag) for user,hashtag_list in zip(users,hashtags) for hashtag in hashtag_list]
    return edges


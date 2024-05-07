from . import config

from pathlib import Path
import pandas as pd


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

def obtainEvaluationBipartiteEdgesRetweets(df):
    # keep only tweet_type == "retweet"
    df = df[df["tweet_type"] == "retweet"]
    bipartiteEdges = df[["screen_name","linked_tweet"]].values
    return bipartiteEdges


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



def obtainIOBipartiteEdgesRetweets(df):
    # only retweets
    df = df[df.is_retweet]
    bipartiteEdges = df[["userid","retweet_tweetid"]].values
    return bipartiteEdges



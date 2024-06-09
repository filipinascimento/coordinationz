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
    mask = df["text"].isna()
    df.loc[mask,"contentText"] = df["text"][mask]

typeToSuffix = {
    ("merged","all"),
    ("coretweet","all"),
    ("cohashtag","all"),
    ("courl","all"),
    ("coword","all"),
    ("merged","alltextsim"),
    ("textsimilarity","alltextsim"),
    ("merged","allusc"),
    ("usctextsimilarity","allusc"),
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




for networkType,suffix in typeToSuffix.items():
    networkPath = networksPath/f"{dataName}_{suffix}_{networkType}.xnet"
    g = xn.load(networkPath)

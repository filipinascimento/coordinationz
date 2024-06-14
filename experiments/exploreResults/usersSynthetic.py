#!/usr/bin/env python

from pathlib import Path
import coordinationz as cz
import coordinationz.preprocess_utilities as czpre
import pickle

dataNameNatSynth = "TA2_full_eval_NO_GT_nat+synth_2024-06-03"
dataNameNat = "TA2_full_eval_NO_GT_nat_2024-06-03"
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

preprocessPath = Path(config["paths"]["PREPROCESSED_DATASETS"])
dfNatSynth = czpre.loadPreprocessedData(dataNameNatSynth, config=config)
dfNat = czpre.loadPreprocessedData(dataNameNat, config=config)


tweetsInSynth = set(dfNatSynth["tweet_id"].values)
tweetsInNat = set(dfNat["tweet_id"].values)

usersInSynth = set(dfNatSynth["user_id"].values)
usersInNat = set(dfNat["user_id"].values)

onlyInSynth = usersInSynth - usersInNat

onlyTweetsInSynth = tweetsInSynth - tweetsInNat

print("Only in synthetic:",len(onlyInSynth))
print("Total in synthetic:",len(usersInSynth))

print("Only in synthetic tweets:",len(onlyTweetsInSynth))
print("Total in synthetic tweets:",len(tweetsInSynth))

usersWithNewTweetsInSynth = set(dfNatSynth[dfNatSynth["tweet_id"].isin(onlyTweetsInSynth)]["user_id"].values)

allUsers = set(dfNat["user_id"].values).union(set(dfNatSynth["user_id"].values))
user2class = {}
for user in allUsers:
    user2class[user] = "Natural"
    if user in usersWithNewTweetsInSynth:
        user2class[user] = "Synth Tweets"
    if user in onlyInSynth:
        user2class[user] = "Synth User"

# save pickle
onlyInSynthPath = preprocessPath/f"{dataNameNatSynth}_extraData.pkl"
with open(onlyInSynthPath, "wb") as f:
    pickle.dump({"Synthetic":user2class}, f)


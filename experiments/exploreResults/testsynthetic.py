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

dataName = "TA2_full_eval_NO_GT_nat+synth_2024-06-03"
networkType = "merged"
suffix = "all"
thresholdAttribute = "quantile"
threshold = 0.9995
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
networkPath = networksPath/f"{dataName}_{suffix}_{networkType}_thres_{thresholdAttribute}_{threshold}.xnet"

df = czpre.loadPreprocessedData(dataName, config=config)

g = xn.load(networkPath)

user_ids = g.vs["Label"]
potentialsyntheticUsersFile = "experiments/exploreResults/added_users.csv"
dfSyntheticUsers = pd.read_csv(potentialsyntheticUsersFile)
syntheticUserIDs = set(dfSyntheticUsers["user_id"].values)

synthetic = np.zeros(len(user_ids))
for userIndex, user_id in enumerate(tqdm(user_ids)):
    if(user_id in syntheticUserIDs):
        synthetic[userIndex] = 1
        continue


suspicious = np.zeros(len(user_ids))
# for userIndex, user_id in enumerate(tqdm(user_ids)):
#     dfUser = df[df["user_id"] == user_id]
#     dfUser = dfUser["text"].dropna()
#     # min len of the text
#     textSizes = dfUser.apply(len)
#     if(min(textSizes) < 10):
#         suspicious[userIndex] = 1
#         continue
    
g.vs["suspicious"] = suspicious
g.vs["synthetic"] = synthetic
xn.save(g, str(networkPath).replace(".xnet","_synth.xnet"))

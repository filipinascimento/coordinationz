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
suffix = "all"

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

tablesOutputPath = Path(config["paths"]["TABLES"]).resolve()

df = czpre.loadPreprocessedData(dataName, config=config)

typeToSuffix = {
    "merged":"all",
    "coretweet":"all",
    "cohashtag":"all",
    "courl":"all",
    "coword":"all",
}

for networkType,suffix in typeToSuffix.items():
    networkPath = networksPath/f"{dataName}_{suffix}_{networkType}.xnet"
    g = xn.load(networkPath)


    tables = czind.suspiciousTables(df,g,
                                    thresholdAttribute = "quantile",
                                    thresholds = [0.999])

    import csv

    for threshold,tableData in tables.items():
        dfEdges = tableData["edges"]
        dfFiltered = tableData["filtered"]
        dfFiltered.to_csv(tablesOutputPath/f"{dataName}_{suffix}_{networkType}_filtered_{threshold}.csv",index=False,quoting=csv.QUOTE_NONNUMERIC,escapechar="\\")
        dfEdges.to_csv(tablesOutputPath/f"{dataName}_{suffix}_{networkType}_edges_{threshold}.csv",index=False,quoting=csv.QUOTE_NONNUMERIC,escapechar="\\")



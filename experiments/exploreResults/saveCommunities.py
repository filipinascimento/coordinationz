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
import csv


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

df = czpre.loadPreprocessedData(dataName, config=config)


networksPath = Path(config["paths"]["NETWORKS"]).resolve()
tablesOutputPath = Path(config["paths"]["TABLES"]).resolve()
figuresOutputPath = Path(config["paths"]["FIGURES"]).resolve()


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


for networkType,suffix in tqdm(typeToSuffix):
        for threshold in [0.9999,0.99999]:
            thresholdAttribute = "quantile"
            thresholdedNetworkPath = networksPath/f"{dataName}_{suffix}_{networkType}_thres_{thresholdAttribute}_{threshold}.xnet"
            g = xn.load(thresholdedNetworkPath)

            tables = czind.suspiciousTables(df,g,
                                            thresholdAttribute = "quantile",
                                            thresholds = [0])


            tableData = tables["0"]
            dfEdges = tableData["edges"]
            dfFiltered = tableData["filtered"]
            dfFiltered.to_csv(tablesOutputPath/f"{dataName}_{suffix}_{networkType}_filtered_{threshold}.csv",index=False,quoting=csv.QUOTE_NONNUMERIC,escapechar="\\")
            dfEdges.to_csv(tablesOutputPath/f"{dataName}_{suffix}_{networkType}_edges_{threshold}.csv",index=False,quoting=csv.QUOTE_NONNUMERIC,escapechar="\\")


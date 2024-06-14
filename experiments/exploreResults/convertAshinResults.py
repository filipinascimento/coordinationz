import networkx
import xnetwork
import igraph as ig
import numpy as np
import pandas as pd
import coordinationz as cz
import coordinationz.experiment_utilities as czexp
import coordinationz.preprocess_utilities as czpre
import coordinationz.indicator_utilities as czind
import coordinationz.network as cznet
from scipy.stats import rankdata
from pathlib import Path

ashwinNetworkPath = "Outputs/Networks/text_similarity.gexf"

dataName = "TA2_full_eval_NO_GT_nat+synth_2024-06-03"
networkType = "usctextsimilarity"
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

df = czpre.loadPreprocessedData(dataName, config=config)
dfFiltered = czind.filterUsersByMinActivities(df,activityType="any", minUserActivities=10)
g = networkx.read_gexf(ashwinNetworkPath)

# convert to igraph
g = ig.Graph.from_networkx(g)

# rename "label"
del g.vs["label"]
g.vs["Label"] = g.vs["_nx_name"]
del g.vs["_nx_name"]

leftCount = len(dfFiltered["user_id"].unique())
pairsCount = leftCount*(leftCount-1)//2
rank = rankdata(g.es["weight"], method="max")
g.es["quantile"] = (pairsCount-len(rank)+rank)/pairsCount

networkPath = networksPath/f"{dataName}_{suffix}_{networkType}.xnet"
xnetwork.save(g,networkPath)


from pathlib import Path
import pandas as pd
import numpy as np
from tqdm.auto import tqdm
import coordinationz as cz
import xnetwork as xn
import coordinationz.experiment_utilities as czexp
import networkx as nx

def load_data(file):
    import json
    
    with open(file, 'r') as f:
        json_data = json.load(f)

    return json_data
    
    
if __name__ == "__main__": # Needed for parallel processing
    config = cz.config
    tqdm.pandas()
    # config = cz.load_config("<path to config>")

    # Load JSON data from file
    cohashtag_0="/N/project/INCAS/merge_network/cuba_bipartite_original_filter_0.json"
    cohashtag_1="/N/project/INCAS/merge_network/cuba_bipartite_original_filter_10.json"
    networksPath = "/N/project/INCAS/merge_network/cuba_cohashtag_0.xnet"
    
    bipartiteEdges = load_data(cohashtag_0)

    # creates a null model output from the bipartite graph
    nullModelOutput = cz.nullmodel.bipartiteNullModelSimilarity(
        bipartiteEdges,
        scoreType=["zscore","pvalue-quantized"], # pvalue-quantized, pvalue, or zscore, 
        pvaluesQuantized=[0.001,0.01,0.05,0.1,0.25,0.5],
        realizations=1000,
        idf="smoothlog", # None, "none", "linear", "smoothlinear", "log", "smoothlog"
        batchSize=100,
        workers=-1,
        minSimilarity = 0.5, # will only consider similarities above 0.5
    )

    g = cz.network.createNetworkFromNullModelOutput(
        nullModelOutput,
        pvalueThreshold=0.05, # only keep edges with pvalue < 0.05
    )

    xn.save(g, networksPath)


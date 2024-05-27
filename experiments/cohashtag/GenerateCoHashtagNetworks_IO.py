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
    # cohashtag_0="/N/project/INCAS/merge_network/cuba_bipartite_original_filter_0.json"
    
    realizations=10000
    pvaluesQuantized=[0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.25, 0.5]
    filter_activity=10
    
    print('***Starting null model***\n')
    print('Realizations :', realizations)
    print('pvaluesQuantized :', pvaluesQuantized)
    print('Activity filtered at :', filter_activity)
    print('\n\n')
    
    cohashtag_1=f"/N/project/INCAS/merge_network/cuba_bipartite_original_filter_{filter_activity}.json"
    networksPath = f"/N/project/INCAS/merge_network/cuba_cohashtag_{realizations}_{filter_activity}.xnet"
    
    bipartiteEdges = load_data(cohashtag_1)

    # creates a null model output from the bipartite graph
    nullModelOutput = cz.nullmodel.bipartiteNullModelSimilarity(
        bipartiteEdges,
        scoreType=["zscore","pvalue-quantized"], # pvalue-quantized, pvalue, or zscore, 
        pvaluesQuantized=pvaluesQuantized,
        realizations=realizations,
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


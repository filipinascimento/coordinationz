# Philippines_5M_sample_0908_2024-04-22

from pathlib import Path
import pandas as pd
import numpy as np
from tqdm.auto import tqdm
import coordinationz as cz
import xnetwork as xn
import coordinationz.experiment_utilities as czexp


if __name__ == "__main__": # Needed for parallel processing
    config = cz.config
    tqdm.pandas()
    # config = cz.load_config("<path to config>")

    networksPath = Path(config["paths"]["NETWORKS"]).resolve()
    networksPath.mkdir(parents=True, exist_ok=True)
    

    dataName = "cuba_082020_tweets"

    # Loads data from the evaluation datasets as pandas dataframes
    dfIO = czexp.loadIODataset(dataName, config=config, flavor="both", minActivities=10)

    # Create a bipartite graph from the retweet data
    bipartiteEdges = czexp.obtainIOBipartiteEdgesRetweets(dfIO)

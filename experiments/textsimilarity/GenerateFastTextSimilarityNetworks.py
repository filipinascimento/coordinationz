from pathlib import Path
import pickle
from tqdm.auto import tqdm
from pynndescent import NNDescent
import coordinationz as cz
import xnetwork as xn
import coordinationz.experiment_utilities as czexp
import numpy as np
import logging
import gc

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()

data_name = "cuba_082020_tweets"
df_type = "io"
column = "tweet_text"
buckets = 5000
min_activity = 10
seed = 9999

if __name__ == "__main__": # Needed for parallel processing
    rng = np.random.default_rng(seed)

    config = cz.config
    tqdm.pandas()

    networks_path = Path(config["paths"]["NETWORKS"]).resolve()
    networks_path.mkdir(parents=True, exist_ok=True)

    if df_type == "io":
        df = czexp.loadIODataset(data_name, config=config, flavor="all", minActivities=10)
    elif df_type == "eval":
        df = czexp.loadEvaluationDataset(data_name, config=config, minActivities=1)
    else:
        raise ValueError(f"Invalid dataframe type {df_type}")

    df = df[df["is_retweet"] == False]

    # filter for users and their tweets that are above an activity threshold
    df_min_active = df.groupby(["userid"])["tweetid"].nunique().to_frame("count").reset_index()
    df_min_active = df_min_active.loc[df_min_active["count"] >= min_activity]

    df = df.loc[df["userid"].isin(df_min_active["userid"])]

    with open(f"{data_name}_embedding.npy", "rb") as f:
        content, sentence_embeddings = pickle.load(f)

    # filter for tweets from active users
    unique = set(df[column])
    mask = [tweet in unique for tweet in content]
    content = [tweet for tweet, val in zip(content, mask) if val]
    sentence_embeddings = sentence_embeddings[mask]

    # get a random sample
    idx = np.arange(len(content))
    rng.shuffle(idx)
    centroids = sentence_embeddings[idx[:buckets]]

    # find the nearest centroid for each tweet
    index = NNDescent(centroids, n_neighbors=100, low_memory=False, diversify_prob=0.0)
    index.prepare()

    buckets, _ = index.query(sentence_embeddings, k=1, epsilon=0.3)

    table = {tweet: b for tweet, b in zip(content, buckets.squeeze(-1))}

    # convert to bipartite network
    df = df[["user_screen_name", column]]
    df[column] = df[column].map(table)
    bipartite_edges = df.to_numpy()

    del df, df_min_active, content, sentence_embeddings, unique, index, table
    gc.collect()

    # creates a null model output from the bipartite graph
    null_model_output = cz.nullmodel.bipartiteNullModelSimilarity(
        bipartite_edges,
        scoreType=["zscore", "pvalue-quantized"], # pvalue-quantized, pvalue, or zscore, 
        pvaluesQuantized=[0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.25, 0.5],
        realizations=10000,
        batchSize=100,
        idf="smoothlog",
        workers=-1,
        minSimilarity=0.5
    )

    g = cz.network.createNetworkFromNullModelOutput(
        null_model_output,
        pvalueThreshold=0.05
    )

    xn.save(g, networks_path/f"{data_name}_fast_text_similarity.xnet")

from pathlib import Path
import pickle
from tqdm.auto import tqdm
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize, StandardScaler
import coordinationz as cz
import xnetwork as xn
import coordinationz.experiment_utilities as czexp


data_name = "cuba_082020_tweets"
df_type = "io"
clusters = 1000

if __name__ == "__main__": # Needed for parallel processing
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

    with open(f"{data_name}_embedding.npy", "rb") as f:
        content, sentence_embeddings = pickle.load(f)

    # train k means from the sentence embeddings then assign a cluster to each tweet
    X = normalize(StandardScaler().fit_transform(sentence_embeddings))
    clusters = KMeans().fit_predict(X)
    table = {c: e for c, e in zip(content, clusters)}

    df = df[df["tweet_type"] == "tweet"]
    df = df[["screen_name", "text"]]
    df["text"] = df["text"].map(table)
    bipartite_edges = df.to_numpy()

    # creates a null model output from the bipartite graph
    null_model_output = cz.nullmodel.bipartiteNullModelSimilarity(
        bipartite_edges,
        scoreType=["zscore", "pvalue-quantized"], # pvalue-quantized, pvalue, or zscore, 
        pvaluesQuantized=[0.001, 0.01, 0.05, 0.1, 0.25, 0.5],
        realizations=1000,
        batchSize=10,
        workers=-1,
        minSimilarity = 0.5, # will only consider similarities above 0.5
    )

    g = cz.network.createNetworkFromNullModelOutput(
        null_model_output,
        pvalueThreshold=0.05, # only keep edges with pvalue < 0.05
    )

    xn.save(g, networks_path/f"{data_name}_text_similarity.xnet")

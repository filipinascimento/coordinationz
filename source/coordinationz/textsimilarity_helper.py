from pathlib import Path
import numpy as np
import emoji
import re

try:
    from sentence_transformers import SentenceTransformer
except:
    # raise new exception instructing user to install sentence_transformers
    message = "Please install the sentence_transformers package"
    message+= " by running the following command:\n\n"
    message+= "pip install sentence-transformers"
    raise ImportError(message)

try:
    from pynndescent import NNDescent
except:
    # raise new exception instructing user to install sentence_transformers
    message = "Please install the pynndescent package"
    message+= " by running the following command:\n\n"
    message+= "pip install pynndescent"
    raise ImportError(message)


def preprocess_tweet(tweet):
    # replace any special tokens
    # based on bertweet normalizer
    tweet = re.sub(r"\U00002026", "...", str(tweet))
    tweet = re.sub(r"\S*https?:\S*", "HTTPSURL", tweet)
    tweet = re.sub(r"\S*@\S*", "@USER", tweet)
    tweet = emoji.demojize(tweet)

    # remove all newlines
    lines = map(lambda line: line.strip(), tweet.splitlines())
    lines = (line if re.match(r"[.?!]$", line) else line + "." for line in lines if line)
    tweet = " ".join(lines)

    return tweet

def get_embeddings(df, data_name, column="text", model="paraphrase-multilingual-MiniLM-L12-v2", cache_path=None):
    if cache_path is not None: 
        cache_path = Path(cache_path).resolve()
        cache_path.mkdir(parents=True, exist_ok=True)

        cache_path = cache_path / f"{data_name}_{model}_{column}.npz"
        
        if cache_path.is_file():
            print("Loading sentence embeddings from cache...")
            cache = np.load(cache_path)

            embed_keys = cache["keys"].tolist()
            sentence_embeddings = cache["embeddings"]

            return embed_keys, sentence_embeddings


    tweets = df[column].unique().tolist()
    processed = list(map(preprocess_tweet, tweets))

    model = SentenceTransformer(model, device="cuda")
    sentence_embeddings = model.encode(processed, show_progress_bar=True)
    
    if cache_path is not None:
        np.savez_compressed(cache_path, keys=tweets, embeddings=sentence_embeddings)

    return tweets, sentence_embeddings

def filter_active(df, embed_keys, sentence_embeddings, min_activity=10, column="text"):
    df = df[df["tweet_type"] != "retweet"]

    # filter for users and their tweets that are above an activity threshold
    df_min_active = df.groupby(["user_id"])["tweet_id"].nunique().to_frame("count").reset_index()
    df_min_active = df_min_active[df_min_active["count"] >= min_activity]

    df = df[df["user_id"].isin(df_min_active["user_id"])]

    # filter for tweets from active users
    unique = set(df[column])
    mask = [tweet in unique for tweet in embed_keys]
    embed_keys = [tweet for tweet, val in zip(embed_keys, mask) if val]
    sentence_embeddings = sentence_embeddings[mask]

    return embed_keys, sentence_embeddings

def get_bipartite(df, embed_keys, sentence_embeddings, n_buckets=5000, column="text", seed=9999):
    # get a random sample
    idx = np.arange(len(embed_keys))
    rng = np.random.default_rng(seed)
    rng.shuffle(idx)
    centroids = sentence_embeddings[idx[:n_buckets]]

    # find the nearest centroid for each tweet
    index = NNDescent(centroids, n_neighbors=100, low_memory=False, diversify_prob=0.0, random_state=seed)
    index.prepare()

    buckets, _ = index.query(sentence_embeddings, k=1, epsilon=0.3)

    table = {tweet: b for tweet, b in zip(embed_keys, buckets.squeeze(-1))}

    # convert to bipartite network
    df = df[["user_id", column]].copy()
    df[column] = df[column].map(table)
    bipartite_edges = df.apply(tuple, axis=1).tolist()

    return bipartite_edges

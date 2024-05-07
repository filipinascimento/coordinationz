from tqdm.auto import tqdm
from sentence_transformers import SentenceTransformer
import pickle
import coordinationz as cz
import coordinationz.experiment_utilities as czexp

data_name = "cuba_082020_tweets"
df_type = "io"
sbert_model = "paraphrase-multilingual-MiniLM-L12-v2"

config = cz.config
tqdm.pandas()

if df_type == "io":
    df = czexp.loadIODataset(data_name, config=config, flavor="all", minActivities=10)
elif df_type == "eval":
    df = czexp.loadEvaluationDataset(data_name, config=config, minActivities=1)
else:
    raise ValueError(f"Invalid dataframe type {df_type}")

content = df["text"].unique()

model = SentenceTransformer(sbert_model, device="cuda")
sentence_embeddings = model.encode(content, show_progress_bar=True)

embed_map = (content.to_list(), sentence_embeddings)

with open(f"{data_name}_embedding.npy", "wb") as f:
    pickle.dump(embed_map, f)

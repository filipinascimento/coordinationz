import pandas as pd
import json

filepathMetadata = "Outputs/finalGaza/hamas_israel_challenge_merged_separated_wimagesv4_merged_component_metadata.json"

with open(filepathMetadata, 'r') as f:
    data = json.load(f)
    
filepath = "Outputs/finalGaza/hamas_israel_challenge_merged_separated_wimagesv4_merged_nodes_0.999995_processed.feather"

df = pd.read_feather(filepath)



# group by ComponentID
grouped = df.groupby("ComponentID")

# create a new dataframe with the first row of each group, keep all the columns
df_grouped = grouped.first().reset_index()

df_grouped = df_grouped.iloc[0:12].copy()

retweetTokens = []
tokens = []
for componentID in df_grouped["ComponentID"]:
    # get the metadata for this componentID
    metadata = data[str(componentID)]
    # get the retweet tokens
    retweetTokensImportance = ", ".join([entry[0] for entry in metadata["retweetTokensImportance"][:10]])
    tokensImportance = ", ".join([entry[0] for entry in metadata["tokensImportance"][:10]])
    retweetTokens.append(retweetTokensImportance)
    # get the tokens
    tokens.append(tokensImportance)
df_grouped["retweetTokens"] = retweetTokens
df_grouped["tokens"] = tokens

#create a latex table
df_grouped[['ComponentID', 'ComponentSize','ComponentID','ComponentTitle','tokens', 'retweetTokens']] \
    .to_latex("Outputs/finalGaza/hamas_israel_challenge_merged_separated_wimagesv4_merged_component_table.tex")


# %%
import pandas as pd
from pathlib import Path


filepaths = Path("Outputs/Tables/").glob("hamas_israel_challenge_merged_separated_wimagesv4_merged_nodes*.csv")
for filepath in filepaths:
    print(filepath)
    data = pd.read_csv(filepath, dtype={
        "CommunityLabel": str,
        "CommunityIndex": "Int64",
    })

    # %%
    communityDescriptors = [
        "Top URLs",
        "Top Hashtags",
        "Top Tokens",
        "Top Retweet Tokens",
        "Surprising URLs",
        "Surprising Hashtags",
        "Surprising Tokens",
        "Surprising Retweet Tokens",
    ]
        
    # only include users with degree non nan
    dataNonNan = data[data["courl_left_degree"].notna()].copy()

    # for users with degree == 0 , set: Top URLs Top Hashtags Top Tokens Top Retweet Tokens to NaN
    dataNonNan.loc[dataNonNan["degree"] == 0, communityDescriptors] = None


    # set a new column named Coordinated to True if degree > 0
    dataNonNan["Coordinated"] = dataNonNan["degree"] > 0

    # %%


    # save as feather
    newFilename = filepath.with_name(filepath.stem + "_final.feather")
    dataNonNan.to_feather(newFilename)


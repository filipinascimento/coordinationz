import json
import pandas as pd
import numpy as np

import coordinationz as cz
import json
from pathlib import Path

config = cz.config
# config = cz.load_config("<path to config>")

dataName = "sampled_20240226"

preprocessedDataPath = Path(config["paths"]["INCAS_PREPROCESSED_DATASETS"]).resolve()

preprocessedFilePath = preprocessedDataPath/f"{dataName}.csv"


df = pd.read_csv(preprocessedFilePath,dtype={"linked_tweet": str})
# save a version without retweets
allSets = {}
allSets["authorScreenNames"] = set(df["author_screenname"].values)
allSets["ids"] = set(df["id"].values)
allSets["linkedTweets"] = set(df["linked_tweet"].values)
# allSets["mentionedUsers"] = set([mentionedUser for mentionedUsers in df["mentionedUsers"] for mentionedUser in json.loads(mentionedUsers.replace("\'", "\""))])
df["mentionedUsers"].fillna("[]", inplace=True)
# from string formatted array, like "[1,2,3,4]" to array [1,2,3,4]
df["mentionedUsers"] = df["mentionedUsers"].apply(lambda x: json.loads(x.replace("\'", "\"")))
# df['screen_name'] = df.screen_name.apply(lambda x: x.split('/')[-3])

allSets["mentionedUsers"] = set([mentionedUser for mentionedUsers in df["mentionedUsers"] for mentionedUser in mentionedUsers])
# allSets["twitterUserIds"] = set(df["mediaTypeAttributes.twitterData.twitterUserId"].values)
allSets["name"] = set(df["name"].values)
allSets["screenNames"] = set(df["screen_name"].values)
allSets["tweetIDs"] = set(df["tweet_id"].values)

# print counts of df["tweet_type"] values
print(df["tweet_type"].value_counts())

# Jaccard similarity
import numpy as np
jaccardMatrix = np.zeros((len(allSets), len(allSets)))
for i, (set1Name,set1) in enumerate(allSets.items()):
    for j, (set2Name,set2) in enumerate(allSets.items()):
        jaccardMatrix[i][j] = len(set1.intersection(set2))/len(set1.union(set2))
        if jaccardMatrix[i][j] > 0 and i!=j:
            print(set1Name,set2Name,jaccardMatrix[i][j])

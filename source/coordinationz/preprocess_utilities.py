

import ast
import pandas as pd
from tqdm.auto import tqdm
import json
import os
from . import config
from pathlib import Path

dropped_columns = [
    "author",
    "dataTags",
    "extraAttributes",
    "segments",
    "geolocation",
    "segments",
    "translatedTitle",
]

unnecessary_so_drop = ["title"]

dtypes = {
    "twitterData.engagementParentId": "string",
}

rename_cols = {
    "contentText": "text",
    "timePublished": "created_at",
    "language": "lang",
    "twitterData.engagementParentId": "linked_tweet",
    "twitterData.engagementType": "tweet_type",
    "twitterData.followerCount": "follower_count",
    "twitterData.followingCount": "following_count",
    "twitterData.likeCount": "like_count",
    "twitterData.retweetCount": "retweet_count",
    "twitterData.tweetId": "tweet_id",
    "twitterData.mentionedUsers": "mentionedUsers",
    "twitterData.twitterAuthorScreenname": "author_screenname",
    # 'twitterData.twitterUserId':'twitter_user_id',
    "embeddedUrls": "urls",
}

mediaTypeAttributesList = [
    "twitterData.engagementParentId",
    "twitterData.engagementType",
    "twitterData.followerCount",
    "twitterData.followingCount",
    "twitterData.likeCount",
    "twitterData.mentionedUsers",
    "twitterData.retweetCount",
    "twitterData.tweetId",
    "twitterData.twitterAuthorScreenname",
    # 'twitterData.twitterUserId'
]



def loadPreprocessedData(dataName,config=config,**kwargs):

    dtype={
        "tweet_id": str,
        "user_id": str,
        "tweet_type": str,
        "text": str,
        # "creation_date": str,
        "linked_tweet": str,
        "urls": str,
        "hashtags": str,
        "mentioned_users": str,
    }
    # if dtype in kwargs, merge with existing dtype
    if("dtype" in kwargs):
        dtype.update(kwargs["dtype"])
        del kwargs["dtype"]

    preprocessedFilePath = Path(config["paths"]["PREPROCESSED_DATASETS"])/(dataName+".csv")

    df = pd.read_csv(preprocessedFilePath, dtype=dtype, **kwargs)

    df["hashtags"] = df["hashtags"].apply(ast.literal_eval)
    df["mentioned_users"] = df["mentioned_users"].apply(ast.literal_eval)
    df["urls"] = df["urls"].apply(ast.literal_eval)

    return df



def generateReport(df):
    numTweets = len(df)
    numUsers = len(df["user_id"].unique())
    report = []
    report.append(("Number of tweets:", len(df), numTweets))
    report.append(("Number of users:", len(df["user_id"].unique()), numUsers))
    report.append(("Number of retweets:", len(df[df["tweet_type"] == "retweet"]), numTweets))
    report.append(("Number of quotes:", len(df[df["tweet_type"] == "quote"]), numTweets))
    report.append(("Number of replies:", len(df[df["tweet_type"] == "reply"]), numTweets))
    report.append(("Number of hashtags:", len(df["hashtags"].explode().unique()), None))
    report.append(("Number of mentioned users:", len(df["mentioned_users"].explode().unique()), None))
    report.append(("Number of urls:", len(df["urls"].explode().unique()), None))
    report.append(("Number of tweets with urls:", len(df[df["urls"].apply(len) > 0]), numTweets))
    report.append(("Number of tweets with hashtags:", len(df[df["hashtags"].apply(len) > 0]), numTweets))
    report.append(("Number of tweets with mentioned users:", len(df[df["mentioned_users"].apply(len) > 0]), numTweets))
    report.append(("Number of tweets with linked tweets:", len(df[df["linked_tweet"].notna()]), numTweets))
    report.append(("", "", None))
    # users with at least 10 tweets
    userActivityCount = df["user_id"].value_counts()
    usersWithMinActivities = userActivityCount[userActivityCount >= 10].index
    report.append(("Number of users with at least 10 tweets:", len(usersWithMinActivities), numUsers))
    # users with at least 10 retweets
    userActivityCount = df[df["tweet_type"] == "retweet"]["user_id"].value_counts()
    usersWithMinActivities = userActivityCount[userActivityCount >= 10].index
    report.append(("Number of users with at least 10 retweets:", len(usersWithMinActivities), numUsers))
    # users with at least 10 unique hashtags
    userHashtags = df.explode("hashtags").dropna(subset=["hashtags"])
    userActivityCount = userHashtags["user_id"].value_counts()
    usersWithMinActivities = userActivityCount[userActivityCount >= 10].index
    report.append(("Number of users with at least 10 unique hashtags:", len(usersWithMinActivities), numUsers))
    # users with at least 10 unique mentioned users
    userMentionedUsers = df.explode("mentioned_users").dropna(subset=["mentioned_users"])
    userActivityCount = userMentionedUsers["user_id"].value_counts()
    usersWithMinActivities = userActivityCount[userActivityCount >= 10].index
    report.append(("Number of users with at least 10 unique mentioned users:", len(usersWithMinActivities), numUsers))
    # users with at least 10 unique urls
    userUrls = df.explode("urls").dropna(subset=["urls"])
    userActivityCount = userUrls["user_id"].value_counts()
    usersWithMinActivities = userActivityCount[userActivityCount >= 10].index
    report.append(("Number of users with at least 10 unique urls:", len(usersWithMinActivities), numUsers))
    report.append(("", "", None))


    allSets = {}
    allSets["user_id"] = set(df["user_id"].values)
    allSets["tweet_id"] = set(df["tweet_id"].values)
    allSets["linked_tweet"] = set(df["linked_tweet"].values)
    allSets["mentioned_users"] = set(df["mentioned_users"].explode().unique())

    # Jaccard similarity
    import numpy as np
    jaccardMatrix = np.zeros((len(allSets), len(allSets)))
    for i, (set1Name,set1) in enumerate(allSets.items()):
        for j, (set2Name,set2) in enumerate(allSets.items()):
            jaccardMatrix[i][j] = len(set1.intersection(set2))/len(set1.union(set2))
            if jaccardMatrix[i][j] > 0 and i<j:
                title = f"Jaccard similarity between {set1Name} and {set2Name}:"
                report.append((title, f"{jaccardMatrix[i][j]:.2f}", None))

    reportOutput = [f"{k} {v} ({v/total*100:.2f}%)"
                    if total is not None
                    else f"{k} {v}"
                    for k, v, total in report]
    return "\n".join(reportOutput)



def preprocessINCASData(inputFilePath, preprocessedFilePath):
    firstTime = True
    progressBar = tqdm(total=os.path.getsize(inputFilePath),desc="Reading jsonl file")
    with open(inputFilePath,"rt") as f:
        while True:
            entriesBuffer = []
            for index,line in enumerate(f):
                # update in terms of bytes
                progressBar.update(len(line))
                try:
                    entriesBuffer.append(json.loads(line)) #4990446
                except:
                    print("Error in line: ", index, "[", line, "]")
                if(len(entriesBuffer) >= 10000):
                    break
            if(len(entriesBuffer) == 0):
                break

            df = pd.DataFrame(entriesBuffer)

            df = df.drop(columns=dropped_columns)
            df = df.drop(columns=unnecessary_so_drop)

            # get twitter only
            df = df[df['mediaType'] == 'Twitter']
            if(len(df) == 0):
                continue
            df = df.drop(columns = ['mediaType'])

            mediaTypeAttributes = pd.json_normalize(df['mediaTypeAttributes'])
            mediaTypeAttributes = mediaTypeAttributes[['twitterData.engagementParentId',
                'twitterData.engagementType', 'twitterData.followerCount',
                'twitterData.followingCount', 'twitterData.likeCount',
                'twitterData.mentionedUsers', 'twitterData.retweetCount',
                'twitterData.tweetId', 'twitterData.twitterAuthorScreenname',
                # 'twitterData.twitterUserId'
                ]]

            df = df.reset_index(drop=True)
            mediaTypeAttributes = mediaTypeAttributes.reset_index(drop=True)
            df = pd.concat([df, mediaTypeAttributes], axis=1)
            df = df.drop(columns=['mediaTypeAttributes'])

            # rename
            df = df.rename(columns=rename_cols)

            # created_at
            df['created_at'] = pd.to_datetime(df['created_at'], unit='ms')

            
            # http://twitter.com/Eye_On_Gaza/statuses/1697413595796328457
            # if url is formatted as url (check)
            # use screen_name based on the url
            try:
                df['screen_name'] = df.url.apply(lambda x: x.split('/')[-3])
                df = df.drop(columns=['url'])
            except:
                pass

            df = df.sort_index(axis=1)
            df['linked_tweet'] = df.linked_tweet.astype(str)
            
            # rename all columns to include suffix data_
            df = df.add_prefix("data_")

            # Keys:
            # 'data_annotations', 'data_author_screenname', 'data_created_at',
            # 'data_follower_count', 'data_following_count', 'data_id',
            # 'data_imageUrls', 'data_lang', 'data_like_count', 'data_linked_tweet',
            # 'data_mentionedUsers', 'data_name', 'data_retweet_count',
            # 'data_screen_name', 'data_text', 'data_translatedContentText',
            # 'data_tweet_id', 'data_tweet_type', 'data_urls'

            # We need:
            # tweet_id,
            # user_id,
            # tweet_type,
            # text,
            # creation_date,
            # linked_tweet,
            # urls,
            # hashtags, 
            # mentioned_users

            remapAttributes = {
                "tweet_id": "tweet_id", # string
                # "screen_name": "user_id", # string
                "author_screenname": "user_id", # string
                "tweet_type": "tweet_type", # string
                "text": "text", # string
                "created_at": "creation_date", # datetime
                "linked_tweet": "linked_tweet", #retweet/quote/etc/ #string
                "linked_tweet_user_id": "linked_tweet_user_id", #string
                "urls": "urls", #list of strings
                "mentionedUsers": "mentioned_users", #list of strings
            }

            # add suffix data_ to keys
            remapAttributes = {f"data_{k}": f"{v}" for k, v in remapAttributes.items()}
            df = df.rename(columns=remapAttributes)
            
            # calculate 
            hashtags = df["text"].str.findall(r"#\w+")
            df["hashtags"] = hashtags
            # replace nan with empty list
            df["hashtags"] = df["hashtags"].map(lambda x: x if x == x and x is not None else [])
            # apply that for mentioned users and urls
            df["mentioned_users"] = df["mentioned_users"].map(lambda x: x if x == x and x is not None else [])
            df["urls"] = df["urls"].map(lambda x: x if x == x and x is not None else [])
            
            if(firstTime):
                df.to_csv(preprocessedFilePath, index=False)
                firstTime = False
            else:
                df.to_csv(preprocessedFilePath, mode='a', header=False, index=False)
    progressBar.close()
    
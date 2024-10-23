

import ast
import pandas as pd
from tqdm.auto import tqdm
import json
import os
from . import config
from pathlib import Path
import csv
import unalix

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
        "linked_tweet_user_id": str,
        "urls": str,
        "hashtags": str,
        "mentioned_users": str,
        
        "category": str,
    }

    # if dtype in kwargs, merge with existing dtype
    if("dtype" in kwargs):
        dtype.update(kwargs["dtype"])
        del kwargs["dtype"]

    preprocessedFilePath = Path(config["paths"]["PREPROCESSED_DATASETS"])/(dataName+".csv")

    df = pd.read_csv(preprocessedFilePath, dtype=dtype,
                     escapechar='\\',
                     encoding='utf-8',
                     quoting=csv.QUOTE_NONNUMERIC,
                     **kwargs)

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
            if("text" in df.columns):
                df = df.dropna(subset=["text"])
                
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
            
            # from the Text get the first mention when it is a retweet
            # eg RT @ICJPalestine -> ICJPalestine
            # mask using tweet_type == retweet
            df["linked_tweet_user_id"] = ""
            retweetMask = (df.tweet_type == "retweet")
            df.loc[retweetMask, "linked_tweet_user_id"] = df[retweetMask].text.str.extract(r'RT @(\w+)', expand=False)

            # calculate 
            hashtags = df["text"].str.lower().str.findall(r"#\w+")
            df["hashtags"] = hashtags
            # replace nan with empty list
            df["hashtags"] = df["hashtags"].map(lambda x: x if x == x and x is not None else [])
            # apply that for mentioned users and urls
            df["mentioned_users"] = df["mentioned_users"].map(lambda x: x if x == x and x is not None else [])
            df["urls"] = df["urls"].map(lambda x: [unalix.clear_url(url) for url in x] if x == x and x is not None else [])
            
            if(firstTime):
                df.to_csv(preprocessedFilePath, index=False,escapechar='\\', encoding='utf-8',quoting=csv.QUOTE_NONNUMERIC)
                firstTime = False
            else:
                df.to_csv(preprocessedFilePath, mode='a', header=False, index=False,escapechar='\\', encoding='utf-8',quoting=csv.QUOTE_NONNUMERIC)
    progressBar.close()
    

def preprocessIOData(dataName,dataPath, preprocessedFilePath, flavors = ["io","control"],keepAllColumns=False):

    flavors = ["io","control"]
    
    print("Reading Data...")
    datasets = {}
    for flavor in flavors:
        print(f"Reading flavor {flavor}...")
        expectedLocation = dataPath/flavor/f'{dataName}_{flavor}.pkl.gz'
        if(not expectedLocation.exists()):
            print(f"File {expectedLocation} does not exist.")
            expectedLocation = dataPath/flavor/f'{dataName}.pkl.gz'
        if(not expectedLocation.exists()):
            print(f"File {expectedLocation} does not exist. Skipping flavor {flavor}.")
            continue
        datasets[flavor] = pd.read_pickle(expectedLocation,
                                          compression='gzip')
        datasets[flavor]["category"] = flavor

    print(f"Combining datasets...")
    # concatenate 
    df = pd.concat(datasets.values(), ignore_index=True)
    

    print(f"Normalizing attributes...")
    df = df.add_prefix("data_")

    remapAttributes = {
        "tweetid": "tweet_id", # string
        # "screen_name": "user_id", # string
        "userid": "user_id", # string
        "tweet_type": "tweet_type", # string
        "tweet_text": "text", # string
        "created_at": "created_at", # datetime
        "linked_tweet": "linked_tweet", #retweet/quote/etc/ #string
        "linked_tweet_user_id": "linked_tweet_user_id", #string
        "urls": "urls", #list of strings
        "hashtags": "hashtags", #list of strings
        "mentions": "mentioned_users", #list of strings
        "category": "category"
    }


    # merge mentions and user_mentions
    df["data_mentions"] = df["data_mentions"].combine_first(df["data_user_mentions"])
    # created_at data format: Fri Jul 31 23:56:25 +0000 2020
    df["data_created_at"] = pd.to_datetime(df["data_created_at"], format='%a %b %d %H:%M:%S %z %Y')
    # same for data_tweet_time but that format: 2014-07-17 00:36
    df["data_tweet_time"] = pd.to_datetime(df["data_tweet_time"], format='%Y-%m-%d %H:%M')
    # merge the data_creation_date and data_tweet_time
    df["data_created_at"] = df["data_created_at"].combine_first(df["data_tweet_time"])

    # normalize hashtags (string mixed with lists)
    # use evaluate literal if it is a string
    # set nan to ""
    df["data_hashtags"] = df["data_hashtags"].fillna("[]")
    df["data_hashtags"] = df["data_hashtags"].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)

    df["data_urls"] = df["data_urls"].fillna("[]")
    df["data_urls"] = df["data_urls"].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)

    df["data_mentions"] = df["data_mentions"].fillna("[]")
    df["data_mentions"] = df["data_mentions"].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
    
    # replace None in quoted_tweet_tweetid, in_reply_to_tweetid, and retweet_tweetid with NaN
    df["data_in_reply_to_tweetid"] = df["data_in_reply_to_tweetid"].replace({None: pd.NA})
    df["data_quoted_tweet_tweetid"] = df["data_quoted_tweet_tweetid"].replace({None: pd.NA})
    df["data_retweet_tweetid"] = df["data_retweet_tweetid"].replace({None: pd.NA})
    df["data_in_reply_to_tweetid"] = df["data_in_reply_to_tweetid"].replace({"None": pd.NA})
    df["data_quoted_tweet_tweetid"] = df["data_quoted_tweet_tweetid"].replace({"None": pd.NA})
    df["data_retweet_tweetid"] = df["data_retweet_tweetid"].replace({"None": pd.NA})

    # repeat for userid
    df["data_in_reply_to_userid"] = df["data_in_reply_to_userid"].replace({None: pd.NA})
    df["data_retweet_userid"] = df["data_retweet_userid"].replace({None: pd.NA})
    df["data_in_reply_to_userid"] = df["data_in_reply_to_userid"].replace({"None": pd.NA})
    df["data_retweet_userid"] = df["data_retweet_userid"].replace({"None": pd.NA})

    

    df["data_tweet_type"] = "tweet"
    # tweet_type = "retweet" when is_retweet 
    # tweet_type = "quote" when quoted_tweet_tweetid is not NaN
    # tweet_type = "reply" when in_reply_to_tweetid is not NaN
    df.loc[df["data_is_retweet"], "data_tweet_type"] = "retweet"
    df.loc[~df["data_quoted_tweet_tweetid"].isna(), "data_tweet_type"] = "quote"
    df.loc[~df["data_in_reply_to_tweetid"].isna(), "data_tweet_type"] = "reply"

    # merge in_reply_to_tweetid, quoted_tweet_tweetid and data_retweet_tweetid into linked_tweet
    
    df["data_linked_tweet"] = df["data_in_reply_to_tweetid"].combine_first(df["data_quoted_tweet_tweetid"]).combine_first(df["data_retweet_tweetid"])
    df["data_linked_tweet_user_id"] = df["data_in_reply_to_userid"].combine_first(df["data_retweet_userid"])

    # add suffix data_ to keys
    remapAttributes = {f"data_{k}": f"{v}" for k, v in remapAttributes.items()}
    df = df.rename(columns=remapAttributes)
        # delete redundant columns

    if(keepAllColumns):
        print(f"Removing redundant columns...")
        df = df.drop(columns=[
            "data_in_reply_to_tweetid",
            "data_quoted_tweet_tweetid",
            "data_retweet_tweetid",
            "data_in_reply_to_userid",
            "data_retweet_userid",
            "data_tweet_time",
            "data_is_retweet",
            "data_tweet_time",
            "data_user_mentions"
        ])
    else:
        print(f"Removing columns not needed for analysis...")
        # drop all columns starting with data_
        df = df.filter(regex='^(?!data_)')

    print(f"Saving to CSV...")
    df.to_csv(preprocessedFilePath, index=False,
                escapechar='\\',
                encoding='utf-8',
                quoting=csv.QUOTE_NONNUMERIC)


'''

'''
import pandas as pd
import os
from tqdm import tqdm
import coordinationz as cz
import json
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


if __name__ == "__main__": # Needed for parallel processing
    config = cz.config
    tqdm.pandas()

    # config = cz.load_config("<path to config>")

    incasDataPath = Path(config["paths"]["INCAS_DATASETS"]).resolve()
    preprocessedDataPath = Path(config["paths"]["INCAS_PREPROCESSED_DATASETS"]).resolve()
    
    preprocessedDataPath.mkdir(parents=True, exist_ok=True)

    dataName = "sampled_20240226"

    inputFilePath = incasDataPath/f"{dataName}.jsonl"
    preprocessedFilePath = preprocessedDataPath/f"{dataName}.csv"

    firstTime = True
    with open(inputFilePath,"rt") as f:
        while True:
            entriesBuffer = []
            for index,line in enumerate(tqdm(f)):
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
            if(df.empty):
                continue
            df = df.drop(columns = ['mediaType'])

            mediaTypeAttributes = pd.json_normalize(df['mediaTypeAttributes'])
            mediaTypeAttributes = mediaTypeAttributes[mediaTypeAttributesList]

            df = df.reset_index(drop=True)
            mediaTypeAttributes = mediaTypeAttributes.reset_index(drop=True)
            df = pd.concat([df, mediaTypeAttributes], axis=1)
            df = df.drop(columns=['mediaTypeAttributes'])

            # rename
            df = df.rename(columns=rename_cols)

            # created_at
            df['created_at'] = pd.to_datetime(df['created_at'], unit='ms')

            # http://twitter.com/Eye_On_Gaza/statuses/1697413595796328457
            df['screen_name'] = df.url.apply(lambda x: x.split('/')[-3])
            # df["screen_name"] = df["url"]
            df = df.drop(columns=['url'])
            # fill na of mentionedUsers with []
            df = df.sort_index(axis=1)
            #
            # df['linked_tweet'] = df.linked_tweet.astype(str)
            if(firstTime):
                df.to_csv(preprocessedFilePath, index=False)
                firstTime = False
            else:
                df.to_csv(preprocessedFilePath, mode='a', header=False, index=False)
            

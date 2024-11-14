import igraph as ig
import numpy as np
import math
from collections import Counter
from tqdm.auto import tqdm
import coordinationz.indicator_utilities as czind

def getNetworksWithCommunities(gThresholded,weightAttribute="weight"):
    gThresholded.vs["CommunityIndex"] = gThresholded.community_leiden(objective_function = "modularity",
                                                                    weights = weightAttribute,
                                                                    ).membership
    gThresholded.vs["CommunityLabel"] = [f"{i}" for i in gThresholded.vs["CommunityIndex"]]
    allCommunities = set(gThresholded.vs["CommunityIndex"])
    community2Size = {}
    for c in allCommunities:
        community2Size[c] = len(gThresholded.vs.select(CommunityIndex_eq=c))
    community2EdgesCount = {}
    community2EdgesDensity = {}
    # community2EdgesDensityAlt = {}
    community2EdgesAvgWeight = {}
    for c in allCommunities:
        edgesInCommunity = gThresholded.es.select(_source_in=gThresholded.vs.select(CommunityIndex_eq=c),_target_in=gThresholded.vs.select(CommunityIndex_eq=c))
        community2EdgesCount[c] = len(gThresholded.es.select(_source_in=gThresholded.vs.select(CommunityIndex_eq=c)))
        if(community2Size[c]>1):
            community2EdgesDensity[c] = community2EdgesCount[c]/(community2Size[c]*(community2Size[c]-1))
        else:
            community2EdgesDensity[c] = 0
        community2EdgesAvgWeight[c] = np.mean(edgesInCommunity["weight"])
    gThresholded.vs["CommunitySize"] = [community2Size[c] for c in gThresholded.vs["CommunityIndex"]]
    gThresholded.vs["CommunityEdgesCount"] = [community2EdgesCount[c] for c in gThresholded.vs["CommunityIndex"]]
    gThresholded.vs["CommunityEdgesDensity"] = [community2EdgesDensity[c] for c in gThresholded.vs["CommunityIndex"]]
    gThresholded.vs["CommunityEdgesAvg_"+weightAttribute] = [community2EdgesAvgWeight[c] for c in gThresholded.vs["CommunityIndex"]]
    if("Type" in gThresholded.es.attributes()):
        # get most common edge Type and associate it to the vertex
        for vertex in gThresholded.vs:
            # get all edges of the vertex
            edges = list(gThresholded.es.select(_source=vertex.index)) + list(gThresholded.es.select(_target=vertex.index))
            # get the type of the edges
            types = [edge["Type"] for edge in edges]
            # get the most common type
            if(types):
                mostCommonType = Counter(types).most_common(1)[0][0]
            else:
                mostCommonType = "None"
            # associate the most common type to the vertex
            vertex["Type"] = mostCommonType

        
    return gThresholded




def filterNgramParts(tokens,maxTokens=6):
    # suppose it receives a list of tokens that can be ngrams
    # ngrams are separated by space
    # from the first to the last, if the ngram parts are repeated, remove them
    # will return at most 6 valid tokens
    prohibitedTokens = set()
    filteredTokens = []
    for token in tokens:
        if token in prohibitedTokens:
            continue
        tokenParts = token.split()
        for i in range(0,len(tokenParts)):
            prohibitedTokens.add(tokenParts[i])
        # also add all the possible ngrams
        for i in range(2,len(tokenParts)+1):
            for j in range(0,len(tokenParts)-i+1):
                prohibitedTokens.add(" ".join(tokenParts[j:j+i]))
        filteredTokens.append(token)
        # print(prohibitedTokens)
        if(len(filteredTokens) == maxTokens):
            break
    return filteredTokens


def getTokens(tweetID,text,tweetID2Tokens):
    if(tweetID in tweetID2Tokens):
        return tweetID2Tokens[tweetID]
    tokens = czind.tokenizeTweet(text,ngram_range=(1,3))
    tweetID2Tokens[tweetID] = tokens
    return tokens

def labelCommunities(df, g, tweetID2TokensCache = {}):
    df = df.copy()
    df["contentText"] = df["text"]
    if("data_translatedContentText" in df and not df["data_translatedContentText"].isna().all()):
        df["contentText"] = df["data_translatedContentText"]
        # for the nans, use the original text
        mask = df["contentText"].isna()
        df.loc[mask,"contentText"] = df["text"][mask]
    
    gThresholded = g
    # all users in gThresholded (Label)
    allUsers = set(gThresholded.vs["Label"])

    # onlyEntries in set
    dfFiltered = df[df["user_id"].isin(allUsers)]
    dfRetweets = dfFiltered[dfFiltered["tweet_type"]=="retweet"]
    dfOriginal = dfFiltered[dfFiltered["tweet_type"]!="retweet"]
    dfInNetworkURLs = dfOriginal.dropna(subset=["urls"])
    dfInNetworkHashtags = dfOriginal.dropna(subset=["hashtags"])
    dfInNetworkRetweets = dfRetweets.dropna(subset=["linked_tweet"])
    # for tokens use czind.tokenizeTweet(string)

    dfInNetworkTokens = dfOriginal.dropna(subset=["contentText"])
    # use translated 
    # apply getTokens to text, tweet_id
    dfInNetworkTokens["tokens"] = dfInNetworkTokens[["tweet_id","contentText"]].progress_apply(lambda x: getTokens(*x,tweetID2TokensCache),axis=1)
    dfInNetworkRetweetTokens = dfInNetworkRetweets.dropna(subset=["contentText"])
    if(dfInNetworkRetweetTokens.empty):
        dfInNetworkRetweetTokens["tokens"] = []
    else:
        dfInNetworkRetweetTokens["tokens"] = dfInNetworkRetweetTokens[["tweet_id","contentText"]].progress_apply(lambda x: getTokens(*x,tweetID2TokensCache),axis=1)

    user2urlCounter = {}
    user2hashtagsCounter = {}
    user2retweetsCounter = {}
    user2tokensCounter = {}
    user2RetweetTokensCounter = {}

    hashtag2TotalCounts = Counter()
    url2TotalCounts = Counter()
    retweet2TotalCounts = Counter()
    token2TotalCounts = Counter()
    retweetToken2TotalCounts = Counter()

    hashtagSumCount = 0
    urlSumCount = 0
    retweetSumCount = 0
    tokenSumCount = 0
    retweetTokenSumCount = 0

    penaltyFactor = 4
    transform = lambda x: x
    # transform = lambda x: np.log(x+1)

    for user,urls in tqdm(dfInNetworkURLs[["user_id","urls"]].values):
        urlsCounter = Counter(urls)
        # normalize by size of urls
        # urlsCounter/=len(urls)
        for url in urlsCounter:
            urlsCounter[url] /= len(urls)
        if user in user2urlCounter:
            user2urlCounter[user].update(urlsCounter)
        else:
            user2urlCounter[user] = Counter(urlsCounter)
        url2TotalCounts.update(urlsCounter)
        urlSumCount += sum(urlsCounter.values())

    for user,hashtags in tqdm(dfInNetworkHashtags[["user_id","hashtags"]].values):
        hashtagsCounter = Counter(hashtags)
        # normalize by size of hashtags
        # hashtagsCounter/=len(hashtags)
        for hashtag in hashtagsCounter:
            hashtagsCounter[hashtag] /= len(hashtags)
        if user in user2hashtagsCounter:
            user2hashtagsCounter[user].update(hashtagsCounter)
        else:
            user2hashtagsCounter[user] = Counter(hashtagsCounter)
        hashtag2TotalCounts.update(hashtagsCounter)
        hashtagSumCount += sum(hashtagsCounter.values())

    for user,linked_tweet in tqdm(dfInNetworkRetweets[["user_id","linked_tweet"]].values):
        retweetsCounter = Counter(linked_tweet)
        # normalize by size of retweets
        # retweetsCounter/=len(linked_tweet)
        for retweet in retweetsCounter:
            retweetsCounter[retweet] /= len(linked_tweet)
        if user in user2retweetsCounter:
            user2retweetsCounter[user].update(retweetsCounter)
        else:
            user2retweetsCounter[user] = Counter(retweetsCounter)
        retweet2TotalCounts.update(retweetsCounter)
        retweetSumCount += sum(retweetsCounter.values())

    for user,tokens in tqdm(dfInNetworkTokens[["user_id","tokens"]].values):
        tokensCounter = Counter(tokens)
        # normalize by size of tokens
        # tokensCounter/=len(tokens)
        for token in tokensCounter:
            tokensCounter[token] /= len(tokens)
        if user in user2tokensCounter:
            user2tokensCounter[user].update(tokensCounter)
        else:
            user2tokensCounter[user] = Counter(tokensCounter)
        token2TotalCounts.update(tokensCounter)
        tokenSumCount += sum(tokensCounter.values())

    for user,tokens in tqdm(dfInNetworkRetweetTokens[["user_id","tokens"]].values):
        tokensCounter = Counter(tokens)
        # normalize by size of tokens
        # tokensCounter/=len(tokens)
        for token in tokensCounter:
            tokensCounter[token] /= len(tokens)
        if user in user2RetweetTokensCounter:
            user2RetweetTokensCounter[user].update(tokensCounter)
        else:
            user2RetweetTokensCounter[user] = Counter(tokensCounter)
        retweetToken2TotalCounts.update(tokensCounter)
        retweetTokenSumCount += sum(tokensCounter.values())

    community2urlCounter = {}
    community2hashtagsCounter = {}
    community2retweetsCounter = {}
    community2tokensCounter = {}
    community2RetweetTokensCounter = {}

    community2urlSumTotal = {}
    community2hashtagsSumTotal = {}
    community2retweetsSumTotal = {}
    community2tokensSumTotal = {}
    community2RetweetTokensSumTotal = {}

    for community in tqdm(set(gThresholded.vs["CommunityIndex"])):
        community2urlCounter[community] = Counter()
        community2hashtagsCounter[community] = Counter()
        community2retweetsCounter[community] = Counter()
        community2tokensCounter[community] = Counter()
        community2RetweetTokensCounter[community] = Counter()

        community2urlSumTotal[community] = 0
        community2hashtagsSumTotal[community] = 0
        community2retweetsSumTotal[community] = 0
        community2tokensSumTotal[community] = 0
        community2RetweetTokensSumTotal[community] = 0

    labels = gThresholded.vs["Label"]
    communities = gThresholded.vs["CommunityIndex"]
    for userIndex in tqdm(range(len(gThresholded.vs))):
        user = labels[userIndex]
        community = communities[userIndex]
        if(user in user2urlCounter):
            community2urlCounter[community].update(user2urlCounter[user])
            community2urlSumTotal[community] += sum(user2urlCounter[user].values())
        if(user in user2hashtagsCounter):
            community2hashtagsCounter[community].update(user2hashtagsCounter[user])
            community2hashtagsSumTotal[community] += sum(user2hashtagsCounter[user].values())
        if(user in user2retweetsCounter):
            community2retweetsCounter[community].update(user2retweetsCounter[user])
            community2retweetsSumTotal[community] += sum(user2retweetsCounter[user].values())
        if(user in user2tokensCounter):
            community2tokensCounter[community].update(user2tokensCounter[user])
            community2tokensSumTotal[community] += sum(user2tokensCounter[user].values())
        if(user in user2RetweetTokensCounter):
            community2RetweetTokensCounter[community].update(user2RetweetTokensCounter[user])
            community2RetweetTokensSumTotal[community] += sum(user2RetweetTokensCounter[user].values())


    community2urlRelativeDifferenceCounter = {}
    community2hashtagsRelativeDifferenceCounter = {}
    community2retweetsRelativeDifferenceCounter = {}
    community2tokensRelativeDifferenceCounter = {}
    community2RetweetTokensRelativeDifferenceCounter = {}

    # incommunity/incommunitytotal - (all-incommunity)/(all-incommunitytotal)
    for community in tqdm(set(gThresholded.vs["CommunityIndex"])):
        community2urlRelativeDifferenceCounter[community] = Counter()
        community2hashtagsRelativeDifferenceCounter[community] = Counter()
        community2retweetsRelativeDifferenceCounter[community] = Counter()
        community2tokensRelativeDifferenceCounter[community] = Counter()
        community2RetweetTokensRelativeDifferenceCounter[community] = Counter()
        for url in community2urlCounter[community]:
            incommunityRelativeFrequency = community2urlCounter[community][url]/community2urlSumTotal[community]
            if (urlSumCount - community2urlSumTotal[community]) == 0:
                outcommunityRelativeFrequency = url2TotalCounts[url]
            else:
                outcommunityRelativeFrequency = (url2TotalCounts[url] - community2urlCounter[community][url])/(urlSumCount - community2urlSumTotal[community])
            community2urlRelativeDifferenceCounter[community][url] = transform(incommunityRelativeFrequency) - penaltyFactor*transform(outcommunityRelativeFrequency)
        for hashtag in community2hashtagsCounter[community]:
            incommunityRelativeFrequency = community2hashtagsCounter[community][hashtag]/community2hashtagsSumTotal[community]
            if (hashtagSumCount - community2hashtagsSumTotal[community]) == 0:
                outcommunityRelativeFrequency = hashtag2TotalCounts[hashtag]
            else:
                outcommunityRelativeFrequency = (hashtag2TotalCounts[hashtag] - community2hashtagsCounter[community][hashtag])/(hashtagSumCount - community2hashtagsSumTotal[community])
            community2hashtagsRelativeDifferenceCounter[community][hashtag] = transform(incommunityRelativeFrequency) - penaltyFactor*transform(outcommunityRelativeFrequency)
        for retweet in community2retweetsCounter[community]:
            incommunityRelativeFrequency = community2retweetsCounter[community][retweet]/community2retweetsSumTotal[community]
            if (retweetSumCount - community2retweetsSumTotal[community]) == 0:
                outcommunityRelativeFrequency = retweet2TotalCounts[retweet]
            else:
                outcommunityRelativeFrequency = (retweet2TotalCounts[retweet] - community2retweetsCounter[community][retweet])/(retweetSumCount - community2retweetsSumTotal[community])
            community2retweetsRelativeDifferenceCounter[community][retweet] = transform(incommunityRelativeFrequency) - penaltyFactor*transform(outcommunityRelativeFrequency)
        for token in community2tokensCounter[community]:
            incommunityRelativeFrequency = community2tokensCounter[community][token]/community2tokensSumTotal[community]
            if (tokenSumCount - community2tokensSumTotal[community]) == 0:
                outcommunityRelativeFrequency = token2TotalCounts[token]
            else:
                outcommunityRelativeFrequency = (token2TotalCounts[token] - community2tokensCounter[community][token])/(tokenSumCount - community2tokensSumTotal[community])
            # if(community==1 and token=="china"):
            #     print("incommunityCount:", community2tokensCounter[community][token])
            #     print("incommunityTotal:", community2tokensSumTotal[community])
            #     print("outcommunityCount:", token2TotalCounts[token] - community2tokensCounter[community][token])
            #     print("outcommunityTotal:", tokenSumCount - community2tokensSumTotal[community])
            #     print("incommunityRelativeFrequency:", incommunityRelativeFrequency)
            #     print("outcommunityRelativeFrequency:", outcommunityRelativeFrequency)
            #     print("relativeDifference:", transform(incommunityRelativeFrequency) - penaltyFactor*transform(outcommunityRelativeFrequency))
            community2tokensRelativeDifferenceCounter[community][token] = transform(incommunityRelativeFrequency) - penaltyFactor*transform(outcommunityRelativeFrequency)
        for token in community2RetweetTokensCounter[community]:
            incommunityRelativeFrequency = community2RetweetTokensCounter[community][token]/community2RetweetTokensSumTotal[community]
            if (retweetTokenSumCount - community2RetweetTokensSumTotal[community]) == 0:
                outcommunityRelativeFrequency = retweetToken2TotalCounts[token]
            else:
                outcommunityRelativeFrequency = (retweetToken2TotalCounts[token] - community2RetweetTokensCounter[community][token])/(retweetTokenSumCount - community2RetweetTokensSumTotal[community])

            community2RetweetTokensRelativeDifferenceCounter[community][token] = transform(incommunityRelativeFrequency) - penaltyFactor*transform(outcommunityRelativeFrequency)

    # 0.05517002081887578-0.21999104936469946
    mostCommonLimit = 8
    community2topURLs = {}
    community2topHashtags = {}
    community2topRetweets = {}
    community2topTokens = {}
    community2topRetweetTokens = {}

    community2topSurprisingURLs = {}
    community2topSurprisingHashtags = {}
    community2topSurprisingRetweets = {}
    community2topSurprisingTokens = {}
    community2topSurprisingRetweetTokens = {}

    for community in tqdm(set(gThresholded.vs["CommunityIndex"])):
        community2topURLs[community] = community2urlCounter[community].most_common(mostCommonLimit)
        community2topHashtags[community] = community2hashtagsCounter[community].most_common(mostCommonLimit)
        community2topRetweets[community] = community2retweetsCounter[community].most_common(mostCommonLimit)
        community2topTokens[community] = community2tokensCounter[community].most_common(mostCommonLimit*3)
        community2topRetweetTokens[community] = community2RetweetTokensCounter[community].most_common(mostCommonLimit*3)

        community2topSurprisingURLs[community] = community2urlRelativeDifferenceCounter[community].most_common(mostCommonLimit)
        community2topSurprisingHashtags[community] = community2hashtagsRelativeDifferenceCounter[community].most_common(mostCommonLimit)
        community2topSurprisingRetweets[community] = community2retweetsRelativeDifferenceCounter[community].most_common(mostCommonLimit)
        community2topSurprisingTokens[community] = community2tokensRelativeDifferenceCounter[community].most_common(mostCommonLimit*3)
        community2topSurprisingRetweetTokens[community] = community2RetweetTokensRelativeDifferenceCounter[community].most_common(mostCommonLimit*3)

    community2topSummaryURLs = {}
    community2topSummaryHashtags = {}
    community2topSummaryRetweets = {}
    community2topSummaryTokens = {}
    community2topSummaryRetweetTokens = {}

    community2SurprisingSummaryURLs = {}
    community2SurprisingSummaryHashtags = {}
    community2SurprisingSummaryRetweets = {}
    community2SurprisingSummaryTokens = {}
    community2SurprisingSummaryRetweetTokens = {}

    for community in tqdm(set(gThresholded.vs["CommunityIndex"])):
        # each summary should be a string of semi comma separated values
        community2topSummaryURLs[community] = "; ".join([f"{entry}" for entry,_ in community2topURLs[community]])
        community2topSummaryHashtags[community] = "; ".join([f"{entry}" for entry,_ in community2topHashtags[community]])
        community2topSummaryRetweets[community] = "; ".join([f"{entry}" for entry,_ in community2topRetweets[community]])
        community2topSummaryTokens[community] = "; ".join(filterNgramParts([f"{entry}" for entry,_ in community2topTokens[community]],maxTokens=mostCommonLimit))
        community2topSummaryRetweetTokens[community] = "; ".join(filterNgramParts([f"{entry}" for entry,_ in community2topRetweetTokens[community]],maxTokens=mostCommonLimit))

        community2SurprisingSummaryURLs[community] = "; ".join([f"{entry}" for entry,_ in community2topSurprisingURLs[community]])
        community2SurprisingSummaryHashtags[community] = "; ".join([f"{entry}" for entry,_ in community2topSurprisingHashtags[community]])
        community2SurprisingSummaryRetweets[community] = "; ".join([f"{entry}" for entry,_ in community2topSurprisingRetweets[community]])
        community2SurprisingSummaryTokens[community] = "; ".join(filterNgramParts([f"{entry}" for entry,_ in community2topSurprisingTokens[community]],maxTokens=mostCommonLimit))
        community2SurprisingSummaryRetweetTokens[community] = "; ".join(filterNgramParts([f"{entry}" for entry,_ in community2topSurprisingRetweetTokens[community]],maxTokens=mostCommonLimit))

    # now set the attributes of nodes based on the community
    # "Top URLs","Top Hashtags","Top Retweets","Top Tokens","Top Retweet Tokens"
    # "Surprising URLs","Surprising Hashtags","Surprising Retweets","Surprising Tokens","Surprising Retweet Tokens"
    for nodeIndex in tqdm(range(len(gThresholded.vs))):
        node = gThresholded.vs[nodeIndex]
        community = node["CommunityIndex"]
        node["Top URLs"] = community2topSummaryURLs[community]
        if(not node["Top URLs"]):
            node["Top URLs"] = "None"
        node["Top Hashtags"] = community2topSummaryHashtags[community]
        if(not node["Top Hashtags"]):
            node["Top Hashtags"] = "None"
        # node["Top Retweets"] = community2topSummaryRetweets[community]
        # if(not node["Top Retweets"]):
        #     node["Top Retweets"] = "None
        node["Top Tokens"] = community2topSummaryTokens[community]
        if(not node["Top Tokens"]):
            node["Top Tokens"] = "None"

        node["Top Retweet Tokens"] = community2topSummaryRetweetTokens[community]
        if(not node["Top Retweet Tokens"]):
            node["Top Retweet Tokens"] = "None"
        node["Surprising URLs"] = community2SurprisingSummaryURLs[community]
        if(not node["Surprising URLs"]):
            node["Surprising URLs"] = "None"
        node["Surprising Hashtags"] = community2SurprisingSummaryHashtags[community]
        if(not node["Surprising Hashtags"]):
            node["Surprising Hashtags"] = "None"
        # node["Surprising Retweets"] = community2SurprisingSummaryRetweets[community]
        # if(not node["Surprising Retweets"]):
        #     node["Surprising Retweets"] = "None"
        node["Surprising Tokens"] = community2SurprisingSummaryTokens[community]
        if(not node["Surprising Tokens"]):
            node["Surprising Tokens"] = "None"
        node["Surprising Retweet Tokens"] = community2SurprisingSummaryRetweetTokens[community]
        if(not node["Surprising Retweet Tokens"]):
            node["Surprising Retweet Tokens"] = "None"
    return gThresholded


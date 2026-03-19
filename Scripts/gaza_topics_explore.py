import pandas as pd
from pathlib import Path
import xnetwork as xn
from collections import Counter
import coordinationz.communities as czcomm
from tqdm.auto import tqdm
import math
import openai
import json
import ast
tqdm.pandas()

postsPath = Path("Data/Preprocessed/hamas_israel_challenge_merged.feather")
networkPath = Path("Outputs/Networks/hamas_israel_challenge_merged_separated_wimagesv4_merged_0.999995_coreness.xnet")
usersDataPath = Path("Outputs/Tables/hamas_israel_challenge_merged_separated_wimagesv4_merged_nodes_0.999995_final.feather")
outputNetworkPath = Path("Outputs/finalGaza/hamas_israel_challenge_merged_separated_wimagesv4_merged_0.999995_coreness_processed_v2.xnet")
outputUsersDataPath = Path("Outputs/finalGaza/hamas_israel_challenge_merged_separated_wimagesv4_merged_nodes_0.999995_processed_v2.feather")
outputComponentDescriptionPath = Path("Outputs/finalGaza/hamas_israel_challenge_merged_separated_wimagesv4_merged_component_description.feather")
outputComponentMetadataPath = Path("Outputs/finalGaza/hamas_israel_challenge_merged_separated_wimagesv4_merged_component_metadata_v2.json")
outputComponentReportPath = Path("Outputs/finalGaza/hamas_israel_challenge_merged_separated_wimagesv4_merged_component_report_v2.txt")

outputNetworkCoordinatingPath = Path("Outputs/finalGaza/hamas_israel_challenge_merged_separated_wimagesv4_merged_0.999995_coreness_processed_onlycoordinating_v2.xnet")

outputNetworkPath.parent.mkdir(parents=True, exist_ok=True)


df_users = pd.read_feather(usersDataPath)



df_posts = pd.read_feather(postsPath)
g = xn.load(networkPath)

componentMemberships = g.connected_components().membership
componentSizes = Counter(componentMemberships)
# reindex by size
membership2ID = {membership: idx for idx, (membership, _) in enumerate(componentSizes.most_common())}
ID2membership = {idx: membership for idx, (membership, _) in enumerate(componentSizes.most_common())}
# reindex components by size
g.vs["ComponentID"] = [membership2ID[membership] for membership in componentMemberships]
g.vs["ComponentSize"] = [componentSizes[membership] for membership in componentMemberships]
for attribute in g.vs.attributes():
    if attribute.startswith("Community"):
        del g.vs[attribute]  # remove all community attributes
    if(attribute.startswith("Surprising")):
        del g.vs[attribute]
    if(attribute.startswith("Top")):
        del g.vs[attribute]
    if(attribute.startswith("Coordinating")):
        del g.vs[attribute]

minCoordinationComponentSize = 5
g.vs["Coordinating"] = [1 if g.vs[i]["ComponentSize"] >= minCoordinationComponentSize else 0 for i in range(g.vcount())]

# save the graph with coordinating nodes
g_coordinating = g.subgraph(g.vs.select(Coordinating_eq=1))
xn.save(g_coordinating, outputNetworkCoordinatingPath)
# save entire graph with coordinating nodes
xn.save(g, outputNetworkPath)



# transform hashtags "['tag1', 'tag2']" to list
def parse_maybe_list(x):
    if not isinstance(x, str):
        return x
    try:
        return ast.literal_eval(x)
    except Exception:
        return x

def ensure_list(x):
    if isinstance(x, list):
        return x
    if pd.isna(x):
        return []
    if isinstance(x, tuple):
        return list(x)
    return []

df_posts["hashtags"] = df_posts["hashtags"].progress_apply(parse_maybe_list).progress_apply(ensure_list)
df_posts["urls"] = df_posts["urls"].progress_apply(parse_maybe_list).progress_apply(ensure_list)
df_posts["linked_tweet"] = df_posts["linked_tweet"].progress_apply(parse_maybe_list).progress_apply(ensure_list)



df_posts = df_posts.copy()
df_posts["contentText"] = df_posts["text"]
if("data_translatedContentText" in df_posts and not df_posts["data_translatedContentText"].isna().all()):
    df_posts["contentText"] = df_posts["data_translatedContentText"]
    # for the nans, use the original text
    mask = df_posts["contentText"].isna()
    df_posts.loc[mask,"contentText"] = df_posts["text"][mask]


tweetID2TokensCache = {}

# onlyEntries in set
# dfFiltered = df_posts[df_posts["user_id"].isin(allUsers)]
dfFiltered = df_posts
dfRetweets = dfFiltered[dfFiltered["tweet_type"]=="retweet"]
dfOriginal = dfFiltered[dfFiltered["tweet_type"]!="retweet"]
dfInNetworkURLs = dfOriginal.dropna(subset=["urls"])
dfInNetworkHashtags = dfOriginal.dropna(subset=["hashtags"])
# for tokens use czind.tokenizeTweet(string)

dfInNetworkTokens = dfOriginal.dropna(subset=["contentText"]).copy()
# use translated 
# apply getTokens to text, tweet_id
dfInNetworkTokens["tokens"] = dfInNetworkTokens[["tweet_id","contentText"]].progress_apply(lambda x: czcomm.getTokens(*x,tweetID2TokensCache),axis=1)
# get all users with degree >0
userIDs = g.vs["Label"]
degrees = g.degree()
userIDsInNetwork = set([userIDs[i] for i in range(len(userIDs)) if degrees[i] > 0])

dfInNetworkRetweets = dfRetweets.dropna(subset=["linked_tweet"])
# from dfInNetworkRetweets keep only users in userIDsInNetwork + a 5% sample
usersSampled = set(dfInNetworkRetweets["user_id"].sample(frac=0.01, random_state=42).unique())
dfInNetworkRetweets = dfInNetworkRetweets[dfInNetworkRetweets["user_id"].isin(userIDsInNetwork.union(usersSampled))]

dfInNetworkRetweetTokens = dfInNetworkRetweets.dropna(subset=["contentText"]).copy()
if(dfInNetworkRetweetTokens.empty):
    dfInNetworkRetweetTokens["tokens"] = pd.Series(dtype=object)
else:
    dfInNetworkRetweetTokens["tokens"] = dfInNetworkRetweetTokens[["tweet_id","contentText"]].progress_apply(lambda x: czcomm.getTokens(*x,tweetID2TokensCache),axis=1)






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



def logodds(corpora_dic, bg_counter):
    """ It calculates the log odds ratio of term i's frequency between 
    a target corpus and another corpus, with the prior information from
    a background corpus. Inputs are:
    
    - a dictionary of Counter objects (corpora of our interest)
    - a Counter objects (background corpus)
    
    Output is a dictionary of dictionaries. Each dictionary contains the log 
    odds ratio of each word. 
    
    """
    corp_size = dict([(c, sum(corpora_dic[c].values())) for c in corpora_dic])
    bg_size = sum(bg_counter.values())
    result = dict([(c, {}) for c in corpora_dic])
    
    for name, c in corpora_dic.items():
        for word in c:
            #if 10 > sum(1 for corpus in corpora_dic.values() if corpus[word]):
            #    continue
            
            fi = c[word]
            fj = sum(co[word] for x, co in corpora_dic.items() if x != name)
            fbg = bg_counter[word]
            ni = corp_size[name]
            nj = sum(x for idx, x in corp_size.items() if idx != name)
            nbg = bg_size
            oddsratio = math.log(fi+fbg) - math.log(ni+nbg-(fi+fbg)) -\
                        math.log(fj+fbg) + math.log(nj+nbg-(fj+fbg))
            std = 1.0 / (fi+fbg) + 1.0 / (fj+fbg)
            z = oddsratio / math.sqrt(std)
            result[name][word] = z
            
    # Sort words by log-odds
    grouped_sorted_ngrams = {key: sorted(entry.items(), key=lambda x: x[1], reverse=True)
                             for key, entry in result.items()}
    return grouped_sorted_ngrams



def relativeDifference(corpora_dic, bg_counter):
    #relDiff_c[word] = (f_c[word]+0.01)/(Tf_c+0.01) - (f_bg[word]-f_c[word]+0.01)/(Tf_bg-f_c[word]+0.01)
    """ It calculates the relative difference of term i's frequency between
    a target corpus and another corpus, with the prior information from
    a background corpus. Inputs are:
    - a dictionary of Counter objects (corpora of our interest)
    - a Counter objects (background corpus)
    Output is a dictionary of dictionaries. Each dictionary contains the relative
    difference of each word.
    """
    corp_size = dict([(c, sum(corpora_dic[c].values())) for c in corpora_dic])
    bg_size = sum(bg_counter.values())
    result = dict([(c, {}) for c in corpora_dic])
    for name, c in corpora_dic.items():
        for word in c:
            #if 10 > sum(1 for corpus in corpora_dic.values() if corpus[word]):
            #    continue
            
            fi = c[word]
            fj = sum(co[word] for x, co in corpora_dic.items() if x != name)
            fbg = bg_counter[word]
            ni = corp_size[name]
            nj = sum(x for idx, x in corp_size.items() if idx != name)
            nbg = bg_size
            relDiff = (fi+0.01)/(ni+0.01) - (fbg-fi+0.01)/(nbg-ni+0.01)
            result[name][word] = relDiff
    # Sort words by relative difference
    grouped_sorted_ngrams = {key: sorted(entry.items(), key=lambda x: x[1], reverse=True)
                                for key, entry in result.items()}   
    return grouped_sorted_ngrams


communities = g.vs["ComponentID"]
userIDs = g.vs["Label"]

community2UserIDs = {}
for userID, community in zip(userIDs, communities):
    if community not in community2UserIDs:
        community2UserIDs[community] = set()
    community2UserIDs[community].add(userID)
    

community2urlCounter = {}
community2hashtagsCounter = {}
community2retweetsCounter = {}
community2tokensCounter = {}
community2RetweetTokensCounter = {}

for community, users in community2UserIDs.items():
    # print(f"Community {community} has users: {users}")
    community2urlCounter[community] = Counter()
    community2hashtagsCounter[community] = Counter()
    community2retweetsCounter[community] = Counter()
    community2tokensCounter[community] = Counter()
    community2RetweetTokensCounter[community] = Counter()
    for user in users:
        if user in user2urlCounter:
            community2urlCounter[community].update(user2urlCounter[user])
        if user in user2hashtagsCounter:
            community2hashtagsCounter[community].update(user2hashtagsCounter[user])
        if user in user2retweetsCounter:
            community2retweetsCounter[community].update(user2retweetsCounter[user])
        if user in user2tokensCounter:
            community2tokensCounter[community].update(user2tokensCounter[user])
        if user in user2RetweetTokensCounter:
            community2RetweetTokensCounter[community].update(user2RetweetTokensCounter[user])

topCommunities = sorted(community2UserIDs.items(), key=lambda x: len(x[1]), reverse=True)

top20Communities = [community for community, _ in topCommunities[:20]]
topCommunity2urlCounter = {community: community2urlCounter[community] for community in top20Communities}
topCommunity2hashtagsCounter = {community: community2hashtagsCounter[community] for community in top20Communities}
topCommunity2retweetsCounter = {community: community2retweetsCounter[community] for community in top20Communities}
topCommunity2tokensCounter = {community: community2tokensCounter[community] for community in top20Communities}
topCommunity2RetweetTokensCounter = {community: community2RetweetTokensCounter[community] for community in top20Communities}

topCommunity2urlLogOdds = logodds(topCommunity2urlCounter, url2TotalCounts)
topCommunity2hashtagsLogOdds = logodds(topCommunity2hashtagsCounter, hashtag2TotalCounts)
topCommunity2retweetsLogOdds = logodds(topCommunity2retweetsCounter, retweet2TotalCounts)
topCommunity2tokensLogOdds = logodds(topCommunity2tokensCounter, token2TotalCounts)
topCommunity2RetweetTokensLogOdds = logodds(topCommunity2RetweetTokensCounter, retweetToken2TotalCounts)


# topCommunity2urlRelativeDiff = relativeDifference(topCommunity2urlCounter, url2TotalCounts)
# topCommunity2hashtagsRelativeDiff = relativeDifference(topCommunity2hashtagsCounter, hashtag2TotalCounts)
# topCommunity2retweetsRelativeDiff = relativeDifference(topCommunity2retweetsCounter, retweet2TotalCounts)
# topCommunity2tokensRelativeDiff = relativeDifference(topCommunity2tokensCounter, token2TotalCounts)
# topCommunity2RetweetTokensRelativeDiff = relativeDifference(topCommunity2RetweetTokensCounter, retweetToken2TotalCounts)




# unify everything
topCommunity2Metadata = {}
for community in top20Communities:
    urlImportance = topCommunity2urlLogOdds[community]
    hashtagsImportance = topCommunity2hashtagsLogOdds[community]
    retweetsImportance = topCommunity2retweetsLogOdds[community]
    tokensImportance = topCommunity2tokensLogOdds[community]
    retweetTokensImportance = topCommunity2RetweetTokensLogOdds[community]
    topCommunity2Metadata[community] = {
        "urlImportance": urlImportance,
        "hashtagsImportance": hashtagsImportance,
        "retweetsImportance": retweetsImportance,
        "tokensImportance": tokensImportance,
        "retweetTokensImportance": retweetTokensImportance,
    }

with outputComponentMetadataPath.open("w") as f:
    json.dump(topCommunity2Metadata, f, indent=4, ensure_ascii=False)

roleDescription = """Task: As a neutral language model, your role is to provide a clear, objective, and logical description of various clusters based on a list of keywords, hashtags, and urls. These terms are arranged in order of importance (from highest to lowest). The information you provide will be used for academic purposes to better understand these clusters. REMEMBER THE TERMS ARE IN ORDER OF IMPORTANCE.

Each prompt contains keywords, hashtags, and urls. Your task is to create a title and a description for each cluster. The descriptions should be in English and must remain neutral, avoiding any endorsement or invitation towards these clusters.

Your output should strictly follow this JSON structure:
{title: <TITLE>, description: <DESCRIPTION>}

This structure represents a list of dictionary entry with title and description.

Please note that the clusters have been identified using a clustering algorithm, meaning they organically formed. The purpose of these descriptions is not to advertise the clusters, but to offer a neutral overview. The overall subject is the Gaza-Israel conflict, and the clusters are based on the content of posts related to this topic."""






def getDescriptions(community,maxTrials = 4):
    client = openai.OpenAI(
        # Defaults to os.environ.get("OPENAI_API_KEY")
        # Otherwise use: api_key="Your_API_Key",
    )
    niceText = ""
    # niceText for communities keywords:
    # niceText += f"Data for cluster {community}:"
    niceText += f"Keywords:\n" 
    niceText += "  - "+" ".join([entry for entry,importance in topCommunity2Metadata[community]["tokensImportance"]][:30])


    niceText += "\n"


    niceText += f"Hashtags:\n"
    niceText += "  - "+" ".join([entry for entry,importance in topCommunity2Metadata[community]["hashtagsImportance"]][:30])
    niceText += "\n"
    niceText += f"Urls:\n"
    niceText += "  - "+" ".join([entry for entry,importance in topCommunity2Metadata[community]["urlImportance"]][:30])
    niceText += "\n"
    niceText += f"RetweetsTokens:\n"
    niceText += "  - "+" ".join([entry for entry,importance in topCommunity2Metadata[community]["retweetTokensImportance"]][:30])
    niceText += "\n"

    communityDescription = {}
    jsonResults = None
    while(jsonResults is None and maxTrials>0):
        print("Trying to generate descriptions... (maxTrials: "+str(maxTrials)+")")
        # print("Prompt:\n",roleDescription)
        print("Data:\n",niceText)
        maxTrials-=1
        # chat_completion  = client.chat.completions.create(
        #     model="gpt-3.5-turbo",
        #     messages=[
        #             {"role": "system", "content": roleDescription},
        #             {"role": "user", "content": niceText},
        #     ]
        # )
        chat_completion = client.chat.completions.create(
            model="gpt-5.2",
            response_format={ "type": "json_object" },
            messages=[
                {"role": "system", "content": roleDescription},
                {"role": "user", "content": niceText}
            ]
        )

        originalResults = chat_completion.choices[0].message.content.replace("\n\n","\n")
        print("\n-------RESULTS:------")
        print(originalResults)

        try: 
            jsonResults = json.loads(originalResults)
        except:
            pass

        if(jsonResults is None):
            results = originalResults.replace("\n",",")
            try:
                jsonResults = json.loads(results)
            except:
                pass
        if(jsonResults is None):
            results = "[" + originalResults + "]"
            try:
                jsonResults = json.loads(results)
            except:
                pass
        if(jsonResults is None):
            results = "[" + originalResults.replace("\n",",") + "]"
            try:
                jsonResults = json.loads(results)
            except:
                pass
        
        try:
            if("title" in jsonResults and "description" in jsonResults):
                communityDescription = jsonResults
            else:
                print("ERROR: No TITLE or DESCRIPTION in results")
                print("Results: ",jsonResults)
                jsonResults = None

        except:
            jsonResults = None #try again
            
    if(jsonResults is None):
        print("ERROR: Could not parse results")
        return None
    else:
        return communityDescription



communities2Descriptions = {}
for communityIndex in top20Communities:
    communityDescription = getDescriptions(communityIndex)
    
    if(communityDescription is None):
        print("ERROR: Could not get descriptions")
        break
    else:
        communities2Descriptions[communityIndex] = communityDescription
        print("Descriptions so far: ",len(communities2Descriptions.keys()))



# save to txt file
with open(outputComponentReportPath,"w") as f:
    for community in top20Communities:
        # Community number and language
        communitySize = len(community2UserIDs[community])
        communityDescription = communities2Descriptions.get(community, {"title": "Unknown", "description": "Description unavailable"})
        f.write(f"Community {community} ({communitySize} users)\n")
        f.write(f"\t Title: {communityDescription['title']}\n")
        f.write(f"\t Description: {communityDescription['description']}\n")
        
        # top 50 keywords
        f.write("\t KEYWORDS: ")

        f.write(", ".join([f"{entry} ({importance:.2f})" for entry,importance in topCommunity2Metadata[community]["tokensImportance"][:50]]))
        f.write("\n")
        # top 50 hashtags
        f.write("\t HASHTAGS: ")
        f.write(", ".join([f"{entry} ({importance:.2f})" for entry,importance in topCommunity2Metadata[community]["hashtagsImportance"][:50]]))
        f.write("\n")
        # top 50 retweet tokens
        f.write("\t RETWEET TOKENS: ")
        f.write(", ".join([f"{entry} ({importance:.2f})" for entry,importance in topCommunity2Metadata[community]["retweetTokensImportance"][:50]]))
        f.write("\n")

        # top 5 urls (one per line with - before it
        f.write("\t URLS: \n")
        for entry,importance in topCommunity2Metadata[community]["urlImportance"][:5]:
            f.write(f"\t - {entry} ({importance:.2f})\n")
        f.write("\n\n")
        

g.vs["ComponentTitle"] = [communities2Descriptions[community]['title'].replace("\"","").replace("\"","") if community in communities2Descriptions else "Other" for community in g.vs["ComponentID"]]

g.vs["Hashtags"] = [", ".join([entry for entry,_ in topCommunity2Metadata[community]["hashtagsImportance"][:5]]) if community in topCommunity2Metadata else "Other" for community in g.vs["ComponentID"]]

g.vs["Tokens"] = [", ".join([entry for entry,_ in topCommunity2Metadata[community]["tokensImportance"][:5]]) if community in topCommunity2Metadata else "Other" for community in g.vs["ComponentID"]]

g.vs["RetweetTokens"] = [", ".join([entry for entry,_ in topCommunity2Metadata[community]["retweetTokensImportance"][:5]]) if community in topCommunity2Metadata else "Other" for community in g.vs["ComponentID"]]

xn.save(g, outputNetworkPath)

g_coordinating = g.subgraph(g.vs.select(Coordinating_eq=1))
xn.save(g_coordinating, outputNetworkCoordinatingPath)

# create a dataframe with all the nodes from the graph
df_nodes = pd.DataFrame({
    "user_id": g.vs["Label"],
    "ComponentID": g.vs["ComponentID"],
    "ComponentSize": g.vs["ComponentSize"],
    "Coordinating": g.vs["Coordinating"],
    "ComponentTitle": g.vs["ComponentTitle"],
    "Degree": g.vs.degree(),
    # "Hashtags": g.vs["Hashtags"],
    # "Tokens": g.vs["Tokens"],
    # "RetweetTokens": g.vs["RetweetTokens"]
})
# make Coordinating a boolean
df_nodes["Coordinating"] = df_nodes["Coordinating"].astype(bool)

# save the nodes dataframe
df_nodes.to_feather(outputUsersDataPath)
from pathlib import Path
import pandas as pd
import numpy as np
from tqdm.auto import tqdm
import coordinationz as cz
import xnetwork as xn

from scipy.sparse import csr_matrix



if __name__ == "__main__":
    config = cz.config
    # config = cz.load_config("<path to config>")

    # dataName = "challenge_filipinos_5DEC"
    dataName = "challenge_problem_two_21NOV_activeusers"
    # dataName = "challenge_problem_two_21NOV"

    dataPath = Path(config["paths"]["EVALUATION_DATASETS"])

    networksPath = Path(config["paths"]["NETWORKS"]).resolve()
    networksPath.mkdir(parents=True, exist_ok=True)

    # pandas tqdm
    tqdm.pandas()

    df = pd.read_csv(dataPath/f'{dataName}.csv',
                    dtype={
                        'screen_name':str,
                        'linked_tweet':str
                        })


    minActivities = 10

    # only keep users with at least minActivities
    userActivityCount = df["screen_name"].value_counts()
    usersWithMinActivities = set(userActivityCount[userActivityCount >= minActivities].index)
    df = df[df["screen_name"].isin(usersWithMinActivities)]


    # keep only tweet_type == "retweet"
    df = df[df["tweet_type"] == "retweet"]
    bipartiteEdges = df[["screen_name","linked_tweet"]].values


    print("Indexing bipartite edges")
    bipartiteIndexedEdges = np.zeros(bipartiteEdges.shape, dtype=int)
    leftIndex2Label = [label for label in np.unique(bipartiteEdges[:,0])]
    leftLabel2Index = {label: index for index, label in enumerate(leftIndex2Label)}
    rightIndex2Label = [label for label in np.unique(bipartiteEdges[:,1])]
    rightLabel2Index = {label: index for index, label in enumerate(rightIndex2Label)}
    
    # create indexed edges in a numpy array integers
    bipartiteIndexedEdges[:,0] = [leftLabel2Index[label] for label in bipartiteEdges[:,0]]
    bipartiteIndexedEdges[:,1] = [rightLabel2Index[label] for label in bipartiteEdges[:,1]]


    leftCount = len(leftIndex2Label)
    rightCount = len(rightIndex2Label)


    edgesCount = len(bipartiteIndexedEdges)
    edges = np.array(bipartiteIndexedEdges)

    # edges = np.array(edges)
    # import coordinationz.fastcosine as fastcosine;

    # result = fastcosine.cosine(edges,
    #                            leftCount=leftCount,
    #                            rightCount=rightCount,
    #                            threshold=0.1,
    #                            returnDictionary=True,
    #                         #    leftEdges = [[0,105]],
    #                            updateCallback="progress")


    # results2 = bipartiteCosineSimilarityMatrixThresholded(decodedEdges, leftCount=leftCount, rightCount=rightCount, threshold=0.1)
    nullModelOutput = cz.nullmodel.bipartiteNullModelSimilarity(
        edges,
        scoreType="pvalue-quantized",
        realizations=1000,
        batchSize=1,
        workers=-1,
        minSimilarity = 0.1,
    )

    # g = cz.network.createNetworkFromNullModelOutput(
    #     nullModelOutput
    # )

    # xn.save(g, networksPath/f"{dataName}_coretweet.xnet")

    # already sorted
    # cResult = [(edge[0],edge[1],similaririy) for edge,similaririy in zip(result[0],result[1])]
    # pyResult = sorted(results2 )

    # # loop over the values of cResult and pyResults until finding a difference
    # for cVal,pyVal in zip(cResult,pyResult):
    #     if(cVal[0]!=pyVal[0] and cVal[1]!=pyVal[1] ):
    #         print(cVal,pyVal)
    #         break
        
# def show_common(edge):
#     common = set(decodedEdges[decodedEdges[:,0]==edge[0]][:,1]).intersection(set(decodedEdges[decodedEdges[:,0]==edge[1]][:,1]))
#     print(common)
#     np.sum(
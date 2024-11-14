
# from __future__ import annotations

import numpy as np
from tqdm.auto import tqdm
from itertools import combinations
import multiprocessing as mp
from multiprocessing import Pool
from . import fastcosine
from numpy.random import SeedSequence, default_rng
from scipy.stats import rankdata
# import tracemalloc
from contextlib import closing

# fastcosine

def _tqdmDummy(*args, **kwargs):
    return args[0]

def _processBatch(parameters):
        batchRealizations,currentTQDM, \
        allowedCombinations, \
        repeatedUniqueLeftDegreesIndices, \
        shuffledEdges,allWeights,rightCount, \
        repeatedUniqueLeftDegrees, \
        isSingleBatch,childSeed = parameters \
        

        # properly handling multiprocessing random number generation
        if(childSeed is not None):
            rng = default_rng(childSeed)
        else:
            rng = default_rng()

        realizationsRange = range(batchRealizations)
        if(isSingleBatch):
            realizationsRange = currentTQDM(realizationsRange, desc="Null model realizations")
        
        batchDegreePair2similarity = {}
        batchReducedShuffledEdges = np.zeros((len(repeatedUniqueLeftDegreesIndices), 2), dtype=int)
        batchReducedShuffledEdges[:,0] = repeatedUniqueLeftDegreesIndices

        
        # print("Realizations:",realizations)
        # print(degreePair2similarity)
        # print(parameters)
        for _ in realizationsRange:
            if allowedCombinations is not None:
                edgeCombinations = allowedCombinations
            else:
                edgeCombinations = combinations(range(len(repeatedUniqueLeftDegrees)), 2)
            # FIXME: There is a chance that the combination of degrees may be larger than the number of edges
            # For now, if that is the case, it will enable replacement
            # Once that only degrees combinations existing in the data will be used
            # then we can remove the replacement
            # print(len(shuffledEdges[:,1]))
            if(len(shuffledEdges[:,1])<len(batchReducedShuffledEdges)):
                choiceIndices = rng.choice(len(shuffledEdges[:,1]), len(batchReducedShuffledEdges), replace=True)
            else:
                choiceIndices = rng.choice(len(shuffledEdges[:,1]), len(batchReducedShuffledEdges), replace=False)
            
            batchReducedShuffledEdges[:,1] = shuffledEdges[choiceIndices,1]
            if(allWeights is not None):
                weights = allWeights[choiceIndices]
            else:
                weights = None
            # FIXME: Incude the option to choose the similarity metric
            # or to send a custom function
            modelSimilarityDictionary={}
            modelSimilarityDictionary = fastcosine.bipartiteCosine(
                batchReducedShuffledEdges,
                weights=weights,
                rightCount=rightCount,
                returnDictionary=True,
                leftEdges= allowedCombinations if allowedCombinations is not None else None,
                threshold=0.0,
            )
            # (batchReducedShuffledEdges,rightCount=rightCount)
            # ------
            # for (fromIndex, toIndex),similarity in modelSimilarityDictionary.items():
            #     # print(fromIndex, toIndex)
            #     fromDegree = repeatedUniqueLeftDegrees[fromIndex]
            #     toDegree = repeatedUniqueLeftDegrees[toIndex]
            #     degreeEdge = (min(fromDegree, toDegree), max(fromDegree, toDegree))
            #     if(degreeEdge not in batchDegreePair2similarity):
            #         batchDegreePair2similarity[degreeEdge] = []
            #     batchDegreePair2similarity[degreeEdge].append(similarity)
            # ------
            # edgeCombinations = list(edgeCombinations)
            # print("edgeCombinations",edgeCombinations)

            # edgeCombinations [(0, 2), (0, 3), (1, 2), (1, 3), (2, 4), (2, 5), (3, 4), (3, 5)]

            for fromIndex, toIndex in edgeCombinations:
                # print(fromIndex, toIndex)
                if(fromIndex==toIndex):
                    continue
                combinationIndices = (fromIndex, toIndex)
                if(combinationIndices not in modelSimilarityDictionary):
                    similarity=0.0
                    # continue
                else:
                    similarity = modelSimilarityDictionary[combinationIndices]
                fromDegree = repeatedUniqueLeftDegrees[fromIndex]
                toDegree = repeatedUniqueLeftDegrees[toIndex]
                degreeEdge = (min(fromDegree, toDegree), max(fromDegree, toDegree))
                if(degreeEdge not in batchDegreePair2similarity):
                    batchDegreePair2similarity[degreeEdge] = []
                batchDegreePair2similarity[degreeEdge].append(similarity)
        return batchDegreePair2similarity

# repetitionCount*realizations = 10000


def calculateRightIDFWeights(bipartiteIndexedEdges, leftCount, rightCount, idf):
    # weights = None
    right2IDF = None
    if(idf is not None and idf.lower() != "none"):
        idfLeftCount = leftCount
        right2IDF = np.zeros(rightCount, dtype=np.float64)
        rightCounts = np.zeros(rightCount, dtype=np.float64)
        # calculate on the Unique right counts
        # get unique edges
        bipartiteIndexedUniqueEdges = np.unique(bipartiteIndexedEdges, axis=0)
        rightBinCounts = np.bincount(bipartiteIndexedUniqueEdges[:, 1])
        # rightBinCounts = np.bincount(bipartiteIndexedEdges[:, 1])
        rightCounts[:len(rightBinCounts)] = rightBinCounts
        if(idf.lower() == "linear"):
            right2IDF = idfLeftCount / rightCounts# unique
        elif(idf.lower() == "smoothlinear"):
            right2IDF = (idfLeftCount + 1) / (rightCounts + 1)
        elif(idf.lower() == "log"):
            right2IDF = np.log(idfLeftCount / rightCounts)
        elif(idf.lower() == "smoothlog"):
            right2IDF = np.log((idfLeftCount / rightCounts) + 1)
        else:
            raise ValueError(f"Invalid idf type: {idf}")
    return right2IDF
    #     weights = right2IDF[bipartiteIndexedEdges[:, 1]]
    #     # print("weights: ",len(weights),"edges: ",len(indexedEdges))
    # # print("weights:",weights)
    # return weights
    
def calculateIDFWeights(bipartiteIndexedEdges, leftCount, rightCount, idf):
    right2IDF = calculateRightIDFWeights(bipartiteIndexedEdges, leftCount, rightCount, idf)
    if(right2IDF is None):
        return None
    weights = right2IDF[bipartiteIndexedEdges[:, 1]]
    return weights

def estimateIDFWeightsShuffled(bipartiteIndexedEdges, leftCount, rightCount, idf, realizations=10000,showProgress=True,workers = 1):
    avgWeights = np.zeros(len(bipartiteIndexedEdges), dtype=np.float64)
    # stdWeights = np.zeros(len(bipartiteIndexedEdges), dtype=np.float64)
    for _ in tqdm(range(realizations),desc="IDF estimation",disable=not showProgress,leave=False):
        shuffledEdges = bipartiteIndexedEdges.copy()
        # shuffle only right side
        np.random.shuffle(shuffledEdges[:,1])
        right2IDF=calculateRightIDFWeights(shuffledEdges, leftCount, rightCount, idf)
        if(right2IDF is None):
            return None
        nullWeights = right2IDF[bipartiteIndexedEdges[:, 1]]
        avgWeights += nullWeights
        # stdWeights += nullWeights**2
    avgWeights /= realizations
    # stdWeights = np.sqrt(stdWeights/realizations - avgWeights**2)

    # varCoefficients = stdWeights/avgWeights
    # print("Average varCoefficients",np.nanmean(varCoefficients))
    # print("Max varCoefficients",np.nanmax(varCoefficients))
    # print("Min varCoefficients",np.nanmin(varCoefficients))
    # print("Median varCoefficients",np.nanmedian(varCoefficients))
    # print("Std varCoefficients",np.nanstd(varCoefficients))
    
    return avgWeights


def bipartiteNullModelSimilarity(
        bipartiteEdges,
        scoreType=["pvalue"], # "pvalue", "onlynullmodel", "quantile", "quantilesimilarity" are valid
        realizations = 10000,
        repetitionCount = 2,
        minSimilarity = 0.0,
        idf = None, # None, "none", "log","log1p",
        IDFWeightsRealizations = 100,
        returnDegreeSimilarities = False,
        returnDegreeValues = False,
        showProgress=True,
        batchSize = 100,
        workers = -1
        ):
    """
    Calculate the null model similarity of a bipartite graph
    given the indexed edges of the bipartite graph.

    Parameters:
    ----------
    bipartiteEdges: np.ndarray
        The indexed edges of the bipartite graph
    scoreType: string or list containing "pvalue", "quantile", "quantilesimilarity" or "onlynullmodel"
        The type of score to calculate, can be "pvalue",
        "quantile","quantilesimilarity", or "onlynullmodel",
        if "onlynullmodel" is selected, then the function will only return the
        degree similarities.
        if "quantile" is used, then it will return the quantile of the similarity across all possible links
        defaults to ["pvalue"]
    realizations: int
        The number of realizations to use for the null model
        if 0, then it will only return the similarities,
        defaults to 10000
    repetitionCount: int
        The number of times to repeat the degrees for the null model.
        Only values > 2 are valid, defaults to 2
    minSimilarity: float
        The minimum similarity to consider for the null model
        defaults to 0.0
    idf: str or None
        The type of idf to use for the cosine similarity.
        Can be:
         - None or "none" for no idf
         - "linear" for an idf term of totalFreq/freq
         - "smoothlinear" for an idf term of (totalFreq+1)/(freq+1)
         - "log" for an idf term of log(totalFreq/freq)
         - "smoothlog for an idf term of log((totalFreq)/(freq+1)+1)
        default: None
    IDFWeightsRealizations: int
        The number of realizations to use for estimating the idf weights of the
        null model.
        defaults to 1000
    returnDegreeSimilarities: bool
        Whether to return the degree similarities
        defaults to False
    returnDegreeValues: bool
        Whether to return the degree values
        defaults to False
    showProgress: bool
        Whether to show the progress bar
        defaults to True
    storeAllSimilarities: bool
        Whether to store all similarities in the null model
    batchSize: int
        The batch size to use for the parallel processing
        defaults to 100
    workers: int
        The number of workers to use for the parallel processing
        if 0, then it will not use parallel processing
        if -1, then it will use the number of CPUs
        defaults to -1 

    Returns:
    --------
    returnValue: dict
        A dictionary containing the indexed edges, the similarities, the pvalues and the labels of the nodes.
        If returnDegreeSimilarities is True, then the dictionary will also
        contain the degree similarities.
        If returnDegreeValues is True, then the dictionary will also contain
        the degrees of the nodes in indexed order.
    """
    # print all parameters:
    # print("bipartiteEdges: ",bipartiteEdges)
    # print("scoreType: ",scoreType)
    # print("realizations: ",realizations)
    # print("repetitionCount: ",repetitionCount)
    # print("minSimilarity: ",minSimilarity)
    # print("idf: ",idf)
    # print("IDFWeightsRealizations: ",IDFWeightsRealizations)
    # print("returnDegreeSimilarities: ",returnDegreeSimilarities)
    # print("returnDegreeValues: ",returnDegreeValues)
    # print("showProgress: ",showProgress)
    # print("batchSize: ",batchSize)
    # print("workers: ",workers)
    # print("\n")

    # scoreType can be a list or string, "pvalue", "onlynullmodel"
    # "onlynullmodel" can not be used in list
    
    if(isinstance(scoreType, str)):
        scoreTypes = set([scoreType])
    else:
        scoreTypes = set(scoreType)

    if("onlynullmodel" in scoreTypes):
        if(len(scoreTypes)>1):
            raise ValueError("onlynullmodel can not be used with other score types")


    shouldCalculatePvalues = "pvalue" in scoreTypes
    onlyNullModel = "onlynullmodel" in scoreTypes

    currentTQDM = tqdm if showProgress else _tqdmDummy
    
    if(workers == -1):
        workers = mp.cpu_count()
    
    # reindexing the edges
    # if bipartiteEdges is not nupmy array, then convert it to numpy array
    # if(not isinstance(bipartiteEdges, np.ndarray)):
    #     bipartiteEdges = np.array(bipartiteEdges)
    bipartiteIndexedEdges = np.zeros((len(bipartiteEdges),2), dtype=int)
    # bipartiteEdges is a python list of tuples
    leftIndex2Label = list(set( edge[0] for edge in bipartiteEdges))
    leftLabel2Index = {label: index for index, label in enumerate(leftIndex2Label)}
    rightIndex2Label = list(set( edge[1] for edge in bipartiteEdges))
    rightLabel2Index = {label: index for index, label in enumerate(rightIndex2Label)}
    
    # create indexed edges in a numpy array integers
    bipartiteIndexedEdges[:,0] = [leftLabel2Index[edge[0]] for edge in bipartiteEdges]
    bipartiteIndexedEdges[:,1] = [rightLabel2Index[edge[1]] for edge in bipartiteEdges]

    leftCount = len(leftIndex2Label)
    rightCount = len(rightIndex2Label)


    weights = calculateIDFWeights(bipartiteIndexedEdges, leftCount, rightCount, idf)
        # print("weights: ",len(weights),"edges: ",len(indexedEdges))
    # print("weights:",weights)
    
    similarityDataIndices = None
    similarityDataValues = None
    
    if(not onlyNullModel):
        for _ in currentTQDM(range(1), desc="Calculating original similarity matrix"):
            # similarityDictionary = bipartiteCosineSimilarityMatrixThresholded(bipartiteIndexedEdges,leftCount=leftCount,rightCount=rightCount,threshold=minSimilarity)
            similarityDictionary = fastcosine.bipartiteCosine(
                bipartiteIndexedEdges,
                weights=weights,
                leftCount=leftCount,
                rightCount=rightCount,
                returnDictionary=True,
                updateCallback = "progress" if showProgress else None,
                threshold=minSimilarity,
            )
            # {(4654, 5164): 0.5051814855409226, (7766, 12298): 0.5477225575051662, (10260, 23710): 0.5000000000000001, (12490, 16857): 0.5454545454545455, (15455, 19562): 0.5000000000000001, (15548, 20969): 0.5222329678670936}
        similarityDataIndices = [(fromIndex, toIndex) for (fromIndex, toIndex), _ in similarityDictionary.items()]
        similarityDataValues = np.array([similarity for (_, _), similarity in similarityDictionary.items()])
        del similarityDictionary

    leftDegrees = np.zeros(leftCount, dtype=int)
    rightDegrees = np.zeros(rightCount, dtype=int)

    for indexedEdge in bipartiteIndexedEdges:
        leftDegrees[indexedEdge[0]] += 1
        rightDegrees[indexedEdge[1]] += 1

    # degrees in the similarity matrix
    degreePairsInSimilarity = set()
    if(similarityDataIndices is not None):
        for (fromIndex, toIndex) in similarityDataIndices:
            fromDegree = leftDegrees[fromIndex]
            toDegree = leftDegrees[toIndex]
            degreePairsInSimilarity.add((min(fromDegree, toDegree), max(fromDegree, toDegree)))
    
    if(onlyNullModel):
        # calculate degrees for each node on both sides


        uniqueLeftDegrees = np.unique(leftDegrees)
        # uniqueRightDegrees = np.unique(rightDegrees) # not used

        # Need to repeat at least one time so that we can calculate the similarity
        # between nodes with the same degree
        repeatedUniqueLeftDegrees = np.repeat(uniqueLeftDegrees, repetitionCount)
        repeatedUniqueLeftDegreesIndices = np.repeat(np.arange(len(repeatedUniqueLeftDegrees)), repeatedUniqueLeftDegrees)

        repeatedUniqueLeftDegrees = repeatedUniqueLeftDegrees.astype(int)
        allowedCombinations = None
    else:
        # use degrees from the degreePairsInSimilarity
        uniqueLeftDegrees = np.unique([degreePair[0] for degreePair in degreePairsInSimilarity]+
                                        [degreePair[1] for degreePair in degreePairsInSimilarity])
        # print(uniqueLeftDegrees)
        repeatedUniqueLeftDegrees = np.repeat(uniqueLeftDegrees, repetitionCount)
        # setdtype to int
        repeatedUniqueLeftDegrees = repeatedUniqueLeftDegrees.astype(int)
        # print(repeatedUniqueLeftDegrees)
        repeatedUniqueLeftDegreesIndices = np.repeat(np.arange(len(repeatedUniqueLeftDegrees)), repeatedUniqueLeftDegrees)

        degreeCombinationIndices = combinations(range(len(repeatedUniqueLeftDegrees)), 2)
        allowedCombinations = [degreeEdge for degreeEdge in degreeCombinationIndices if
                                (repeatedUniqueLeftDegrees[degreeEdge[0]],repeatedUniqueLeftDegrees[degreeEdge[1]]) in degreePairsInSimilarity]
    
    # print("\n")
    # print("degreePairsInSimilarity",degreePairsInSimilarity)
    # print("uniqueLeftDegrees", uniqueLeftDegrees)
    # print("repeatedUniqueLeftDegrees", repeatedUniqueLeftDegrees)
    # print("repeatedUniqueLeftDegreesIndices", repeatedUniqueLeftDegreesIndices)
    # print("degreeCombinationIndices", list(combinations(range(len(repeatedUniqueLeftDegrees)), 2)))
    # print("allowedCombinations", allowedCombinations)
    # print("\n")



    degreePair2EdgeIndices = {} 
    for edgeIndex, (fromIndex, toIndex) in enumerate(similarityDataIndices):
        fromDegree = leftDegrees[fromIndex]
        toDegree = leftDegrees[toIndex]
        degreePair = (min(fromDegree, toDegree), max(fromDegree, toDegree))
        if(degreePair not in degreePair2EdgeIndices):
            degreePair2EdgeIndices[degreePair] = []
        degreePair2EdgeIndices[degreePair].append(edgeIndex)


    

    shuffledEdges = bipartiteIndexedEdges.copy()
    
    


    # divide realizations into batches of size batchSize

    degreePair2similarityComplete = {}

    
    similarityPValues = np.zeros(similarityDataValues.shape[0], dtype=int)
    if(realizations>0 and degreePairsInSimilarity):
        estimatedIDFWeights = estimateIDFWeightsShuffled(bipartiteIndexedEdges, leftCount, rightCount, idf, IDFWeightsRealizations, showProgress=showProgress,workers=workers)
        workers = max(1, workers)
        batchRealizations = realizations//batchSize
        batchRealizationsRemainder = realizations%batchSize
        batchRealizationsList = [batchSize]*batchRealizations

        # create streams for the children
        seedSequence = SeedSequence()
        childSeeds = seedSequence.spawn(batchRealizations)

        
        if(batchRealizationsRemainder>0):
            batchRealizationsList.append(batchRealizationsRemainder)

        batchParameters = [(batchRealizations,None,allowedCombinations,repeatedUniqueLeftDegreesIndices,
                            shuffledEdges,estimatedIDFWeights,rightCount,repeatedUniqueLeftDegrees,
                            False,childSeed) for batchRealizations,childSeed in zip(batchRealizationsList,childSeeds)]
        
        # print("Shapes of all parameters")
        # # print("allowedCombinations: ",len(allowedCombinations))
        # print("repeatedUniqueLeftDegreesIndices: ",repeatedUniqueLeftDegreesIndices.shape)
        # print("shuffledEdges: ",shuffledEdges.shape)
        # print("repeatedUniqueLeftDegrees: ",repeatedUniqueLeftDegrees.shape)
        # print("batchParameters: ",len(batchParameters))


        # tracemalloc.start()
        belowSimilarityEdgesCount = np.zeros(similarityDataValues.shape[0], dtype=int)
        totalSimilarityEdgesCount = np.zeros(similarityDataValues.shape[0], dtype=int)

        print("Using workers: ",workers)
        with closing(Pool(workers)) as pool:
        # if True:
            # use imap_unordered to process the batches in parallel
            # also show the progress bar for the batches
            realizationBatchIndex = 0
            for batchDegreePair2Similarity in currentTQDM(
                pool.imap_unordered(_processBatch, batchParameters),
                desc="Null model batches", total=len(batchParameters)):
            # for parameterSet in tqdm(batchParameters, desc="Null model batches", total=len(batchParameters)):
            #     batchDegreePair2Similarity = _processBatch(parameterSet)
                for degreePair, nullModelSimilarities in batchDegreePair2Similarity.items():
                    edgeIndices = degreePair2EdgeIndices[degreePair]
                    edgeSimilarities = similarityDataValues[edgeIndices]
                    nullModelSimilarities = np.array(nullModelSimilarities) # 
                    # print("-----")
                    # print("degreePair",degreePair)
                    # print("edgeIndices",edgeIndices)
                    # print("belowSimilarityEdgesCount",belowSimilarityEdgesCount)
                    # print("totalSimilarityEdgesCount",totalSimilarityEdgesCount)
                    # print("nullModelSimilarities",nullModelSimilarities)
                    # print("edgeSimilarities",edgeSimilarities)
                    # print("(nullModelSimilarities[:,None]<edgeSimilarities[None,:]).shape",(nullModelSimilarities[:,None]<edgeSimilarities[None,:]).shape)
                    # print("np.sum(nullModelSimilarities[:,None]<edgeSimilarities[None,:], axis=0)",np.sum(nullModelSimilarities[:,None]<edgeSimilarities[None,:], axis=0))
                    # print("-----")
                    # need to count the number of null model similarities that are below the original similarity for each edge
                    belowSimilarityEdgesCount[edgeIndices] += np.sum(nullModelSimilarities[:,None]>=edgeSimilarities[None,:], axis=0)
                    totalSimilarityEdgesCount[edgeIndices] += len(nullModelSimilarities)

                    # if(realizationBatchIndex%100==0):
                    #     snapshot = tracemalloc.take_snapshot()
                    #     top_stats = snapshot.statistics('lineno')
                    #     print("==========MEMORY==========")
                    #     print("[ Top 10 Memory Consumers ]")
                    #     for stat in top_stats[:10]:
                    #         print(stat)
                    #     print("")
                    
                    realizationBatchIndex+=1
                    if returnDegreeSimilarities:
                        print("Returning Degree Similarities")
                        if(degreePair not in degreePair2similarityComplete):
                            degreePair2similarityComplete[degreePair] = []
                        degreePair2similarityComplete[degreePair].extend(nullModelSimilarities)
                    del nullModelSimilarities
        similarityPValues = (belowSimilarityEdgesCount+1)/(totalSimilarityEdgesCount+1)
    # print({degreePair: len(similarities) for degreePair, similarities in degreePair2similarity.items()})

    if degreePair2similarityComplete:
        degreePair2similarityComplete = {degreePair: np.array(similarities) for degreePair, similarities in degreePair2similarityComplete.items()}

        if(onlyNullModel):
            returnValues = {}
            returnValues["nullmodelDegreePairsSimilarities"] = degreePair2similarityComplete
            return returnValues
    
    # tracemalloc.stop()
    # # apply arctanh to the similarities
    # if(fisherTransform):
    #     degreePair2similarity = {degreePair: np.arctanh(similarities) for degreePair, similarities in degreePair2similarity.items()}
    #     # originalSimilarityMatrix.data = np.arctanh(originalSimilarityMatrix.data)
    #     # apply np.arctanh(similarity) on similarityData
    #     similarityData = [(fromIndex, toIndex, np.arctanh(similarity)) for fromIndex, toIndex, similarity in similarityData]
    


    # get >0 indices: fromIndex, toIndex, similarity
    # remember that  originalSimilarityMatrix is a symmetric matrix sparse CRS matrix

    # fromIndices, toIndices = originalSimilarityMatrix.nonzero()
    # similarities = originalSimilarityMatrix.data
    # aboveThresholdIndices = similarities>minSimilarity
    # fromIndices = fromIndices[aboveThresholdIndices]
    # toIndices = toIndices[aboveThresholdIndices]
    # similarities = similarities[aboveThresholdIndices]
    # originalSimilarityEdges = zip(fromIndices, toIndices, similarities)
    # originalSimilarityEdges = [(fromIndex, toIndex, similarity) for fromIndex, toIndex, similarity in originalSimilarityEdges if similarity>minSimilarity]


    # FIXME: This should be faster, but requires fixing the code
    # mask = originalSimilarityMatrix.data > minSimilarity
    # dataThresholded = originalSimilarityMatrix.data[mask]
    # fromIndices = originalSimilarityMatrix.indices[mask]

    # toPointers = originalSimilarityMatrix.indptr
    # toIndices = np.zeros(len(dataThresholded), dtype=int)
    # startIndex = 0
    # for row in currentTQDM(range(len(toPointers) - 1)):
    #     endIndex = toPointers[row + 1]
    #     toIndices[startIndex:endIndex] = row
    #     startIndex = endIndex
    # toIndices = toIndices[mask]

    # originalSimilarityEdges = zip(fromIndices, toIndices, dataThresholded)

    

    returnValues = {}
    resultsIndexedEdges = []
    resultSimilarities = []
    for (fromIndex, toIndex), originalSimilarity in zip(similarityDataIndices, similarityDataValues):
        resultsIndexedEdges.append((fromIndex, toIndex))
        resultSimilarities.append(originalSimilarity)
    returnValues["indexedEdges"] = resultsIndexedEdges
    returnValues["similarities"] = resultSimilarities

    # calculate quantiles
    if("quantile" in scoreTypes):
        # will calculate the quantiles, i.e., how many similarities are above each similarity value
        # (for each pair in resultSimilarities)
        pairsCount = leftCount*(leftCount-1)//2
        rank = rankdata(resultSimilarities, method="max")
        returnValues["quantiles"] = (pairsCount-len(rank)+rank)/pairsCount

    if("quantilesimilarity" in scoreTypes):
        # will calculate the quantiles, i.e., how many similarities are above each similarity value
        # (for each pair in resultSimilarities)
        pairsCount = leftCount*(leftCount-1)//2
        rank = rankdata(resultSimilarities, method="max")
        returnValues["quantilesimilarities"] = ((pairsCount-len(rank)+rank)/pairsCount)*resultSimilarities

        # print(f"LeftCount: {pairsCount} len(resultSimilarities): {len(resultSimilarities)}")
    if(returnDegreeValues):
        returnValues["degrees"] = leftDegrees
    returnValues["labels"] = leftIndex2Label

    if(shouldCalculatePvalues and realizations>0):
        returnValues["pvalues"] = similarityPValues
    
    if(returnDegreeSimilarities):
        returnValues["nullmodelDegreePairsSimilarities"] = degreePair2similarityComplete
    return returnValues


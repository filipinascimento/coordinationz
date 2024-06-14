
# from __future__ import annotations

import numpy as np
from tqdm.auto import tqdm
from scipy.sparse import csr_matrix, lil_matrix
from itertools import combinations
import multiprocessing as mp
from multiprocessing import Pool


def compute_cosine_similarity_thresholded(matrix, threshold, showProgress=True):
    results = {}
    fromNormalizations = np.sqrt(matrix.power(2).sum(axis=1))

    indicesRange = range(matrix.shape[0])
    if(showProgress):
        indicesRange = tqdm(indicesRange, desc="Calculating cosine similarity", leave=False)

    # Iterate over each row to calculate cosine similarity with all other rows
    for fromIndex in indicesRange:
        if fromNormalizations[fromIndex] == 0:
            continue  # Skip rows with no connections to avoid division by zero
        dotProducts = matrix[fromIndex].dot(matrix.transpose())
        similarities = dotProducts / (fromNormalizations[fromIndex] * fromNormalizations.transpose())
        # similarities is a scipy.sparse._coo.coo_matrix
        similaritiesValues = similarities.data
        toIndices = similarities.col
        if(threshold>0):
            mask = similaritiesValues > threshold
            toIndices = toIndices[mask]
            similaritiesValues = similaritiesValues[mask]
        for edgeIndex, toIndex in enumerate(toIndices):
            if(toIndex>fromIndex):
                results[(fromIndex,toIndex)] = similaritiesValues[edgeIndex]
    return results


def bipartiteCosineSimilarityMatrixThresholded(indexedEdges, leftCount=None, rightCount=None, threshold=0.0):
    """
    Calculate the cosine similarity matrix of a bipartite graph
    given the indexed edges of the bipartite graph.

    Parameters:
    ----------
    indexedEdges: np.ndarray
        The indexed edges of the bipartite graph
    leftCount: int
        The number of nodes on the left side of the bipartite graph
    rightCount: int
        The number of nodes on the right side of the bipartite graph
    threshold: float
        The threshold to use for the cosine similarity matrix
    Returns:
    --------
    cosineSimilarities: dict
        A dictionary containing the cosine similarities among the left nodes
    """
    data = np.ones(len(indexedEdges))
    fromIndices = indexedEdges[:, 0]
    toIndices = indexedEdges[:, 1]

    if(leftCount is None):
        leftCount = np.max(fromIndices) + 1
    if(rightCount is None):
        rightCount = np.max(toIndices) + 1

    adjacencyMatrix = csr_matrix((data, (fromIndices, toIndices)), shape=(leftCount, rightCount))


    return compute_cosine_similarity_thresholded(adjacencyMatrix, threshold)




def bipartiteCosineSimilarityMatrix(indexedEdges, leftCount=None, rightCount=None):
    """
    Calculate the cosine similarity matrix of a bipartite graph
    given the indexed edges of the bipartite graph.

    Parameters:
    ----------
    indexedEdges: np.ndarray
        The indexed edges of the bipartite graph
    leftCount: int
        The number of nodes on the left side of the bipartite graph
    rightCount: int
        The number of nodes on the right side of the bipartite graph

    Returns:
    --------
    cosineMatrix: csr_matrix
        The cosine similarity matrix of the bipartite graph in CSR format
    """
    data = np.ones(len(indexedEdges))
    fromIndices = indexedEdges[:, 0]
    toIndices = indexedEdges[:, 1]

    if(leftCount is None):
        leftCount = np.max(fromIndices) + 1
    if(rightCount is None):
        rightCount = np.max(toIndices) + 1

    adjacencyMatrix = csr_matrix((data, (fromIndices, toIndices)), shape=(leftCount, rightCount))
    fromNormalized = np.sqrt(adjacencyMatrix.power(2).sum(axis=1))
    normalizedMatrix = adjacencyMatrix.multiply(1 / fromNormalized)

    cosineMatrix = normalizedMatrix.dot(normalizedMatrix.transpose())
    cosineMatrix = cosineMatrix.tocsr()
    return cosineMatrix



def _tqdmDummy(*args, **kwargs):
    return args[0]

def _processBatch(parameters):
        batchRealizations,currentTQDM, \
        allowedCombinations, \
        repeatedUniqueLeftDegreesIndices, \
        shuffledEdges,rightCount, \
        repeatedUniqueLeftDegrees, \
        isSingleBatch = parameters \
        
        if allowedCombinations is None:
            allowedCombinations = combinations(repeatedUniqueLeftDegreesIndices, 2)

        realizationsRange = range(batchRealizations)
        if(isSingleBatch):
            realizationsRange = currentTQDM(realizationsRange, desc="Null model realizations")
        
        batchDegreePair2similarity = {}
        batchReducedShuffledEdges = np.zeros((len(repeatedUniqueLeftDegreesIndices), 2), dtype=int)
        batchReducedShuffledEdges[:,0] = repeatedUniqueLeftDegreesIndices

        for _ in realizationsRange:
            batchReducedShuffledEdges[:,1] = np.random.choice(shuffledEdges[:,1], len(batchReducedShuffledEdges), replace=False)
            
            # FIXME: Incude the option to choose the similarity metric
            # or to send a custom function
            similarityMatrix = bipartiteCosineSimilarityMatrix(batchReducedShuffledEdges,rightCount=rightCount)
            
            for fromIndex, toIndex in allowedCombinations:
                if(fromIndex==toIndex):
                    continue
                similarity = similarityMatrix[fromIndex, toIndex]
                fromDegree = repeatedUniqueLeftDegrees[fromIndex]
                toDegree = repeatedUniqueLeftDegrees[toIndex]
                degreeEdge = (min(fromDegree, toDegree), max(fromDegree, toDegree))
                if(degreeEdge not in batchDegreePair2similarity):
                    batchDegreePair2similarity[degreeEdge] = []
                batchDegreePair2similarity[degreeEdge].append(similarity)
        return batchDegreePair2similarity


def bipartiteNullModelSimilarity(
        bipartiteEdges,
        scoreType="zscore", # "zscore", "pvalue", "pvalue-quantized", "onlynullmodel" are valid
        pvaluesQuantized = None, # can be a number or a list of pvalues, defaults to 2*orders of magnitude(realizations)
        fisherTransform = False,
        realizations = 10000,
        repetitionCount = 2,
        minSimilarity = 0.0,
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
    scoreType: "pvalue", "pvalue-quantized" or "zscore"
        The type of score to calculate, can be "pvalue", "pvalue-quantized", "zscore"
        or "onlynullmodel",
        if "onlynullmodel" is selected, then the function will only return the
        degree similarities.
        defaults to "zscore"
    pvaluesQuantized: int or list
        The number of quantiles to use for the pvalues, defaults to the orders
        of magnitude of the realizations,
        or a list of pvalues to use for the quantiles
    fisherTransform: bool
        Whether to apply the Fisher transformation to the similarities
        defaults to False
    realizations: int
        The number of realizations to use for the null model
        defaults to 10000
    repetitionCount: int
        The number of times to repeat the degrees for the null model.
        Only values > 2 are valid, defaults to 2
    minSimilarity: float
        The minimum similarity to consider for the null model
        defaults to 0.0
    returnDegreeSimilarities: bool
        Whether to return the degree similarities
        defaults to False
    returnDegreeValues: bool
        Whether to return the degree values
        defaults to False
    showProgress: bool
        Whether to show the progress bar
        defaults to True
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
        A dictionary containing the indexed edges, the similarities, the pvalues
        or zscores and the labels of the nodes.
        If returnDegreeSimilarities is True, then the dictionary will also
        contain the degree similarities.
        If returnDegreeValues is True, then the dictionary will also contain
        the degrees of the nodes in indexed order.
    """

    currentTQDM = tqdm if showProgress else _tqdmDummy
    if (scoreType == "pvalue-quantized"):
        # if pvalues is a number (any type), then we will use it as the number of quantiles
        if(pvaluesQuantized is None):
            pvaluesQuantized = 2*np.log10(realizations)+1
        if(isinstance(pvaluesQuantized, (int, float))):
            pvaluesCount = int(pvaluesQuantized)
            # Predefine the pvalues based on the number of realizations
            # we want to have 10 quantiles 
            # distribute over the log space based on the number of realizations
            realizationMagnitude = np.log10(realizations)
            pvaluesQuantized = np.logspace(-realizationMagnitude, 0, pvaluesCount)
        else:
            pvaluesQuantized = list(pvaluesQuantized)
            # sort
            pvaluesQuantized.sort()
    
    if(workers == -1):
        workers = mp.cpu_count()
    
    # reindexing the edges
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

    similarityDictionary = None
    if(scoreType != "onlynullmodel"):
        for _ in currentTQDM(range(1), desc="Calculating original similarity matrix"):
            similarityDictionary = bipartiteCosineSimilarityMatrixThresholded(bipartiteIndexedEdges,leftCount=leftCount,rightCount=rightCount,threshold=minSimilarity)

    

    # calculate degrees for each node on both sides
    leftDegrees = np.zeros(leftCount, dtype=int)
    rightDegrees = np.zeros(rightCount, dtype=int)

    for indexedEdge in bipartiteIndexedEdges:
        leftDegrees[indexedEdge[0]] += 1
        rightDegrees[indexedEdge[1]] += 1

    uniqueLeftDegrees = np.unique(leftDegrees)
    # uniqueRightDegrees = np.unique(rightDegrees) # not used

    shuffledEdges = bipartiteIndexedEdges.copy()

    # Need to repeat at least one time so that we can calculate the similarity
    # between nodes with the same degree
    repeatedUniqueLeftDegrees = np.repeat(uniqueLeftDegrees, repetitionCount)
    repeatedUniqueLeftDegreesIndices = np.repeat(np.arange(len(repeatedUniqueLeftDegrees)), repeatedUniqueLeftDegrees)

    
    degreePairsInSimilarity = set()
    if(similarityDictionary is not None):
        for (fromIndex, toIndex) in similarityDictionary.keys():
            fromDegree = leftDegrees[fromIndex]
            toDegree = leftDegrees[toIndex]
            degreePairsInSimilarity.add((min(fromDegree, toDegree), max(fromDegree, toDegree)))

    print(repeatedUniqueLeftDegrees)
    print(repeatedUniqueLeftDegreesIndices)
    allowedCombinations = None


    # divide realizations into batches of size batchSize

    degreePair2similarity = {}

    
    if(workers <= 1):
        degreePair2similarity = _processBatch((realizations,
                    currentTQDM, allowedCombinations, repeatedUniqueLeftDegreesIndices,
                    shuffledEdges,rightCount,repeatedUniqueLeftDegrees,
                    True))
    else:
        batchRealizations = realizations//batchSize
        batchRealizationsRemainder = realizations%batchSize
        batchRealizationsList = [batchSize]*batchRealizations
        if(batchRealizationsRemainder>0):
            batchRealizationsList.append(batchRealizationsRemainder)
        batchParameters = [(batchRealizations,None,allowedCombinations,repeatedUniqueLeftDegreesIndices,
                            shuffledEdges,rightCount,repeatedUniqueLeftDegrees,
                            False) for batchRealizations in batchRealizationsList]
        
        with Pool(workers) as pool:
            # use imap_unordered to process the batches in parallel
            # also show the progress bar for the batches
            for batchDegreePair2Similarity in currentTQDM(
                pool.imap_unordered(_processBatch, batchParameters),
                desc="Null model batches", total=len(batchParameters)):
                for degreePair, similarities in currentTQDM(batchDegreePair2Similarity.items(),desc="Internal process",leave=False):
                    if(degreePair not in degreePair2similarity):
                        degreePair2similarity[degreePair] = []
                    degreePair2similarity[degreePair].extend(similarities)
    
    degreePair2similarity = {degreePair: np.array(similarities) for degreePair, similarities in degreePair2similarity.items()}

    if(scoreType == "onlynullmodel"):
        returnValues = {}
        returnValues["nullmodelDegreePairsSimilarities"] = degreePair2similarity
        return returnValues
    
    
    # apply arctanh to the similarities
    if(fisherTransform):
        degreePair2similarity = {degreePair: np.arctanh(similarities) for degreePair, similarities in degreePair2similarity.items()}
        # originalSimilarityMatrix.data = np.arctanh(originalSimilarityMatrix.data)
        for edge,similarity in similarityDictionary.items():
            similarityDictionary[edge] = np.arctanh(similarity)
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
    if(scoreType == "pvalue"):
        # FIXME: This need to be optimized!
        # Some potential solutions:
        # 1. Use parallel processing
        # 2. Use cython, numba or C extensions
        # 3. Keep a cache of combinations of similarity and degree pairs
        resultsIndexedEdges = []
        resultSimilarities = []
        resultPValues = []
        # for fromIndex, toIndex, originalSimilarity in currentTQDM(originalSimilarityEdges, desc="Calculating pvalues",leave=False,total=len(fromIndices)):
        for (fromIndex, toIndex), originalSimilarity in currentTQDM(similarityDictionary.items(), desc="Calculating pvalues",leave=False,total=len(similarityDictionary)):
            fromDegree = leftDegrees[fromIndex]
            toDegree = leftDegrees[toIndex]
            degreeEdge = (min(fromDegree, toDegree), max(fromDegree, toDegree))
            similarities = degreePair2similarity[degreeEdge]
            totalSimilarities = len(similarities)
            similaritiesAbove = np.sum(similarities>=originalSimilarity)
            pvalue = similaritiesAbove/totalSimilarities
            # pvalue = 1.0-percentileofscore(similarities, originalSimilarity, kind="weak")
            # pvalues.append((fromIndex, toIndex, originalSimilarity, pvalue))
            resultsIndexedEdges.append((fromIndex, toIndex))
            resultSimilarities.append(originalSimilarity)
            resultPValues.append(pvalue)
        returnValues["indexedEdges"] = resultsIndexedEdges
        returnValues["similarities"] = resultSimilarities
        returnValues["pvalues"] = resultPValues
    elif(scoreType == "pvalue-quantized"):
        resultsIndexedEdges = []
        resultSimilarities = []
        resultPValues = []
        precalculatedThresholds = []
        for pvalue in pvaluesQuantized:
            precalculatedThresholds.append({degreePair: np.quantile(similarities, 1.0-pvalue) for degreePair, similarities in degreePair2similarity.items()})
        # for fromIndex, toIndex, originalSimilarity in currentTQDM(originalSimilarityEdges, desc="Calculating pvalues",leave=False,total=len(fromIndices)):
        for (fromIndex, toIndex), originalSimilarity in currentTQDM(similarityDictionary.items(), desc="Calculating pvalues",leave=False,total=len(similarityDictionary)):
            fromDegree = leftDegrees[fromIndex]
            toDegree = leftDegrees[toIndex]
            degreeEdge = (min(fromDegree, toDegree), max(fromDegree, toDegree))
            similarities = degreePair2similarity[degreeEdge]
            # use the precalculated thresholds
            pvalue = 1.0
            for index, threshold in enumerate(precalculatedThresholds): 
                if(originalSimilarity>threshold[degreeEdge]):
                    pvalue = pvaluesQuantized[index]
                    break
            resultsIndexedEdges.append((fromIndex, toIndex))
            resultSimilarities.append(originalSimilarity)
            resultPValues.append(pvalue)
        returnValues["indexedEdges"] = resultsIndexedEdges
        returnValues["similarities"] = resultSimilarities
        returnValues["pvalues"] = resultPValues
            
    elif(scoreType == "zscore"):
        resultsIndexedEdges = []
        resultSimilarities = []
        resultZScores = []
        precalculatedMeans = {degreePair: np.nanmean(similarities) for degreePair, similarities in degreePair2similarity.items()}
        precalculatedSTDs = {degreePair: np.nanstd(similarities) for degreePair, similarities in degreePair2similarity.items()}
        # for fromIndex, toIndex, originalSimilarity in currentTQDM(originalSimilarityEdges, desc="Calculating zscores",leave=False,total=len(fromIndices)):
        for (fromIndex, toIndex), originalSimilarity in currentTQDM(similarityDictionary.items(), desc="Calculating pvalues",leave=False,total=len(similarityDictionary)):
            fromDegree = leftDegrees[fromIndex]
            toDegree = leftDegrees[toIndex]
            degreeEdge = (min(fromDegree, toDegree), max(fromDegree, toDegree))
            mean = precalculatedMeans[degreeEdge]
            std = precalculatedSTDs[degreeEdge]
            zscore = (originalSimilarity-mean)/std
            resultsIndexedEdges.append((fromIndex, toIndex))
            resultSimilarities.append(originalSimilarity)
            resultZScores.append(zscore)
        returnValues["indexedEdges"] = resultsIndexedEdges
        returnValues["similarities"] = resultSimilarities
        returnValues["zscores"] = resultZScores
    else:
        raise ValueError("scoreType must be either 'pvalue','pvalue-quantized' or 'zscore'")

    returnValues["labels"] = leftIndex2Label
    if(returnDegreeSimilarities):
        returnValues["nullmodelDegreePairsSimilarities"] = degreePair2similarity
    if(returnDegreeValues):
        returnValues["degrees"] = leftDegrees
    return returnValues


from . import fastcosine_core
import numpy as np
from tqdm.auto import tqdm

_cosine_core = fastcosine_core.cosine

def _make_pbar(desc=None):
    pbar = None
    def inner(current,total):
        nonlocal pbar
        if(pbar is None):
            pbar= tqdm(total=total, desc=desc)
        pbar.update(current - pbar.n)
    return inner


def bipartiteCosine(indexedEdges,
                    weights=None,
                    leftCount=None,
                    rightCount=None,
                    # idf=None, # None, "none", "linear", "smooth", "log", "log1p",
                    threshold=0.0,
                    leftEdges=None,
                    returnDictionary = False,
                    updateCallback=None,
                    callbackUpdateInterval=0):
    """
    Calculate the cosine similarity matrix of a bipartite graph
    given the indexed edges of the bipartite graph.
    
    Parameters:
    ----------
    indexedEdges: np.ndarray
        The indexed edges of the bipartite graph
    # weights: np.ndarray
    #     The weights of the edges in the bipartite graph
    #     default: None (all weights are 1)
    leftCount: int
        The number of nodes on the left side of the bipartite graph
        default: maximum index of the indexedEdges[:, 0] + 1
    rightCount: int
        The number of nodes on the right side of the bipartite graph
        default: maximum index of the indexedEdges[:, 1] + 1
    threshold: float
        The threshold to use for the cosine similarity matrix
        default: 0.0
    leftEdges: np.ndarray or list of 2 element tuples (optional)
        Will only calculate the cosine similarity of the leftEdges pairs
        default: None (calculate all pairs)
    idf: str or None
        The type of idf to use for the cosine similarity.
        Can be:
         - None or "none" for no idf
         - "linear" for an idf term of totalFreq/freq
         - "smoothlinear" for an idf term of (totalFreq+1)/(freq+1)
         - "log" for an idf term of log(totalFreq/freq)
         - "smoothlog for an idf term of log((totalFreq)/(freq+1)+1)
        default: None

    returnDictionary: bool
        If True, return a dictionary of the cosine similarities
        default: False
    updateCallback: str or callable
        If "progress", show a progress bar
        If callable, call the function with two int parameters: current and total
        default: None
    callbackUpdateInterval: int
        The interval at which to update the callback
        default: 0 (update every 10* leftCount iterations
    Returns:
    --------
    tuple: (edges, similarities) if returnDictionary is False
        edges: np.ndarray
            The indexed edges of the cosine similarities above the threshold
        similarities: np.ndarray
            The cosine similarities
    dict: if returnDictionary is True
        A dictionary containing the cosine similarities among the left nodes
    """
    edgesCount = len(indexedEdges)
    indexedEdges = np.array(indexedEdges)

    if(leftCount is None):
        leftCount = np.max(indexedEdges[:, 0]) + 1
    if(rightCount is None):
        rightCount = np.max(indexedEdges[:, 1]) + 1

    if(updateCallback == "progress"):
        updateCallback = _make_pbar("Cosine Similarity")
    
    encodedEdges = np.zeros(edgesCount, dtype=np.int64)
    # edge = left * rightCount + right
    encodedEdges = indexedEdges[:, 0] * rightCount + indexedEdges[:, 1]

    indices = np.argsort(encodedEdges)
    encodedEdges = encodedEdges[indices]
    decodedEdges = np.zeros((edgesCount, 2), dtype=np.int64)
    decodedEdges[:, 0] = encodedEdges // rightCount
    decodedEdges[:, 1] = encodedEdges % rightCount

    if(weights is not None):
        weights = np.array(weights)[indices]


    return _cosine_core(decodedEdges,
                        weights=weights,
                        leftCount=leftCount,
                        rightCount=rightCount,
                        threshold=threshold,
                        leftEdges=leftEdges,
                        returnDictionary=returnDictionary,
                        updateCallback=updateCallback,
                        callbackUpdateInterval=callbackUpdateInterval
                        )









def _slowCosineThresholded(matrix, threshold, showProgress=True):
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


def bipartiteSlowCosineThresholded(indexedEdges, leftCount=None, rightCount=None, threshold=0.0):
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
    # import csr_matrix
    from scipy.sparse import csr_matrix
    data = np.ones(len(indexedEdges))
    fromIndices = indexedEdges[:, 0]
    toIndices = indexedEdges[:, 1]

    if(leftCount is None):
        leftCount = np.max(fromIndices) + 1
    if(rightCount is None):
        rightCount = np.max(toIndices) + 1

    adjacencyMatrix = csr_matrix((data, (fromIndices, toIndices)), shape=(leftCount, rightCount))

    return _slowCosineThresholded(adjacencyMatrix, threshold)



def bipartiteSlowCosine(indexedEdges, leftCount=None, rightCount=None):
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
    from scipy.sparse import csr_matrix
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
    return cosineMatrix


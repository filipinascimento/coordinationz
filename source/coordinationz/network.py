
import numpy as np
from tqdm.auto import tqdm

def dummyTQDM(*args, **kwargs):
    return args[0]

def createNetworkFromNullModelOutput(nullModelOutput,
                                     similarityThreshold = 0.0,
                                     zscoreThreshold = 0.0,
                                     pvalueThreshold = 1.0,
                                     quantileThreshold = 0.0,
                                     useZscoreWeights = False,
                                     usePValueWeights = False,
                                     useQuantileWeights = False,
                                     showProgress = True):
    """
    Creates a network from the null model output

    Parameters:
    -----------
    nullModelOutput: dict
        The null model output dictionary
    similarityThreshold: float
        The similarity threshold to use for the network
    zscoreThreshold: float
        The zscore threshold to use for the network
    pvalueThreshold: float
        The pvalue threshold to use for the network
    showProgress: bool
        If True, show a progress bar
        default: True
    
    Returns:
    --------
    igraph.Graph
        The network created from the null model output
    """

    import igraph as ig

    if(showProgress):
        progressbar = tqdm(total = 5)
        progressbar.set_description("Processing edges")

    edges = np.array(nullModelOutput["indexedEdges"])
    vertexCount = len(nullModelOutput["labels"])

    if(showProgress):
        progressbar.update(1)
        progressbar.set_description("Processing labels")
    
    vertexAttributes = {
        "Label": nullModelOutput["labels"]
    }

    if(showProgress):
        progressbar.update(1)
        progressbar.set_description("Setting edge attributes")
    
    edgeAttributes = {}
    edgeAttributes["weight"] = np.array(nullModelOutput["similarities"])

    if("zscore" in nullModelOutput and useZscoreWeights):
        edgeAttributes["weight"] = np.array(nullModelOutput["zscores"])
        # filter nans and set infs to 10
        edgeAttributes["weight"] = np.nan_to_num(edgeAttributes["weight"], nan=0.0, posinf=10.0, neginf=-10.0)
    
    if("pvalues" in nullModelOutput and usePValueWeights):
        edgeAttributes["weight"] = 1.0-np.array(nullModelOutput["pvalues"])
        edgeAttributes["weight"] = np.nan_to_num(edgeAttributes["weight"], nan=1.0, posinf=1.0, neginf=1.0)

    if("quantiles" in nullModelOutput and useQuantileWeights):
        edgeAttributes["weight"] = np.array(nullModelOutput["quantiles"])
        edgeAttributes["weight"] = np.nan_to_num(edgeAttributes["quantiles"], nan=0.0, posinf=1.0, neginf=0.0)


    if("zscores" in nullModelOutput):
        edgeAttributes["zscore"] = np.array(nullModelOutput["zscores"])
    if("pvalues" in nullModelOutput):
        edgeAttributes["pvalue"] = np.array(nullModelOutput["pvalues"])
    if("quantiles" in nullModelOutput):
        edgeAttributes["quantile"] = np.array(nullModelOutput["quantiles"])
    if("degrees" in nullModelOutput):
        vertexAttributes["left_degree"] = np.array(nullModelOutput["degrees"])

    if(showProgress):
        progressbar.update(1)
        progressbar.set_description("Applying similarity filters")
    
    if(similarityThreshold > 0.0 or zscoreThreshold > 0.0 or pvalueThreshold < 1.0):
        edgesMask = np.ones(len(edges), dtype=bool)
        if(similarityThreshold > 0.0):
            edgesMask *= edgeAttributes["weight"] > similarityThreshold
        if(zscoreThreshold > 0.0 and "zscores" in nullModelOutput):
            edgesMask *= edgeAttributes["zscore"] > zscoreThreshold
        if(pvalueThreshold < 1.0 and "pvalues" in nullModelOutput):
            # if quantized, use <=, otherwise use <
            if(nullModelOutput["pvaluesQuantized"]):
                edgesMask *= edgeAttributes["pvalue"] <= pvalueThreshold
            else:
                edgesMask *= edgeAttributes["pvalue"] < pvalueThreshold
        if(quantileThreshold > 0.0 and "quantiles" in nullModelOutput):
            edgesMask *= edgeAttributes["quantile"] > quantileThreshold

        edges = edges[edgesMask, :]
        edgeAttributes["weight"] = edgeAttributes["weight"][edgesMask]
        if("zscores" in nullModelOutput):
            edgeAttributes["zscore"] = edgeAttributes["zscore"][edgesMask]
        if("pvalues" in nullModelOutput):
            edgeAttributes["pvalue"] = edgeAttributes["pvalue"][edgesMask]
        if("quantiles" in nullModelOutput):
            edgeAttributes["quantile"] = edgeAttributes["quantile"][edgesMask]

    if(showProgress):
        progressbar.update(1)
        progressbar.set_description("Creating network")

    g = ig.Graph(
        vertexCount,
        edges,
        directed = False,
        vertex_attrs = vertexAttributes,
        edge_attrs = edgeAttributes
    )

    if(showProgress):
        progressbar.update(1)
        progressbar.set_description("Network ready")
        
    
    return g



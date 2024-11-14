
import numpy as np
from tqdm.auto import tqdm
import warnings
import pandas as pd

def dummyTQDM(*args, **kwargs):
    return args[0]

def createNetworkFromNullModelOutput(nullModelOutput,
                                     similarityThreshold = 0.0,
                                     pvalueThreshold = 1.0,
                                     quantileThreshold = 0.0,
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
    
    if("pvalues" in nullModelOutput and usePValueWeights):
        edgeAttributes["weight"] = 1.0-np.array(nullModelOutput["pvalues"])
        edgeAttributes["weight"] = np.nan_to_num(edgeAttributes["weight"], nan=1.0, posinf=1.0, neginf=1.0)

    if("quantiles" in nullModelOutput and useQuantileWeights):
        edgeAttributes["weight"] = np.array(nullModelOutput["quantiles"])
        edgeAttributes["weight"] = np.nan_to_num(edgeAttributes["quantiles"], nan=0.0, posinf=1.0, neginf=0.0)


    if("pvalues" in nullModelOutput):
        edgeAttributes["pvalue"] = np.array(nullModelOutput["pvalues"])
    if("quantiles" in nullModelOutput):
        edgeAttributes["quantile"] = np.array(nullModelOutput["quantiles"])
    if("degrees" in nullModelOutput):
        vertexAttributes["left_degree"] = np.array(nullModelOutput["degrees"])

    if(showProgress):
        progressbar.update(1)
        progressbar.set_description("Applying similarity filters")
    
    if(similarityThreshold > 0.0 or pvalueThreshold < 1.0):
        edgesMask = np.ones(len(edges), dtype=bool)
        if(similarityThreshold > 0.0):
            edgesMask *= edgeAttributes["weight"] > similarityThreshold
        if(pvalueThreshold < 1.0 and "pvalues" in nullModelOutput):
            edgesMask *= edgeAttributes["pvalue"] < pvalueThreshold
        if(quantileThreshold > 0.0 and "quantiles" in nullModelOutput):
            edgesMask *= edgeAttributes["quantile"] > quantileThreshold

        edges = edges[edgesMask, :]
        edgeAttributes["weight"] = edgeAttributes["weight"][edgesMask]
        if("pvalues" in nullModelOutput):
            edgeAttributes["pvalue"] = edgeAttributes["pvalue"][edgesMask]
        if("quantiles" in nullModelOutput):
            edgeAttributes["quantile"] = edgeAttributes["quantile"][edgesMask]

    if(showProgress):
        progressbar.update(1)
        progressbar.set_description("Creating network")

    if edges is None or len(edges) == 0:
        warnings.warn("No edges found in the network... returning empty graph")
        edges=None
    
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



def removeSingletons(g):
    """
    Removes singleton nodes from the graph

    Parameters:
    -----------
    g: igraph.Graph
        The graph to remove singleton nodes from

    Returns:
    --------
    igraph.Graph
        The graph with singleton nodes removed
    """

    gCopy = g.copy()
    gCopy.delete_vertices(gCopy.vs.select(_degree = 0))
    return gCopy

def thresholdNetwork(g,thresholds):
    """
    Thresholds a network based on an attribute

    Parameters:
    -----------
    g: igraph.Graph
        The graph to threshold
    threshold: dict
        A dictionary with the threshold attribute and the threshold value

    Returns:
    --------
    igraph.Graph
        The thresholded graph
    """

    gThresholded = g.copy()
    mask = np.ones(gThresholded.ecount(),dtype=bool)
    for thresholdAttribute, threshold in thresholds.items():
        if(thresholdAttribute not in set(gThresholded.es.attributes())):
            print(f"Attribute {thresholdAttribute} not found in the graph")
            continue
        if(thresholdAttribute=="pvalue"):
            attributeArray = np.array(gThresholded.es["pvalue"])
            mask &= attributeArray < threshold
        else:
            attributeArray = np.array(gThresholded.es[thresholdAttribute])
            mask &= attributeArray > threshold
    gThresholded.delete_edges(np.where(mask == False)[0])
    # remove degree 0 nodes
    gThresholded.delete_vertices(gThresholded.vs.select(_degree=0))
    return gThresholded


def getNetworkTables(gForTable, currentNodes):
    # save two tables, one with edges with all edge attributes source, target, ... *all edge attributes*
    nodesTable = pd.DataFrame(gForTable.vs["Label"], columns=["user_id"])
    for key in gForTable.vertex_attributes():
        if(key!="Label"):
            nodesTable[key] = gForTable.vs[key]
    # include degree as an attribute
    nodesTable["degree"] = gForTable.degree()
    # include strength as an attribute
    nodesTable["strength"] = gForTable.strength(weights="weight")
    # add the missing nodes from currentNodes
    missingNodes = set(currentNodes) - set(nodesTable["user_id"])
    if(len(missingNodes)>0):
        missingNodesTable = pd.DataFrame(list(missingNodes), columns=["user_id"])
        nodesTable = pd.concat([nodesTable,missingNodesTable], ignore_index=True)
    # fill blanks for degree and strength with zeros
    nodesTable["degree"] = nodesTable["degree"].fillna(0)
    nodesTable["strength"] = nodesTable["strength"].fillna(0)

    networkIndex2UserID = gForTable.vs["Label"]
    userIDEdges = [(networkIndex2UserID[source],networkIndex2UserID[target]) for source,target in gForTable.get_edgelist()]
    edgesTable = pd.DataFrame(userIDEdges, columns=["source_id","target_id"])
    for key in gForTable.edge_attributes():
        edgesTable[key] = gForTable.es[key]
    return {"nodes":nodesTable,"edges":edgesTable}
    # save a table with the nodes and their attributes

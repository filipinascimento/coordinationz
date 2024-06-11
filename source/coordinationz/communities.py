import igraph as ig
import numpy as np
import math
from collections import Counter
from tqdm.auto import tqdm

def getNetworksWithCommunities(g,
                             thresholdAttribute = "quantile",
                             thresholds = [0.99,0.999]):
    networks = {}
    for threshold in thresholds:
        gThresholded = g.copy()
        mask = np.ones(gThresholded.ecount(),dtype=bool)
        if(thresholdAttribute=="pvalue"):
            attributeArray = np.array(gThresholded.es["pvalue"])
            mask &= attributeArray < threshold
        else:
            attributeArray = np.array(gThresholded.es[thresholdAttribute])
            mask &= attributeArray > threshold
        gThresholded.delete_edges(np.where(mask == False)[0])
        # remove degree 0 nodes
        gThresholded.delete_vertices(gThresholded.vs.select(_degree=0))
        gThresholded.vs["CommunityIndex"] = gThresholded.community_leiden(objective_function = "modularity",
                                                                        weights = thresholdAttribute,
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
            community2EdgesAvgWeight[c] = np.mean(edgesInCommunity[thresholdAttribute])
        gThresholded.vs["CommunitySize"] = [community2Size[c] for c in gThresholded.vs["CommunityIndex"]]
        gThresholded.vs["CommunityEdgesCount"] = [community2EdgesCount[c] for c in gThresholded.vs["CommunityIndex"]]
        gThresholded.vs["CommunityEdgesDensity"] = [community2EdgesDensity[c] for c in gThresholded.vs["CommunityIndex"]]
        gThresholded.vs["CommunityEdgesAvg_"+thresholdAttribute] = [community2EdgesAvgWeight[c] for c in gThresholded.vs["CommunityIndex"]]
        if("Type" in gThresholded.es.attributes()):
            # get most common edge Type and associate it to the vertex
            for vertex in gThresholded.vs:
                # get all edges of the vertex
                edges = list(gThresholded.es.select(_source=vertex.index)) + list(gThresholded.es.select(_target=vertex.index))
                # get the type of the edges
                types = [edge["Type"] for edge in edges]
                # get the most common type
                mostCommonType = Counter(types).most_common(1)[0][0]
                # associate the most common type to the vertex
                vertex["Type"] = mostCommonType

            
        networks[f"{threshold}"] = gThresholded
    return networks









from pathlib import Path
from tqdm.auto import tqdm
import coordinationz as cz
import xnetwork as xn
import coordinationz.indicator_utilities as czind
import coordinationz.preprocess_utilities as czpre
import numpy as np

"""
This script tests the overlap between two sets of nodes in a bipartite graph.
It loads data from evaluation datasets, filters the nodes based on their degrees and strengths,
and then randomly selects two sets of nodes from the filtered bipartite graph.
It calculates the number of overlaps between the two sets and prints the result.
"""

if __name__ == "__main__": # Needed for parallel processing
    config = cz.config

    # dataName = "challenge_filipinos_5DEC"
    # dataName = "challenge_problem_two_21NOV_activeusers"
    dataName = "challenge_problem_two_21NOV"

    # Pairs of degrees that will be tested
    probeDegrees = [
        (13,13),
        (13,15),
        (13,50),
        (13,100),
        (100,100),
        (100,120)
    ]

    realizations = 1000
    
    # Loads data from the evaluation datasets as pandas dataframes
    df = czpre.loadPreprocessedData(dataName, config=config)

    bipartiteEdges = czind.obtainBipartiteEdgesRetweets(df)

    filteredBipartiteEdges = czind.filterNodes(bipartiteEdges,
                                           minRightDegree=1,
                                           minRightStrength=1,
                                           minLeftDegree=10,
                                           minLeftStrength=1)
        
    intersections = {}
    for _ in tqdm(range(realizations)):
        for pairDegrees in probeDegrees:
            firstSet = set(np.random.choice(filteredBipartiteEdges[:,1], size=pairDegrees[0], replace=False))
            secondSet = set(np.random.choice(filteredBipartiteEdges[:,1], size=pairDegrees[1], replace=False))
            # np.random.shuffle(bipartiteEdges[:,1])
            # firstSet = set(bipartiteEdges[:pairDegrees[0],1])
            # secondSet = set(bipartiteEdges[pairDegrees[0]:(pairDegrees[0]+pairDegrees[1]),1])
            if(pairDegrees not in intersections):
                intersections[pairDegrees] = []
            intersections[pairDegrees].append(len(firstSet.intersection(secondSet)))

    print("Overlaps (non-zero entries):")
    for pairDegrees,intersection in intersections.items():
        intersectionsCount = np.sum(np.array(intersection)>0)
        print(f"\tpair {pairDegrees}: {intersectionsCount} ({intersectionsCount/realizations*100:.2f}%)")


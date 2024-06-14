import coordinationz.fastcosine as fastcosine;
import numpy as np
from tqdm.auto import tqdm

from scipy.sparse import csr_matrix

def make_pbar(desc=None):
    pbar = None
    def inner(current,total):
        nonlocal pbar
        if(pbar is None):
            pbar= tqdm(total=total, desc=desc)
        pbar.update(current - pbar.n)
    return inner




# A= [0,1,1,1]
# B = [0,0,1,1]
# cosine = 2/3
edges =[
    (0, 1),
    (0, 1),
    (0, 1),
    (0, 1),
    (0, 2),
    (0, 2),
    (0, 2),
    (0, 3),

    (1, 2),
    (1, 3),

    (2, 1),
    (2, 1),
    (2, 4),
    (2, 5),
    (2, 6),
    (2, 8),
    (2, 8),

    (3, 5),
    (3, 6),
    (3, 7),
    (3, 8),
    (3, 8),
    (3, 8),
    (3, 8),
]
leftCount = 4
rightCount = 9

# # create random set of 100000 edges with leftCount=1000 and rightCount=1000

# edgesCount = 3_147_621
# leftCount = 1158148
# rightCount = 244487

edgesCount = 500_000
leftCount = 20000
rightCount = 20000
edges = np.zeros((edgesCount, 2), dtype=np.int64)
edges[:, 0] = np.random.randint(0, leftCount, edgesCount)
edges[:, 1] = np.random.randint(0, rightCount, edgesCount)


edgesCount = len(edges)
edges = np.array(edges)

for _ in range(1000000):
    result = fastcosine.bipartiteCosine(edges,
                               leftCount=leftCount,
                               rightCount=rightCount,
                               threshold=0.1,
                               returnDictionary=True,
                               updateCallback=make_pbar("Cosine Similarity"))

print(result)
# print(bipartiteCosineSimilarityMatrix(edges, leftCount=leftCount, rightCount=rightCount).toarray())





# resultFull = bipartiteCosineSimilarityMatrixThresholded(edges, leftCount=leftCount, rightCount=rightCount, threshold=0.1)
# print(resultFull)


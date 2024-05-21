from pathlib import Path
import pandas as pd
import numpy as np
from tqdm.auto import tqdm
import coordinationz as cz
import coordinationz.experiment_utilities as czexp
from scipy.stats import skew,kurtosis

import matplotlib.pyplot as plt
import matplotlib as mpl

# illustrator pdf font support
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42


if __name__ == "__main__": # Needed for parallel processing
    config = cz.config
    tqdm.pandas()
    # config = cz.load_config("<path to config>")
    
    figuresPath = Path(config["paths"]["FIGURES"]).resolve()
    figuresPath.mkdir(parents=True, exist_ok=True)


    # dataName = "challenge_filipinos_5DEC"
    dataName = "challenge_problem_two_21NOV_activeusers"
    # dataName = "challenge_problem_two_21NOV"

    # Loads data from the evaluation datasets as pandas dataframes
    dfEvaluation = czexp.loadEvaluationDataset(dataName, config=config, minActivities=1)

    # Create a bipartite graph from the retweet data
    bipartiteEdges = czexp.obtainEvaluationBipartiteEdgesRetweets(dfEvaluation)

    # creates a null model output from the bipartite graph
    nullModelOutput = cz.nullmodel.bipartiteNullModelSimilarity(
        bipartiteEdges,
        scoreType="onlynullmodel",
        realizations=100000,
        idf="linear", # None, "none", "linear", "smoothlinear", "log", "smoothlog"
        batchSize=100,
        workers=-1,
    )



    
    for useFisherTransform in [True, False]:

        degreePair2similarity = nullModelOutput["nullmodelDegreePairsSimilarities"]
        if(useFisherTransform):
            degreePair2similarity = {degreePair: np.arctanh(similarities[similarities<1.0]) for degreePair, similarities in degreePair2similarity.items()}

        variantName = "fisher" if useFisherTransform else "cosine"

        fig, ax = plt.subplots()
        # plot scatter plot of degreePairs product vs similarity
        degreePairsArray = np.array(list(degreePair2similarity.keys()))
        averageSimilarities = np.array([np.nanmean(similarities) for similarities in degreePair2similarity.values()])
        degreesProduct = degreePairsArray[:,0]*degreePairsArray[:,1]
        ax.scatter(degreesProduct, averageSimilarities, label="similarity")
        ax.set_xlabel("Degree Pairs product")
        ax.set_ylabel("Avg. Similarity")
        ax.set_xscale('log')
        if(useFisherTransform):
            ax.set_yscale('log')
        ax.set_title(f"Avg. of cosine sim. ({variantName})")
        ax.legend()
        plt.tight_layout()
        plt.savefig(figuresPath/f"{dataName}_{variantName}_avgSimilarity.png")
        plt.close()


        fig, ax = plt.subplots()
        # plot scatter plot of degreePairs product vs similarity
        degreePairsArray = np.array(list(degreePair2similarity.keys()))
        averageSimilarities = np.array([np.nanstd(similarities) for similarities in degreePair2similarity.values()])
        degreesProduct = degreePairsArray[:,0]*degreePairsArray[:,1]
        ax.scatter(degreesProduct, averageSimilarities, label="similarity")
        ax.set_xlabel("Degree Pairs product")
        ax.set_ylabel("Std. Similarity")
        ax.set_xscale('log')
        if(useFisherTransform):
            ax.set_yscale('log')
        ax.set_title(f"Std. of cosine sim. ({variantName})")
        ax.legend()
        plt.tight_layout()
        plt.savefig(figuresPath/f"{dataName}_{variantName}_stdSimilarity.png")
        plt.close()



        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        # plot scatter plot of degreePairs product vs skewness
        degreePairsArray = np.array(list(degreePair2similarity.keys()))
        skewness = np.array([skew(similarities,nan_policy="omit") for similarities in degreePair2similarity.values()])
        degreesProduct = degreePairsArray[:,0]*degreePairsArray[:,1]
        ax.scatter(degreesProduct, skewness, label="skewness")
        ax.set_xlabel("Degree Pairs product")
        ax.set_ylabel("Skewness")
        ax.set_xscale('log')
        if(useFisherTransform):
            ax.set_yscale('log')
        ax.set_title(f"Skewness of cosine sim. ({variantName})")
        ax.legend()
        plt.tight_layout()
        plt.savefig(figuresPath/f"{dataName}_{variantName}_skewness.png")
        plt.close()


        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        # plot scatter plot of degreePairs product vs kurtosis
        degreePairsArray = np.array(list(degreePair2similarity.keys()))
        kurtosis_values = np.array([kurtosis(similarities,nan_policy="omit") for similarities in degreePair2similarity.values()])
        degreesProduct = degreePairsArray[:,0]*degreePairsArray[:,1]
        ax.scatter(degreesProduct, kurtosis_values, label="kurtosis")
        ax.set_xlabel("Degree Pairs product")
        ax.set_ylabel("Kurtosis")
        ax.set_xscale('log')
        if(useFisherTransform):
            ax.set_yscale('log')
        ax.set_title(f"Kurtosis of cosine sim. ({variantName})")
        ax.legend()
        plt.tight_layout()
        plt.savefig(figuresPath/f"{dataName}_{variantName}_kurtosis.png")
        plt.close()
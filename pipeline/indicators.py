#!/usr/bin/env python

from pathlib import Path
import pandas as pd
import numpy as np
from tqdm.auto import tqdm
import coordinationz as cz
import xnetwork as xn
import coordinationz.experiment_utilities as czexp
import coordinationz.preprocess_utilities as czpre
import coordinationz.indicator_utilities as czind
import coordinationz.network as cznet
import sys
import argparse
import shutil
import json



# sampled_twitter_en_tl_global_0908
# challenge_problem_two_21NOV
# hamas_israel_challenge_problem_all_20240229
# TA2_small_eval_set_2024-02-16
# sampled_20240226
# 

if __name__ == "__main__": # Needed for parallel processing

    dataName = "sampled_20240226"
    indicators = ["coretweet","cohashtag","courl"]

    dataNameHelp = """Name of the dataset. A file named <dataName>.csv should be in the preprocessed datasets folder.
    if a dataName has a .csv extension, it will copy that file to the preprocessed datasets folder and use it."""
    parser = argparse.ArgumentParser()
    parser.add_argument("dataname", type=str, default=dataName,
                        help=dataNameHelp)
    parser.add_argument("-s","--suffix", type=str, default="", help="Will save the output files with this suffix. Default is timestamp.")
    parser.add_argument("-i","--indicators", nargs="+", default=["all"], help="List of indicators could be coretweet, cohashtag, courl or all. Default is all three.")

    # optional argument config (which should point to a config.toml file with the necessary settings)
    parser.add_argument("-c","--config", type=str, default=None, help="Path to the config file. Default is config.toml in the current directory.")

    args = parser.parse_args()
    dataName = args.dataname
    indicators = args.indicators
    suffix = args.suffix

    if("all" in indicators):
        indicators = ["coretweet","cohashtag","courl","coretweetusers","textsimilarity"]
    
    configPath = args.config
    if(configPath is not None):
        config = cz.load_config(configPath)
        # print("------")
        print("Loading config from",configPath,"...")
        # print("------")
    else:
        config = cz.config
        # print("------")
        print("Loading config from default location...")
        # print("------")

    if(dataName.endswith(".csv")):
        # get base file name
        source_file = Path(dataName)
        dataName = source_file.stem
        destination_file = Path(config["paths"]["PREPROCESSED_DATASETS"]) / f"{dataName}.csv"
        print("Will copy data to the preprocessed folder.")
        trial = 1
        while(destination_file.exists()):
            if(trial==1):
                print(f"File {destination_file} already exists.")
            #  will append a number to the file name
            dataName = f"{dataName}_{trial}"
            destination_file = Path(config["paths"]["PREPROCESSED_DATASETS"]) / f"{dataName}.csv"
            trial+=1
    
        print(f"Copying {source_file} to {destination_file}")
        shutil.copy(source_file, destination_file)
    
    # create a help/description entry if no arguments are provided
    if len(sys.argv) == 1:
        parser.print_help(sys.stderr)
        sys.exit(1)

    tqdm.pandas()
    # config = cz.load_config("<path to config>")
    
    networksPath = Path(config["paths"]["NETWORKS"]).resolve()
    networksPath.mkdir(parents=True, exist_ok=True)

    configsPath = Path(config["paths"]["CONFIGS"]).resolve()
    configsPath.mkdir(parents=True, exist_ok=True)

    figuresPath = Path(config["paths"]["FIGURES"]).resolve()
    figuresPath.mkdir(parents=True, exist_ok=True)

    tablesPath = Path(config["paths"]["TABLES"]).resolve()
    tablesPath.mkdir(parents=True, exist_ok=True)

    def text_similarity_partial(df):
        return czind.obtainBipartiteEdgesTextSimilarity(df, dataName, **config["indicator"]["textsimilarity"])

    # Available indicators
    bipartiteMethod = {
        "coretweet": czind.obtainBipartiteEdgesRetweets,
        "cohashtag": czind.obtainBipartiteEdgesHashtags,
        "courl": czind.obtainBipartiteEdgesURLs,
        "coretweetusers": czind.obtainBipartiteEdgesRetweetsUsers,
        "textsimilarity": text_similarity_partial
    }

    runParameters = czind.parseParameters(config,indicators)

    print("Loading data...")
    # # Loads data from the evaluation datasets as pandas dataframes
    df = czpre.loadPreprocessedData(dataName, config=config)

    # creates a null model output from the bipartite graph

    if(not suffix): 
        suffix = czind.timestamp()

    allUsers = set()
    generatedNetworks = {}
    for networkName in indicators:
        print(f"Creating the {networkName} network...")
        bipartiteEdges = bipartiteMethod[networkName](df)
        if(len(bipartiteEdges)==0):
            print(f"\n-------\nWARNING: No {networkName} edges found.\n-------\n")
            continue

        bipartiteEdges = czind.filterNodes(bipartiteEdges,**runParameters["filter"][networkName])
        # (user_ids, items)
        allUsers.update(set([userid for userid,_ in bipartiteEdges]))

        if(len(bipartiteEdges)==0):
            print(f"\n-------\nWARNING: No {networkName} edges found after filtering.\n-------\n")
            continue
        
        
        # bipartiteEdges.to_csv(networksPath/f"{dataName}_{networkName}_bipartiteEdges.csv", index=False)
        # creates a null model output from the bipartite graph
        nullModelOutput = cz.nullmodel.bipartiteNullModelSimilarity(
            bipartiteEdges,
            returnDegreeSimilarities=False, # will return the similarities of the nodes
            returnDegreeValues=True, # will return the degrees of the nodes
            **runParameters["nullmodel"][networkName]
        )
        # print(runParameters["nullmodel"][networkName])
        import matplotlib as plt
        import matplotlib.pyplot as plt
        similarities = nullModelOutput["similarities"]
        plt.figure()
        plt.scatter(similarities, nullModelOutput["quantiles"])
        plt.xlabel("Similarity")
        plt.ylabel("Quantile")
        plt.title("Similarity vs Quantile")
        plt.savefig(figuresPath / f"{dataName}_{suffix}_{networkName}_similarity_vs_quantile.png")
        plt.close()

        # Create a network from the null model output with a pvalue threshold of 0.05
        g = cznet.createNetworkFromNullModelOutput(
            nullModelOutput,
            # useZscoreWeights = True,
            # usePValueWeights = True,
            **runParameters["network"][networkName]
        )
            
        if("category" in df.columns):
            # dictionary
            user2category = dict(df[["user_id","category"]].drop_duplicates().values)
            g.vs["category"] = [user2category.get(user,"None") for user in g.vs["Label"]]

        xn.save(g, networksPath/f"{dataName}_{suffix}_{networkName}.xnet")
        generatedNetworks[networkName] = g
    
    print(f"Merging networks...")
    mergingMethod = runParameters["merging"]["method"]
    del runParameters["merging"]["method"]
    mergedNetwork = czind.mergeNetworks(generatedNetworks,
                                        **runParameters["merging"])

    xn.save(mergedNetwork, networksPath/f"{dataName}_{suffix}_merged.xnet")


    print(f"Saving data...")
    incasOutput = czind.generateEdgesINCASOutput(mergedNetwork, allUsers,
                                                 **runParameters["output"])
    
    # suspiciousEdgesData = czind.mergedSuspiciousEdges(mergedNetwork)
    # suspiciousClustersData = czind.mergedSuspiciousClusters(mergedNetwork)
    # suspiciousEdgesData.to_csv(tablesPath/f"{dataName}_{suffix}_merged_edges.csv",index=False)
    # suspiciousClustersData.to_csv(tablesPath/f"{dataName}_{suffix}_merged_clusters.csv")
    # save incasOutput to a json file
    for threshold,outputData in incasOutput.items():
        with open(tablesPath/f"{dataName}_{suffix}_segments_{threshold}.json", "w") as f:
            json.dump(outputData, f)
    
    config["run"]= {
        "dataName":dataName,
        "suffix":suffix,
        "indicators":indicators,
        "timestamp":czind.timestamp(),
        "currentPath":str(Path.cwd().resolve().absolute())
    }

    cz.save_config(configsPath/f"{dataName}_{suffix}.toml",config)


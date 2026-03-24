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
import coordinationz.communities as czcom
import sys
import argparse
import shutil
import json
import pickle
import multiprocessing as mp
import igraph as ig
import re

def filter_and_remove_edges(graph, conditions):
    """
    Removes edges connected to nodes that satisfy the given conditions.

    Parameters:
        graph (igraph.Graph): The input igraph graph.
        conditions (list of str): A list of filtering conditions (e.g., ["coreness>2", "degree>=1"]).

    Returns:
        igraph.Graph: A modified graph with edges removed.
    """

    # Define a mapping of node attributes to igraph functions
    attr_map = {
        "degree": graph.degree,
        "strength": graph.strength,
        "coreness": graph.coreness,
        "betweenness": graph.betweenness,
        "closeness": graph.closeness,
        "pagerank": graph.pagerank,
        "eccentricity": graph.eccentricity
    }

    # Regular expression to extract attribute, operator, and value
    pattern = re.compile(r"(\w+)([<>!=]=?)([\d\.]+)")

    nodes_to_remove_edges = set()

    for condition in conditions:
        match = pattern.match(condition)
        if not match:
            raise ValueError(f"Invalid condition format: {condition}")

        attr, operator, value = match.groups()
        value = float(value)  # Convert value to float

        if attr not in attr_map:
            raise ValueError(f"Unsupported attribute: {attr}")

        # Get the attribute values
        values = attr_map[attr]()

        # Apply the condition dynamically
        op_map = {
            ">": lambda x: x > value,
            "<": lambda x: x < value,
            ">=": lambda x: x >= value,
            "<=": lambda x: x <= value,
            "==": lambda x: x == value,
            "!=": lambda x: x != value
        }

        if operator not in op_map:
            raise ValueError(f"Unsupported operator: {operator}")

        # Identify nodes to remove edges from
        nodes_to_remove_edges.update(i for i, v in enumerate(values) if op_map[operator](v))

    # Identify and remove edges connected to the filtered nodes
    edges_to_remove = [e.index for e in graph.es if e.source in nodes_to_remove_edges or e.target in nodes_to_remove_edges]
    graph.delete_edges(edges_to_remove)

    return graph

# # Example usage
# gxx = ig.Graph.Erdos_Renyi(n=100, p=0.3)  # Create a random graph
# gxx = filter_and_remove_edges(gxx, ["degree<1"])
# gxx.ecount()

# sampled_twitter_en_tl_global_0908
# challenge_problem_two_21NOV
# hamas_israel_challenge_problem_all_20240229
# TA2_small_eval_set_2024-02-16
# sampled_20240226
# 

if __name__ == "__main__": # Needed for parallel processing
    try:
        mp.set_start_method('spawn')
    except RuntimeError:
        print("Spawn method already set.")
    
    dataName = "hamas_israel_challenge_problem_all_20240229"
    indicators = ["coretweet","cohashtag","courl","coretweetusers"]
    tweetIDTextCache = {}
    dataNameHelp = """Name of the dataset. A file named <dataName>.csv should be in the preprocessed datasets folder.
    if a dataName has a .csv extension, it will copy that file to the preprocessed datasets folder and use it."""
    parser = argparse.ArgumentParser()
    parser.add_argument("dataname", type=str, default=dataName,
                        help=dataNameHelp)
    parser.add_argument("-s","--suffix", type=str, default="", help="Will save the output files with this suffix. Default is timestamp.")
    parser.add_argument("-i","--indicators", nargs="+", default=["all"], help="List of indicators could be coretweet, cohashtag, courl or all. Default is all three.")

    # optional argument config (which should point to a config.toml file with the necessary settings)
    parser.add_argument("-c","--config", type=str, default=None, help="Path to the config file. Default is config.toml in the current directory.")

    parser.add_argument("-b","--balance-class", default="None", help="Column to balance the dataset.", required=False)

    args = parser.parse_args()
    dataName = args.dataname
    indicators = args.indicators
    suffix = args.suffix

    if("all" in indicators):
        indicators = ["coretweet","cohashtag","courl","coretweetusers","coword"]
    
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


    preprocessPath = Path(config["paths"]["PREPROCESSED_DATASETS"])
    extraPropertiesPath = preprocessPath/f"{dataName}_extraData.pkl"
    if(extraPropertiesPath.exists()):
        with open(extraPropertiesPath, "rb") as f:
            extraProperties = pickle.load(f)
    else:
        extraProperties = {}
    
    def text_similarity_partial(df):
        if "textsimilarity" in config["indicator"]:
            parameters = config["indicator"]["textsimilarity"]
        else:
            parameters = {}
        return czind.obtainBipartiteEdgesTextSimilarity(df, dataName, **parameters)

    def external_bipartite(filePath):
        # 
        edges = []
        with open(filePath,"r") as f:
            for line in f:
                tokens = line.strip().split(" ")
                user = tokens[1]
                multiplicity = int(float(tokens[2]))
                for i in range(multiplicity):
                    edges.append((user,tokens[0]))
        return edges



    # Available indicators
    bipartiteMethod = {
        "coretweet": czind.obtainBipartiteEdgesRetweets,
        "cohashtag": czind.obtainBipartiteEdgesHashtags,
        "courl": czind.obtainBipartiteEdgesURLs,
        "coretweetusers": czind.obtainBipartiteEdgesRetweetsUsers,
        "coword": czind.obtainBipartiteEdgesWords,
        "textsimilarity": text_similarity_partial,
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
        dfFiltered = czind.filterUsersByMinActivities(df,activityType=networkName, **runParameters["user"][networkName])

        print(f"Creating the {networkName} network...")
        independentNetworkExportPath = networksPath/f"{dataName}_{suffix}_{networkName}.xnet"
        if(independentNetworkExportPath.exists()):
            print(f"Network {independentNetworkExportPath} already exists. Skipping...")
            g = xn.load(independentNetworkExportPath)
            currentNodes = set(g.vs["Label"])
            allUsers.update(currentNodes)
        else:
            if(networkName=="usctextsimilarity"):
                import coordinationz.usc_text_similarity as cztext
                g = cztext.text_similarity(dfFiltered)
                currentNodes = set(g.vs["Label"])
            else:
                if(networkName.startswith("external")):
                    # external:filename.txt
                    # networkName will be ext_filename.stem
                    filePath = Path(networkName.split(":")[1])
                    # networkName = "ext_"+filePath.stem
                    allowedUsers = set(dfFiltered.user_id)
                    bipartiteEdges = external_bipartite(filePath)
                    bipartiteEdges = [(user,item) for user,item in bipartiteEdges if user in allowedUsers]
                    
                    # add networkName to the runParameters["filter"], runParameters["nullmodel"] and runParameters["network"]
                else:
                    bipartiteEdges = bipartiteMethod[networkName](dfFiltered)
                
                
                print(f"Filtering the the nodes in the bipartite network...")
                bipartiteEdges = czind.filterNodes(bipartiteEdges,**runParameters["filter"][networkName])

                if(len(bipartiteEdges)==0):
                    print(f"\n-------\nWARNING: No {networkName} edges found.\n-------\n")
                    continue


                # filter bipartite for balancing classes
                if (args.balance_class and args.balance_class in dfFiltered.columns):
                    print(f"Balancing the dataset by {args.balance_class}...")
                    user_classes = dict(dfFiltered[[ "user_id", args.balance_class]].drop_duplicates().values)
                    # Get the minimum class size
                    bipartite_nodes = set([userid for userid,_ in bipartiteEdges])
                    class_counts = {}
                    for user in bipartite_nodes:
                        cls = user_classes.get(user, None)
                        if cls is not None:
                            class_counts[cls] = class_counts.get(cls, 0) + 1
                    if not class_counts:
                        print(f"No users with class information found. Skipping balancing.")
                    else:
                        min_class_size = min(class_counts.values())
                        print(f"Class distribution before balancing: {class_counts}, minimum class size: {min_class_size}")
                        # Sample users from each class
                        sampled_users = set()
                        for cls, count in class_counts.items():
                            users_in_class = [user for user in bipartite_nodes if user_classes.get(user) == cls]
                            if len(users_in_class) > min_class_size:
                                sampled = np.random.choice(users_in_class, min_class_size, replace=False)
                            else:
                                sampled = users_in_class
                            sampled_users.update(sampled)
                        print(f"Sampled {len(sampled_users)} users.")
                        bipartiteEdges = [edge for edge in bipartiteEdges if edge[0] in sampled_users]


                # (user_ids, items)
                currentNodes = set([userid for userid,_ in bipartiteEdges])

                allUsers.update(currentNodes)
            
                if(len(bipartiteEdges)==0):
                    print(f"\n-------\nWARNING: No {networkName} edges found after filtering.\n-------\n")
                    continue
                
                
                # bipartiteEdges.to_csv(networksPath/f"{dataName}_{networkName}_bipartiteEdges.csv", index=False)
                # creates a null model output from the bipartite graph
                # save the parameters used to create the null model into a pickle file
                print(f"Creating the null model for the {networkName} network...")
                # with open(networksPath/f"{dataName}_{networkName}_nullmodel_parameters.pkl", "wb") as f:
                #     toSaveData = runParameters["nullmodel"][networkName].copy()
                #     toSaveData["bipartiteEdges"] = bipartiteEdges
                #     toSaveData["returnDegreeSimilarities"] = False
                #     toSaveData["returnDegreeValues"] = True
                #     toSaveData["filterNodesParameters"] = runParameters["filter"][networkName]
                #     pickle.dump(toSaveData, f,protocol=pickle.HIGHEST_PROTOCOL)
                
                nullModelOutput = cz.nullmodel.bipartiteNullModelSimilarity(
                    bipartiteEdges,
                    returnDegreeSimilarities=False, # will return the similarities of the nodes
                    returnDegreeValues=True, # will return the degrees of the nodes
                    **runParameters["nullmodel"][networkName]
                )
                # print(runParameters["nullmodel"][networkName])

                # Create a network from the null model output with a pvalue threshold of 0.05
                g = cznet.createNetworkFromNullModelOutput(
                    nullModelOutput,
                    **runParameters["network"][networkName]
                )

            if("category" in dfFiltered.columns):
                # dictionary
                user2category = dict(dfFiltered[["user_id","category"]].drop_duplicates().values)
                g.vs["category"] = [user2category.get(user,"None") for user in g.vs["Label"]]

        # g = cznet.removeSingletons(g)

        gThresholded = cznet.thresholdNetwork(g, **runParameters["threshold"][networkName])

        if(extraProperties):
            for key in extraProperties:
                gThresholded.vs[key] = [extraProperties[key].get(user,"None") for user in gThresholded.vs["Label"]]

        if(gThresholded.vcount()==0 or gThresholded.ecount()==0):
            print(f"\n-------\nWARNING: No {networkName} nodes or edges found after thresholding.\n-------\n")
            continue
        xn.save(gThresholded, independentNetworkExportPath)
        generatedNetworks[networkName] = gThresholded
        

        gForTable = gThresholded
        
        # if("community" in runParameters and runParameters["community"]["detectCommunity"]):
        #     print(f"Finding communities in the {networkName} network...")
        #     gCommunities = czcom.getNetworksWithCommunities(gThresholded.copy()) #**runParameters["communities"][networkName]
        #     # if(runParameters["community"]["computeCommunityLabels"]):
        #     #     print(f"Computing community labels for the {networkName} network...")
        #     #     gCommunities = czcom.labelCommunities(df,gCommunities,tweetIDTextCache)
        #     xn.save(gCommunities, networksPath/f"{dataName}_{suffix}_{networkName}_community.xnet")
        #     gForTable = gCommunities
        
        

        
        networkTables = cznet.getNetworkTables(gForTable, currentNodes)
        networkTables["nodes"].to_csv(tablesPath/f"{dataName}_{suffix}_{networkName}_nodes.csv",index=False)
        networkTables["edges"].to_csv(tablesPath/f"{dataName}_{suffix}_{networkName}_edges.csv",index=False)

    
    print(f"Merging networks...")
    # mergingMethod = runParameters["merging"]["method"]
    # del runParameters["merging"]["method"]
    originalMergedNetwork = czind.mergeNetworks(generatedNetworks,
                                        **runParameters["merging"])
    
    thresholdAttribute = runParameters["output"]["thresholdAttribute"]
    extraFilters = runParameters["output"]["filters"].copy()
    extraFilters["nofilter"] = []
    

    for threshold in runParameters["output"]["thresholds"]:
        for filterName, filterDefinition in extraFilters.items():
            thresholdOptions = {}
            thresholdOptions[thresholdAttribute] = threshold
            
            mergedNetwork = cznet.thresholdNetwork(originalMergedNetwork,thresholdOptions)
            
            if("community" in runParameters and runParameters["community"]["detectCommunity"]):
                print(f"Finding communities in the merged network...")
                mergedNetwork = czcom.getNetworksWithCommunities(mergedNetwork.copy()) #**runParameters["communities"][networkName]
                if(runParameters["community"]["computeCommunityLabels"]):
                    print(f"Computing community labels for the merged network...")
                    mergedNetwork = czcom.labelCommunities(df,mergedNetwork,tweetIDTextCache)
            
            if("extraThresholds" in runParameters["output"] and runParameters["output"]["extraThresholds"]):
                mergedNetwork = cznet.thresholdNetwork(mergedNetwork,runParameters["output"]["extraThresholds"])
            
            if(filterDefinition):
                print(f"Filtering the merged network with {filterName}...")
                mergedNetwork = filter_and_remove_edges(mergedNetwork, filterDefinition)
            
            xn.save(mergedNetwork, networksPath/f"{dataName}_{suffix}_merged_{threshold}_{filterName}.xnet")
            networkTables = cznet.getNetworkTables(mergedNetwork, allUsers)
            networkTables["nodes"].to_csv(tablesPath/f"{dataName}_{suffix}_merged_nodes_{threshold}_{filterName}.csv",index=False)
            networkTables["edges"].to_csv(tablesPath/f"{dataName}_{suffix}_merged_edges_{threshold}_{filterName}.csv",index=False)

            
            print(f"Saving data...")
            # allUsers = set(df["user_id"].values)
            incasOutput = czind.generateEdgesINCASOutput(mergedNetwork, allUsers,
                                                        rankingAttribute = thresholdAttribute)
            
            # suspiciousEdgesData = czind.mergedSuspiciousEdges(mergedNetwork)
            # suspiciousClustersData = czind.mergedSuspiciousClusters(mergedNetwork)
            # suspiciousEdgesData.to_csv(tablesPath/f"{dataName}_{suffix}_merged_edges.csv",index=False)
            # suspiciousClustersData.to_csv(tablesPath/f"{dataName}_{suffix}_merged_clusters.csv")
            # save incasOutput to a json file
            with open(tablesPath/f"{dataName}_{suffix}_segments_{threshold}_{filterName}.json", "w") as f:
                json.dump(incasOutput, f)

    config["run"]= {
        "dataName":dataName,
        "suffix":suffix,
        "indicators":indicators,
        "timestamp":czind.timestamp(),
        "currentPath":str(Path.cwd().resolve().absolute())
    }

    cz.save_config(configsPath/f"{dataName}_{suffix}.toml",config)


import coordinationz.indicator_utilities as czind
import coordinationz.preprocess_utilities as czpre
import coordinationz.usc_text_similarity as cztext
import coordinationz as cz
import argparse
from tqdm.auto import tqdm
import sys
from pathlib import Path
import xnetwork as xn

if __name__ == "__main__": # Needed for parallel processing

    dataName = "sampled_20240226"
    networkName = "usctextsimilarity"
    dataNameHelp = """Name of the dataset. A file named <dataName>.csv should be in the preprocessed datasets folder.
    if a dataName has a .csv extension, it will copy that file to the preprocessed datasets folder and use it."""
    parser = argparse.ArgumentParser()
    parser.add_argument("dataname", type=str, default=dataName,
                        help=dataNameHelp)
    parser.add_argument("-s","--suffix", type=str, default="", help="Will save the output files with this suffix. Default is timestamp.")

    # optional argument config (which should point to a config.toml file with the necessary settings)
    parser.add_argument("-c","--config", type=str, default=None, help="Path to the config file. Default is config.toml in the current directory.")

    args = parser.parse_args()
    dataName = args.dataname
    suffix = args.suffix
    
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

    # create a help/description entry if no arguments are provided
    if len(sys.argv) == 1:
        parser.print_help(sys.stderr)
        sys.exit(1)

    tqdm.pandas()
    # config = cz.load_config("<path to config>")
    
    networksPath = Path(config["paths"]["NETWORKS"]).resolve()
    networksPath.mkdir(parents=True, exist_ok=True)

    print("Loading data...")
    # # Loads data from the evaluation datasets as pandas dataframes
    df = czpre.loadPreprocessedData(dataName, config=config)
    g = cztext.text_similarity(df)
    xn.save(g, networksPath/f"{dataName}_{suffix}_{networkName}.xnet")



import pandas as pd
import os
from tqdm import tqdm
import coordinationz as cz
import json
from pathlib import Path
import coordinationz.preprocess_utilities as czpre
import sys
import argparse
import shutil
import ast
import csv


if __name__ == "__main__": # Needed for parallel processing
    dataName = "cuba_082020_tweets"

    dataNameHelp = """Name of the dataset. A file named [control|io]/<dataName>_<control|io>.pkl.gz should be in the IO datasets folder."""
    parser = argparse.ArgumentParser()
    parser.add_argument("dataname", type=str, default=dataName,
                        help=dataNameHelp)

    # optional argument config (which should point to a config.toml file with the necessary settings)
    parser.add_argument("-c","--config", type=str, default=None, help="Path to the config file. Default is config.toml in the current directory.")

    args = parser.parse_args()
    dataName = args.dataname

    configPath = args.config
    if(configPath is not None):
        config = cz.load_config(configPath)
    else:
        config = cz.config

    
    # create a help/description entry if no arguments are provided
    if len(sys.argv) == 1:
        parser.print_help(sys.stderr)
        sys.exit(1)

    # config = cz.load_config("<path to config>")

    IODataPath = Path(config["paths"]["IO_DATASETS"]).resolve()
    preprocessedDataPath = Path(config["paths"]["PREPROCESSED_DATASETS"]).resolve()
    preprocessedDataPath.mkdir(parents=True, exist_ok=True)

    preprocessedFilePath = preprocessedDataPath/f"{dataName}.csv"

    czpre.preprocessIOData(dataName, IODataPath, preprocessedFilePath)
  
    print("Loading preprocessed data for testing...")
    df = czpre.loadPreprocessedData(dataName, config=config)

    report = czpre.generateReport(df)

    # Saving report...
    reportFilename = preprocessedFilePath.with_name(preprocessedFilePath.stem + '_report.txt')
    with open(reportFilename, 'w') as f:
        f.write(report)

    print("Report:")
    print(report)



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



if __name__ == "__main__": # Needed for parallel processing
    dataName = "sampled_20240226"

    dataNameHelp = """Name of the dataset. A file named <dataName>.jsonl should be in the INCAS datasets folder.
    if a dataName has a .jsonl extension, it will copy that file to the preprocessed datasets folder and use it."""
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

    if(dataName.endswith(".jsonl")):
        # get base file name
        source_file = Path(dataName)
        dataName = source_file.stem
        destination_file = Path(config["paths"]["INCAS_DATASETS"]) / f"{dataName}.jsonl"
        print("Will copy data to the preprocessed folder.")
        trial = 1
        while(destination_file.exists()):
            if(trial==1):
                print(f"File {destination_file} already exists.")
            #  will append a number to the file name
            dataName = f"{dataName}_{trial}"
            destination_file = Path(config["paths"]["INCAS_DATASETS"]) / f"{dataName}.jsonl"
            trial+=1
        print(f"Copying {source_file} to {destination_file}")
        shutil.copy(source_file, destination_file)
    
    # create a help/description entry if no arguments are provided
    if len(sys.argv) == 1:
        parser.print_help(sys.stderr)
        sys.exit(1)

    # config = cz.load_config("<path to config>")

    incasDataPath = Path(config["paths"]["INCAS_DATASETS"]).resolve()
    preprocessedDataPath = Path(config["paths"]["PREPROCESSED_DATASETS"]).resolve()
    
    preprocessedDataPath.mkdir(parents=True, exist_ok=True)

    inputFilePath = incasDataPath/f"{dataName}.jsonl"
    preprocessedFilePath = preprocessedDataPath/f"{dataName}.csv"

    czpre.preprocessINCASData(inputFilePath, preprocessedFilePath)

    df = czpre.loadPreprocessedData(dataName, config=config)

    report = czpre.generateReport(df)

    # Saving report...
    reportFilename = preprocessedFilePath.with_name(preprocessedFilePath.stem + '_report.txt')
    with open(reportFilename, 'w') as f:
        f.write(report)

    print("Report:")
    print(report)



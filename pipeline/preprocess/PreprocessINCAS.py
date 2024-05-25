import pandas as pd
import os
from tqdm import tqdm
import coordinationz as cz
import json
from pathlib import Path
import coordinationz.preprocess_utilities as czpre
import sys

dataName = "sampled_20240226"

if len(sys.argv) > 1:
    dataName = sys.argv[1]
if len(sys.argv) > 2:
    indicators = sys.argv[2:]

    

if __name__ == "__main__": # Needed for parallel processing
    config = cz.config
    tqdm.pandas()

    # config = cz.load_config("<path to config>")

    incasDataPath = Path(config["paths"]["INCAS_DATASETS"]).resolve()
    preprocessedDataPath = Path(config["paths"]["PREPROCESSED_DATASETS"]).resolve()
    
    preprocessedDataPath.mkdir(parents=True, exist_ok=True)

    inputFilePath = incasDataPath/f"{dataName}.jsonl"
    preprocessedFilePath = preprocessedDataPath/f"{dataName}.csv"

    czpre.preprocessINCASData(inputFilePath, preprocessedFilePath)

    df = czpre.loadPreprocessedData(dataName)

    report = czpre.generateReport(df)

    # Saving report...
    reportFilename = preprocessedFilePath.with_name(preprocessedFilePath.stem + '_report.txt')
    with open(reportFilename, 'w') as f:
        f.write(report)

    print("Report:")
    print(report)



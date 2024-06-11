# coordinationz
Collection of scripts and package to analyze coordination in social media data.

To install the package, download the git repository and run the following command in the root directory:
```bash
pip install .
```

To install the package in development mode, run the following commands in the root directory:
```bash
pip install meson-python ninja numpy
pip install --no-build-isolation -e .
```

For debug mode, use the following command for local installation:
```bash
pip install --no-build-isolation -U -e . -Csetup-args=-Dbuildtype=debug
```
To debug the C code, use gdb:
```bash
gdb -ex=run -args python <python file>
```

## Run for INCAS datasets (e.g., phase2a or phase2b)
First install the package as described above.
The next step is setting up the config.toml file. You can use config_template.toml as a template.

```bash
cp config_template.toml config.toml
```

Setup the paths for the INCAS datasets and networks
```toml
# Location of jsonl files
INCAS_DATASETS = "/mnt/osome/INCAS/datasets" 

# Location where the preprocessed datasets will be stored
PREPROCESSED_DATASETS = "Data/Preprocessed"

#Logation of the outputs 
NETWORKS = "Outputs/Networks"
FIGURES = "Outputs/Figures"
TABLES = "Outputs/Tables"
CONFIGS = "Outputs/Configs"
```

The `INCAS_DATASETS` folder should contain the uncompressed jsonl files.

First, the files should be preprocessed. This can be done by running the following python script:
```bash
python pipeline/preprocess/preprocessINCAS.py <dataname>
``` 
where `dataname` is the name of the dataset, which correspondts to the `<INCAS_DATASETS>/<dataname>.jsonl` file. Together with the preprocessed data, the script will generate a .txt file with some information about the dataset.

The parameters of the indicators can be set in the config.toml file.

Currently, only co-hashtag, co-URL and co-retweets are supported.

To run the indicators, you can use the `pipeline/indicators.py` script by running the following command:
```bash
python pipeline/indicators.py <dataname>
```
where `dataname` is the name of the dataset and `indicator` is the indicator to be run.

You an add a suffix to the output files by adding the `--suffix` parameter:
```bash
python pipeline/indicators.py <dataname> --suffix <suffix>
```
if no suffix is provided, the a timestamp will be used as suffix.

Such a process will generate files in the output directories defined by `NETWORKS`, `TABLES`, and `CONFIGS`.

In particular, the `TABLES` folder will contain the suspicious pairs of users and clusters in CSV format.

The `NETWORKS` folder will contain the networks in xnet format. xnet format can be read by using the xnetwork package:
```bash
pip install xnetwork
```
and using the following code:
```python
import xnetwork as xn
g = xn.load("network.xnet")
```

The result is an igraph network. You can convert it to the networkx format by using the following code:
```python
network = g.to_networkx()
```

The config file used to generate the data will be copied to the "CONFIG" directory. A new section will be added to the config with extra parameters about the run.

## Text similarity indicators
The text similarity indicators can be run by including `usctextsimilarity`, `textsimilarity` or `coword` to the indicator list. For instance `pipeline/indicators.py <data_name> -i cohashtag coretweet courl textsimilarity. `usctextsimilarity` and textsimilarity requires the instalation of packages faiss and sentence-transformers. GPU is recommended for performance.

## Run for IO datasets
Repeat the same steps as for INCAS datasets, but set the `IO_DATASETS` variable in the config.toml file to the location of the IO datasets. Also, for preprocessing, use the `pipeline/preprocess/preprocessIO.py` script.

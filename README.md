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

## Run for Phase2B dataset
First install the package as described above.
The next step is setting up the config.toml file. Setup the paths for the INCAS
datasets and networks
```toml
INCAS_DATASETS = "/mnt/osome/INCAS/phase2b"
INCAS_PREPROCESSED_DATASETS = "/mnt/osome/INCAS/phase2b"
NETWORKS = "./Data/Networks"
```
The `INCAS_DATASETS` folder should contain the uncompressed jsonl files.

The scripts for processing the phase2b results are in `experiments/phase2b/` folder.

The first step is to generate the preprocessed dataset. This can be done by running the following python script:
```bash
experiments/phase2b/PreprocessPhase2b.py
```

Currently, only co-hashtag, co-URL and co-retweets are supported.

To run the analysis, you can run each of the Generate scripts in the `experiments/phase2b/`
folder. For example, to generate the co-hashtag network, run the following command:
```bash
python experiments/phase2b/GenerateCoHashtagNetwork.py
```

After generating all the three networks, you can run the following command to generate the merged
network:
```bash
python experiments/phase2b/MergeINCASpvalue.py
```
This will generate the merged network in the provided results folder.
In addition to that, an CSV file containing the suspicious users and respective
cluster memberships will be generated at the provided results folder.



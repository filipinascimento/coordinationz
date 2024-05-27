import toml
from pathlib import Path
import warnings
import os

def load_config(configPath = None):
    if(configPath is not None):
        path = Path(configPath)
    else:
        # if COORDINATIONZ_CONFIG_PATH is set, use that instead
        if 'COORDINATIONZ_CONFIG_PATH' in os.environ:
            path = Path(os.environ['COORDINATIONZ_CONFIG_PATH'])
        else:
            path = Path('config.toml').resolve()
            if not path.is_file():
                # Try ../config.toml just for convenience
                path = path.parent/'config.toml'
                if not path.is_file():
                    # Try ../../config.toml just for convenience
                    path = path.parent / 'config.toml'
    if not path.is_file():
        raise FileNotFoundError(f"Config file not found: {configPath}")
    
    with open(path, 'r') as file:
        config = toml.load(file)
    return config

try:
    config = load_config()
except FileNotFoundError as e:
    warnings.warn(str(e))
    warnings.warn("Tried to find config.toml in this directory, and parent directories.")
    warnings.warn("Please create a config.toml file with the necessary settings.")
    warnings.warn("Or manually call coordinationz.load_config('path/to/config.toml') in your code.")
    config = {}

def reload_config():
    newConfig = load_config()
    # remove all keys from config
    config.clear()
    # update config with newConfig
    config.update(newConfig)
    return config


def save_config(configPath, config=config):
    with open(configPath, 'w') as file:
        toml.dump(config, file)
    return configPath

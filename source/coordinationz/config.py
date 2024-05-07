import toml
from pathlib import Path
import warnings

def load_config(configPath = None):
    if(configPath is not None):
        path = Path(configPath)
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

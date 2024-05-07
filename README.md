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


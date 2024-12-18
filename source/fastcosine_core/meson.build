
c_sources = files(
    'fastcosine_core.c',
    'cxnetwork/fib.c',
    'cxnetwork/CVSet.c',
    'cxnetwork/CVSimpleQueue.c',
    'cxnetwork/CVNetwork.c',
    'cxnetwork/CVDictionary.c',
    'cxnetwork/CVDistribution.c',
    'cxnetwork/CVNetworkCentrality.c',
)


# Fetch NumPy include directory via Python
numpy_inc_dir = run_command(py3, '-c', 'import numpy; print(numpy.get_include())').stdout().strip()


# Define include directories
inc_np = include_directories(numpy_inc_dir)
# add defines WIN32
# add dependencies of numpy and openmp



# # NumPy include directory - needed in all submodules
# # The chdir is needed because within numpy there's an `import signal`
# # statement, and we don't want that to pick up scipy's signal module rather
# # than the stdlib module. The try-except is needed because when things are
# # split across drives on Windows, there is no relative path and an exception
# # gets raised. There may be other such cases, so add a catch-all and switch to
# # an absolute path. Relative paths are needed when for example a virtualenv is
# # placed inside the source tree; Meson rejects absolute paths to places inside
# # the source tree.
# # For cross-compilation it is often not possible to run the Python interpreter
# # in order to retrieve numpy's include directory. It can be specified in the
# # cross file instead:
# #   [properties]
# #   numpy-include-dir = /abspath/to/host-pythons/site-packages/numpy/core/include
# #
# # This uses the path as is, and avoids running the interpreter.
# incdir_numpy = meson.get_external_property('numpy-include-dir', 'not-given')
# if incdir_numpy == 'not-given'
#   incdir_numpy = run_command(py3,
#     [
#       '-c',
#       '''import os
# os.chdir(os.path.join("..", "tools"))
# import numpy as np
# try:
#   incdir = os.path.relpath(np.get_include())
# except Exception:
#   incdir = np.get_include()
# print(incdir)
#   '''
#     ],
#     check: true
#   ).stdout().strip()

#   # We do need an absolute path to feed to `cc.find_library` below
#   _incdir_numpy_abs = run_command(py3,
#     ['-c', 'import os; os.chdir(".."); import numpy; print(numpy.get_include())'],
#     check: true
#   ).stdout().strip()
# else
#   _incdir_numpy_abs = incdir_numpy
# endif
# inc_np = include_directories(incdir_numpy)
# # Don't use the deprecated NumPy C API. Define this to a fixed version instead of
# # NPY_API_VERSION in order not to break compilation for released SciPy versions
# # when NumPy introduces a new deprecation.
numpy_nodepr_api = ['-DNPY_NO_DEPRECATED_API=NPY_1_9_API_VERSION']
np_dep = declare_dependency(include_directories: inc_np, compile_args: numpy_nodepr_api)
# np_dep = declare_dependency(include_directories: inc_np)


# include openmp depend
# omp_dep = cc.find_library('openmp', required: true)
dependencies = []
dependencies += [np_dep]


host_system = host_machine.system()

# Conditional configurations based on host system
if host_system != 'darwin'
    omp_dep = dependency('openmp')
    dependencies += [omp_dep]
endif

py3_extension = py3.extension_module(
    'fastcosine_core', 
    sources: c_sources,
    install: true,
    subdir: 'coordinationz',
    dependencies: dependencies,
    # include_directories: inc_dirs
)





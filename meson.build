# project('my_python_package', 'c', 'python')

project('coordinationz',
        'c',
        default_options : [
                # 'c_std=c11',
                'warning_level=3',
                # 'b_ndebug=true',
                # 'debug=true',
                'optimization=3',         # Enable debug symbols
                # 'b_sanitize=address',  # Enable AddressSanitizer
                # 'b_lundef=false',
                ],
        version:'0.0.1',
        license:'MIT',
        )

cc = meson.get_compiler('c')

# Conditional configurations using built-in options
extra_options = []

host_system = host_machine.system()

# Conditional configurations based on host system
if host_system == 'darwin'
        extra_options += ['-DOSX']
        extra_options += ['-DCV_USE_LIBDISPATCH=1', '-Xpreprocessor', '-fblocks', '-lBlocksRuntime','-std=c11']
elif host_system == 'windows'
        # extra_options += ['-DWIN32', '-D__WIN32__']
        # extra_options += ['-DCV_USE_OPENMP=1', '-fopenmp','-std=c11']
        extra_options += ['/DWIN32', '/D__WIN32__']
        extra_options += ['/DCV_USE_OPENMP=1', '/openmp','/std:c11']
else
        extra_options += ['-DLinux', '-D_GNU_SOURCE=1']
        extra_options += ['-DCV_USE_OPENMP=1', '-fopenmp','-std=c11']
endif

# Apply extra compiler options
add_project_arguments(extra_options, language: 'c')
subdir('source')

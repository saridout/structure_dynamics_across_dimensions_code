from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
import sys
import setuptools
import pathlib

__version__ = '0.0.1'


# Touch source file to force recompile
pathlib.Path('cpp_src/python_bindings.cpp').touch()


class get_pybind_include(object):
    """Helper class to determine the pybind11 include path
    The purpose of this class is to postpone importing pybind11
    until it is actually installed, so that the ``get_include()``
    method can be invoked. """

    def __init__(self, user=False):
        self.user = user

    def __str__(self):
        import pybind11
        return pybind11.get_include(self.user)
    

use_alpha_cell_complex = True
use_optimal_cycles = False

print(get_pybind_include())
print(get_pybind_include(user=True))    

include_dirs=[
    # Path to pybind11 headers
    get_pybind_include(),
    get_pybind_include(user=True)
]
libraries = ["m"]
library_dirs = []
define_macros = []
# This option ensures that all linked shared libraries actually exist
# extra_link_args = ["-Wl,--no-undefined"]
extra_link_args = []
extra_compile_args = ["-g0"]

if use_alpha_cell_complex:
    #libraries.append("CGAL")
    libraries.append("gmp")
    define_macros.append(("ALPHA", None))
    

if use_optimal_cycles:
#     CPLEX_path = "/opt/ibm/ILOG/CPLEX_Studio128"
    CPLEX_path = "/home/rocks/ibm/ILOG/CPLEX_Studio128"
    
    include_dirs.append(CPLEX_path+"/concert/include")
    include_dirs.append(CPLEX_path+"/cplex/include") 
    
    library_dirs.append(CPLEX_path+"/concert/lib/x86-64_linux/static_pic")
    library_dirs.append(CPLEX_path+"/cplex/lib/x86-64_linux/static_pic")
    
    
    # The order of these libraries matters
    libraries.append("concert")
    libraries.append("ilocplex")
    libraries.append("cplex")
    # libraries.append("pthread")
    libraries.append("dl")
    
    define_macros.append(("IL_STD", None))
    define_macros.append(("OPTIMAL", None))
    
    
ext_modules = [
    Extension(
        'phom',
        ['cpp_src/python_bindings.cpp'],
        include_dirs=include_dirs,
        library_dirs=library_dirs,
        libraries=libraries,
        define_macros=define_macros,
        extra_link_args=extra_link_args,
        extra_compile_args=extra_compile_args,
        language='c++'
    ),
]


# As of Python 3.6, CCompiler has a `has_flag` method.
# cf http://bugs.python.org/issue26689
def has_flag(compiler, flagname):
    """Return a boolean indicating whether a flag name is supported on
    the specified compiler.
    """
    import tempfile
    with tempfile.NamedTemporaryFile('w', suffix='.cpp') as f:
        f.write('int main (int argc, char **argv) { return 0; }')
        try:
            compiler.compile([f.name], extra_postargs=[flagname])
        except setuptools.distutils.errors.CompileError:
            return False
    return True


def cpp_flag(compiler):
    """Return the -std=c++[11/14] compiler flag.
    The c++14 is prefered over c++11 (when it is available).
    """
    if has_flag(compiler, '-std=c++14'):
        return '-std=c++14'
    elif has_flag(compiler, '-std=c++11'):
        return '-std=c++11'
    else:
        raise RuntimeError('Unsupported compiler -- at least C++11 support '
                           'is needed!')


class BuildExt(build_ext):
    """A custom build extension for adding compiler-specific options."""
    c_opts = {
        'msvc': ['/EHsc'],
        'unix': [],
    }

    if sys.platform == 'darwin':
        c_opts['unix'] += ['-stdlib=libc++', '-mmacosx-version-min=10.7']

    def build_extensions(self):
        ct = self.compiler.compiler_type
        opts = self.c_opts.get(ct, [])
        if ct == 'unix':
            opts.append('-DVERSION_INFO="%s"' % self.distribution.get_version())
            opts.append(cpp_flag(self.compiler))
            if has_flag(self.compiler, '-fvisibility=hidden'):
                opts.append('-fvisibility=hidden')
        elif ct == 'msvc':
            opts.append('/DVERSION_INFO=\\"%s\\"' % self.distribution.get_version())
        for ext in self.extensions:
            ext.extra_compile_args.extend(opts)
        build_ext.build_extensions(self)

setup(
    name='phom',
    version=__version__,
    author='Jason W. Rocks',
    author_email='rocks@sas.upenn.edu',
    url='https://bitbucket.org/jrocks/lib_persistent_homology',
    description='Persistent Homology Algorithms and Utilities',
    long_description='',
    ext_modules=ext_modules,
    install_requires=['pybind11>=3.4'],
    cmdclass={'build_ext': BuildExt},
    zip_safe=False,
)

from distutils.core import setup
from Cython.Build import cythonize

setup(
  name = 'OPAL packages',
  ext_modules = cythonize("opacity.pyx"),
)
#setup cython code
import sys, os
from setuptools import setup, find_packages
from Cython.Build import cythonize
import numpy as np
import scipy as sp

#os.chdir('..') #move up one level
setup(
	name = "pycompass",
        install_requires=['Cython'],
        packages=find_packages(),
	ext_modules = cythonize("pycompass/SNE/pdf.pyx"),
	include_dirs = [np.get_include(),sp.get_include()]
	)

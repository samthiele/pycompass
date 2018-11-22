#setup cython code
import sys, os
from distutils.core import setup
from Cython.Build import cythonize
import numpy as np
import scipy as sp

os.chdir('..') #move up one level
setup(
	name = "pdf",
	ext_modules = cythonize("pycompass/SNE/pdf.pyx"),
	include_dirs = [np.get_include(),sp.get_include()]
	)
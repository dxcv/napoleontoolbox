#!/usr/bin/env python3
# coding: utf-8

""" Setup script. """

# Built-in packages
import sys
import os
from setuptools import setup, find_packages
from distutils.extension import Extension
from distutils.command.build_ext import build_ext

# Third party packages
import numpy

# Set this to True to enable building extensions using Cython.
# Set it to False to build extensions from the C file (that
# was previously created using Cython).
USE_CYTHON = 'auto'

CLASSIFIERS = [
    'Development Status :: 4 - Beta',
    'Intended Audience :: Financial and Insurance Industry',
    'Intended Audience :: Science/Research',
    'License :: OSI Approved :: MIT License',
    'Operating System :: POSIX :: Linux',
    'Programming Language :: Python',
    'Programming Language :: Cython',
    'Programming Language :: Python :: 3',
    'Programming Language :: Python :: 3.5',
    'Programming Language :: Python :: 3.6',
    'Programming Language :: Python :: 3.7',
    'Topic :: Office/Business :: Financial',
    'Topic :: Scientific/Engineering :: Artificial Intelligence',
]

MAJOR = 1
MINOR = 0
PATCH = 8
VERSION = '{}.{}.{}'.format(MAJOR, MINOR, PATCH)

DESCRIPTION = 'Python and Cython scripts of machine learning, econometrics '
DESCRIPTION += 'and statistical tools for financial analysis [In progress]'

build_requires = [
    'Cython>=0.29.0',
    'matplotlib>=3.0.1',
    'numpy>=1.15.3',
    'pandas>=0.23.4',
    'scipy>=1.2.0',
    'seaborn>=0.9.0',
]

setup(
    name='napoleontoolbox',
    version='0.6.0',
    packages=find_packages(),
    download_url='https://github.com/stef564/napoleontoolbox/archive/0.6.0.tar.gz',
    author='Napoleon Group',
    author_email='dsi@napoleonx.ai',
    description='Dashboard for financial market data',
    license='MIT',
    install_requires=build_requires,
    classifiers=CLASSIFIERS,
)
#sudo python setup.py sdist
#sudo twine upload dist/* (napoleonAM defacto pwd)pyt
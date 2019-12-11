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

if USE_CYTHON or USE_CYTHON == 'auto':
    try:
        from Cython.Build import cythonize
        from Cython.Distutils import build_ext

        ext = '.pyx'
        print('Using cython.')
        USE_CYTHON = True

    except ImportError:
        if not USE_CYTHON == 'auto':
            print("If USE_CYTHON is set to True, Cython is required to",
                  "compile fynance. Please install Cython or don't set",
                  "USE_CYTHON to True.")

            raise ImportError

        else:
            print('Not using cython.')
            ext = '.c'
            USE_CYTHON = False

else:
    ext = '.c'

if 'build_ext' in sys.argv[1:] or USE_CYTHON or USE_CYTHON == 'auto':
    cmdclass = {'build_ext': build_ext}

else:
    cmdclass = {}




ext_modules = None
print('deploying from pip')
for dirname, dirnames, filenames in os.walk('.'):
    # print path to all subdirectories first.
    for subdirname in dirnames:
        print(os.path.join(dirname, subdirname))

    # print path to all filenames.
    for filename in filenames:
        print(os.path.join(dirname, filename))

print('exploratory done')

try:
    extensions = [
        Extension(
            'napoleontoolbox.utility.metrics_cy',
            ['napoleontoolbox/utility/metrics_cy' + ext],
            include_dirs=[numpy.get_include(), '.']
        ),
        Extension(
            'napoleontoolbox.utility.momentums_cy',
            ['napoleontoolbox/utility/momentums_cy' + ext],
            include_dirs=[numpy.get_include(), '.']
        )
    ]
    if USE_CYTHON or USE_CYTHON == 'auto':
        ext_modules = cythonize(extensions, annotate=True)

    else:
        ext_modules = extensions

except ValueError as e :
    print(str(e))


if ext_modules is None :
    try:
        extensions = [
            Extension(
                'utility.metrics_cy',
                ['utility/metrics_cy' + ext],
                include_dirs=[numpy.get_include(), '.']
            ),
            Extension(
                'utility.momentums_cy',
                ['utility/momentums_cy' + ext],
                include_dirs=[numpy.get_include(), '.']
            )
        ]
        if USE_CYTHON or USE_CYTHON == 'auto':
            ext_modules = cythonize(extensions, annotate=True)

        else:
            ext_modules = extensions

    except ValueError as e:
        print(str(e))



setup(
    name='napoleontoolbox',
    version='0.0.7',
    packages=find_packages(),
    download_url='https://github.com/stef564/napoleontoolbox/archive/0.0.7.tar.gz',
    author='Napoleon Group',
    author_email='dsi@napoleonx.ai',
    description='Dashboard for financial market data',
    license='MIT',
    cmdclass=cmdclass,
    ext_modules=ext_modules,
    install_requires=build_requires,
    classifiers=CLASSIFIERS,
)

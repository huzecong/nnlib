.. nnlib documentation master file, created by
   sphinx-quickstart on Sun Dec 30 22:02:57 2018.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

nnlib 0.1
=========

Welcome to nnlib's documentation!

nnlib is a Pytorch-based, type-annotated utilities for neural networks in Python 3.6+. It is designed with the following criteria in mind:

- **Type-Annotated:** Though useful, dynamic typing can be error prone. nnlib uses type annotations anywhere possible so that type mismatch errors can be spotted before running.
- **PyCharm-Friendly:** nnlib was developed using PyCharm, so compatibility with the IDE is a high priority.
- **Generalize Boilerplates:** Everyone hates writing the same thing over and over again, and that's basically why we use libraries.


Installation
============

Since nnlib is still in its early stages, PyPI packages are not provided. It is recommended to include nnlib as a submodule in your git repository:

.. code-block:: bash

	cd /path/to/your/repo
	git submodule add https://github.com/huzecong/nnlib.git

and just use away! ::

	import nnlib


Contents
========

.. toctree::
   :maxdepth: 2

   arguments
   data
   modules
   torchutils
   train
   utils
   workaround


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

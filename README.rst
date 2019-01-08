=========
nnlib 0.1
=========

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

and just use away!

For code autocompletion in PyCharm to work, you would also need to mark the ``nnlib`` submodule root as sources root, which can be done by right-clicking the folder in the Project panel, and selecting "Mark Directory as... > Sources Root".

To update the submodule to the newest version:

.. code-block:: bash
	
	git submodule update --remote --recursive nnlib


Documentation
=============

The documentations are still in an early draft. They can be viewed at ...

License
=======

nnlib is MIT-style licensed.

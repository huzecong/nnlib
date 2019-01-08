.. nnlib documentation master file, created by
   sphinx-quickstart on Sun Dec 30 22:02:57 2018.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

nnlib.utils
=================================

.. automodule:: nnlib.utils
.. currentmodule:: nnlib.utils

The `utils` package contains useful utilities for machine learning applications and general Python scripting.


Batching
--------

The :mod:`batching` module contains utility functions for convenient batch creation.

.. autofunction:: minibatches_from
.. autofunction:: pad_sequences
.. autofunction:: batch_sequences
.. autofunction:: shift_packed_seq
.. autofunction:: mask_dropout_embeddings


File System
-----------

The :mod:`filesystem` module aims to provide additional functionality based on the Python 3 :mod:`pathlib` built-in package. It also defines the custom type :class:`PathType`, a union of :class:`str` and :class:`pathlib.Path`.

.. autofunction:: path_lca
.. autofunction:: path_add_suffix


Functional
----------

The :mod:`functional` module introduces building blocks to writing a functional dialect of Python.

.. autofunction:: scanl
.. autofunction:: scanr
.. autofunction:: is_none
.. autofunction:: not_none
.. autofunction:: filter_none
.. autofunction:: split_according_to
.. autofunction:: split_by


IO
---

The :mod:`io` module provides support for advanced IO operations.

.. autofunction:: shut_up
.. autofunction:: reverse_open


Iterable
--------

The :mod:`iterable` module includes utility functions and lazy data structures for working with iterators.

.. autofunction:: flat_iter
.. autoclass:: LazyList
.. autoclass:: ListWrapper
.. autoclass:: MeasurableGenerator
.. autoclass:: Range


Logging
-------

The :mod:`logging` module contains a global logger that supports verbosity level control, colored output, and advanced formatting.

.. autofunction:: timestamp

.. autoclass:: Logging


Math
----

The :mod:`math` module provides math utilities.

.. autoclass:: FNVHash
.. autofunction:: ceil_div
.. autofunction:: normalize
.. autofunction:: pow
.. autofunction:: prod
.. autofunction:: sum
.. autofunction:: random_subset
.. autofunction:: softmax


Miscellaneous
-------------

The :mod:`misc` module contains everything that cannot be categorized into the other categories.

.. autofunction:: progress
.. autofunction:: deprecated
.. autofunction:: map_to_list
.. autofunction:: memoization
.. autofunction:: reverse_map


NER
---

The :mod:`ner` module includes utilities for the task of Named Entity Recognition.

.. autofunction:: bieso2bio
.. autofunction:: bio2bieso


String
------

The :mod:`string` module provides methods for name manipulation and natural logging.

.. autofunction:: ordinal
.. autofunction:: to_camelcase
.. autofunction:: to_underscore
.. autofunction:: to_capitalized


Timing
------

The :mod:`timing` module contains utility functions for timing code performance.

.. autofunction:: work_in_progress

The following are MATLAB-style timing functions.

.. autofunction:: tic
.. autofunction:: toc
.. autofunction:: tic_toc
.. autofunction:: report_timing


Values
------

The :mod:`values` module provides utilities to conveniently wrap, accumulate, or record values.

.. autoclass:: AnnealedValue
	:members:
.. autoclass:: LambdaValue
	:members:
.. autoclass:: MilestoneCounter
	:members:

The following are accumulator objects.

.. autoclass:: Average
.. autoclass:: SimpleAverage
.. autoclass:: MovingAverage
.. autoclass:: WeightedAverage
.. autoclass:: HarmonicMean

The following are :class:`Average`\ -based record keepers.

.. autofunction:: add_record
.. autofunction:: record_value
.. autofunction:: summarize_values

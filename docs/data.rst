Data Preprocessing Utilities - ``nnlib.data``
=============================================

.. automodule:: nnlib.data

The ``data`` package provides utilities for loading standard datasets and preprocessing data (e.g. tokenization). It also provides standard data iterators, and an extensible interface for a general purpose data loader.


Datasets
--------

.. currentmodule:: nnlib.data.datasets

.. autoclass:: nnlib.data.datasets.dataset.NMTDataset
    :members:
.. autoclass:: IWSLT
    :members:
.. autoclass:: TEDTalks
    :members:


Data Loader Interface
---------------------

.. currentmodule:: nnlib.data.dataloader

.. autoclass:: DataLoader
    :members:
.. autoclass:: Vocabulary
    :members:
.. autofunction:: read_file
.. autofunction:: read_bitext_files


Data Iterators
--------------

.. currentmodule:: nnlib.data.iterator

.. autofunction:: bucketed_bitext_iterator
.. autofunction:: sorted_bitext_iterator
.. autofunction:: bitext_index_iterator
.. autofunction:: bitext_iterator
.. autofunction:: multi_dataset_iterator


Preprocessing
-------------

.. currentmodule:: nnlib.data.preprocess

.. autofunction:: tokenize
.. autofunction:: moses_tokenize
.. autofunction:: spacy_tokenize
.. autofunction:: spm_tokenize
.. autofunction:: get_tokenization_args

"""
Generalized dataset interface. Each dataset should provide a loader function, taking at least the parameters:
  train=False, dev=False, test=False, directory='data/'
indicating which dataset splits to use, and the save directory.
The loader function should download the dataset files (or use cached if possible), possibly combine them, and return
one path per split to the data files.

This might not be the best design for general NLP tasks, but is probably enough for NMT.
"""

from .iwslt import IWSLT
from .ted_talks import TEDTalks

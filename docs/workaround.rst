Workaround for PyTorch Autocompletion - ``nnlib.torch``
=======================================================

The sad thing about PyTorch is that, since version 0.4, its major methods are dynamically loaded from a precompiled binary, which means autocompletion was not supported in PyCharm.
To overcome this, a stubs file is provided in ``nnlib.workaround``. Stubs files contain signatures for the classes and functions, which can be indexed by IDEs.

To enable autocompletion in PyCharm, simple replace your PyTorch imports::

	import torch
	import torch.nn as nn
	import torch.nn.functional as F

with a simple::
	
	from nnlib.torch import *

For details, refer to the source code of ``nnlib.torch``.

import math
from typing import Callable, List, Optional, TypeVar, Union

from ..torch import *
from ..utils import Logging

__all__ = ['Linear', 'MLP', 'FC']


class Linear(nn.Linear):
    # noinspection PyMissingConstructor
    def __init__(self, in_features: int, out_features: int, bias=True, *, _weight: Optional[Tensor] = None):
        # skip super class and call __init__ of super super class
        super(__class__.__bases__[0], self).__init__()  # type: ignore
        self.in_features = in_features
        self.out_features = out_features
        if _weight is not None:
            self.weight = _weight
        else:
            self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()


Activation = Union[str, Callable[[Tensor], Tensor]]
T = TypeVar('T')
MaybeList = Union[T, List[T]]


class MLP(nn.Module):
    """
    Multi-layer perceptron. A convenient interface to stacked fully-connected layers. Each layer is a linear followed
    by an optional activation, with optional dropout.
    """

    activation_func = {
        'relu': torch.relu,
        'sigmoid': torch.sigmoid,
        'tanh': torch.tanh,
        'id': lambda x: x
    }
    activation_func['id'].__name__ = 'linear'

    def __init__(self, num_layers: int, input_dim: int, output_dim: int, hidden_dims: Optional[List[int]] = None,
                 activation: MaybeList[Activation] = 'id',
                 bias: MaybeList[bool] = True, bias_init: Optional[MaybeList[Optional[float]]] = None,
                 dropout: Optional[MaybeList[float]] = None):
        """
        Parameters `activation`, `bias`, `bias_init`, and `dropout` can be either a single value or a list of values
        for each layer. If using a single value, then the value is specified for every layer.

        :param hidden_dims: List of dimensions for each intermediate hidden representation.
        :param activation: This parameter can take the following as arguments:
            * str 'id' (default):
                Indicating simple affine transform.
            * str among ['relu', 'sigmoid', 'tanh'], or a function:
                The specified function will be used as activation for all layers.
            There are also special rules for `activation`:
            * `activation` can take a list of length `num_layers - 1`, in which case the final activation is identity.
            * When single value is specified and `num_layers` is greater than 1, the final activation is default to
              identity. To override this behavior, manually specify activation for each layer.
        :param bias: Whether to include additive bias.
        :param bias_init: If not None, use a constant initialization for the bias.
        :param dropout: If not None, apply dropout **before** linear transform.
        """
        # validate num_layers
        if not isinstance(num_layers, int) or num_layers < 1:
            raise ValueError("`layers` should be a positive integer.")
        # validate hidden_dims
        hidden_dims = hidden_dims or []
        if len(hidden_dims) != num_layers - 1:
            raise ValueError("Length of `hidden_dim` should be `layers` - 1.")
        # validate bias
        if isinstance(bias, (bool, int)):
            bias = [bias] * num_layers
        if not (len(bias) == num_layers and all(isinstance(b, (bool, int)) for b in bias)):
            raise ValueError("`bias` should be either a boolean, or a list of booleans of length `layers`.")
        # validate bias_init
        if bias_init is not None:
            if isinstance(bias_init, (float, int)):
                bias_init = [bias_init] * num_layers
            if not (len(bias_init) == num_layers and all(b is None or isinstance(b, (float, int)) for b in bias_init)):
                raise ValueError("`bias_init` should be either a float, or a list of floats of length `layers`.")
        else:
            bias_init = [None] * num_layers
        # validate dropout
        if dropout is not None:
            if isinstance(dropout, float):
                dropout = [dropout] * num_layers
            if not (len(dropout) == num_layers and all(isinstance(d, float) and 0 <= d < 1 for d in dropout)):
                raise ValueError("`dropout` should be either a float in range [0, 1),"
                                 " or a list of floats of length `layers`.")
        else:
            dropout = [0.0] * num_layers
        # validate activation
        if isinstance(activation, str) or callable(activation):
            if activation == 'id' and num_layers > 1:
                is_bottleneck = num_layers == 2 and input_dim > hidden_dims[0] and output_dim > hidden_dims[0]
                if not is_bottleneck:
                    Logging.warn("Using identity transform for non-bottleneck MLPs with more than one layer. "
                                 "This is likely an incorrect setting.")
            if num_layers == 1:
                activation = [activation]
            else:
                activation = [activation] * (num_layers - 1) + ['id']
        elif len(activation) == num_layers - 1:
            activation = activation + ['id']
        if not (isinstance(activation, list) and len(activation) == num_layers and
                all((isinstance(f, str) and f in self.activation_func) or callable(f) for f in activation)):
            raise ValueError("Format of `activation` is incorrect. Refer to docstring for details.")

        super().__init__()

        dims = [input_dim] + hidden_dims + [output_dim]
        self.layers = nn.ModuleList(
            [nn.Linear(in_features=in_dim, out_features=out_dim, bias=b)
             for in_dim, out_dim, b in zip(dims[:-1], dims[1:], bias)])
        self.activations = [f if callable(f) else self.activation_func[f] for f in activation]
        self.dropouts = [float(d) for d in dropout]
        self.bias_init = [float(b) if b is not None else None for b in bias_init]

        self.reset_parameters()

    def reset_parameters(self) -> None:
        for linear, activation, bias_init in zip(self.layers, self.activations, self.bias_init):
            gain = nn.init.calculate_gain(activation.__name__)
            std = gain * math.sqrt(6.0 / sum(linear.weight.size()))
            nn.init.uniform_(linear.weight, -std, std)
            if linear.bias is not None:
                if bias_init is not None:
                    nn.init.constant_(linear.bias, bias_init)
                else:
                    nn.init.uniform_(linear.bias, -std, std)

    def forward(self, xs: Tensor) -> Tensor:
        for linear, activation, dropout in zip(self.layers, self.activations, self.dropouts):
            if dropout > 0.0:
                xs = F.dropout(xs, dropout, self.training)
            xs = activation(linear.forward(xs))
        return xs


class FC:
    """
    Fully-connected layer. A wrapper for 1-layer MLP.
    """

    def __new__(cls, input_dim: int, output_dim: int, activation: Activation = 'id',
                bias: bool = True, bias_init: Optional[float] = None, dropout: Optional[float] = None):
        return MLP(1, input_dim, output_dim, None, activation, bias, bias_init, dropout)

from typing import List, Tuple, Union

from .. import utils
from ..torch import *

__all__ = ['get_parameters', 'param_count', 'dtype_size', 'print_module_memory_consumption']


def get_parameters(module: Union[nn.Module, torch.optim.Optimizer]) -> List[nn.Parameter]:
    if isinstance(module, nn.Module):
        # This is useful because certain parameters are replaced by wrappers
        return [p for p in module.parameters() if isinstance(p, nn.Parameter) and p.requires_grad]
    elif isinstance(module, torch.optim.Optimizer):
        optim = module
        return [param for param_group in optim.param_groups for param in param_group['params']]
    else:
        raise TypeError(f"`get_parameters` accepts argument of type `nn.Module` or `torch.optim.Optimizer`, "
                        f"but received {type(module)}")


def param_count(param: nn.Parameter) -> int:
    return utils.prod(param.shape) or 1  # empty shape means scalar


def dtype_size(dtype: torch.dtype) -> int:
    if dtype in [torch.float32, torch.int32]:
        return 4
    if dtype in [torch.float64, torch.int64]:
        return 8
    if dtype in [torch.float16, torch.int16]:
        return 2
    if dtype in [torch.int8, torch.uint8]:
        return 1
    raise ValueError(f"Invalid dtype: {dtype:r}")


def _print_module_memory_consumption_impl(prefix: str, module: nn.Module, depth: int, output_list: List[str]) \
        -> Tuple[int, int]:
    memory = n_param = 0
    for name, child in module.named_children():
        sub_mem, sub_n_param = _print_module_memory_consumption_impl(name, child, depth + 1, output_list)
        memory += sub_mem
        n_param += sub_n_param
    # noinspection PyProtectedMember
    for name, param in module._parameters.items():
        if isinstance(param, nn.Parameter):
            param_cnt = param_count(param)
            param_mem = param_cnt * dtype_size(param.dtype)
            output_list.append(f"{' |' * (depth + 1)} {'GPU ' if param.is_cuda else ''}Param: {prefix}.{name} "
                               f"(shape: {list(param.shape)}) {param_mem / (1024 * 1024):.2f}MB")
            memory += param_mem
            n_param += param_cnt
    output_list.append(f"{' |' * depth} Module {prefix} (params: {n_param}) {memory / (1024 * 1024):.2f}MB")
    return memory, n_param


def print_module_memory_consumption(module: nn.Module, model_name: str = "model"):
    tree: List[str] = []
    _print_module_memory_consumption_impl(model_name, module, depth=0, output_list=tree)
    print('\n'.join(reversed(tree)))

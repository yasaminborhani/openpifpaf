"""Backbone networks, head networks and tools for training."""

from .basenetworks import BaseNetwork
from .factory import Factory, local_checkpoint_path
from .heads import HeadNetwork
from .nets import Shell
from .running_cache import RunningCache
from .tracking_base import TrackingBase
from .tracking_heads import TBaseSingleImage, Tcaf
from .trainer import Trainer
from . import losses
import torch


def convert_instance_norm(module, affine=False):
    """Converts batch norm to instance norm."""
    module_output = module
    if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
        module_output = torch.nn.InstanceNorm2d(module.num_features,
                                                eps=module.eps,
                                                momentum=module.momentum,
                                                affine=affine)
    for name, child in module.named_children():
        module_output.add_module(name, convert_instance_norm(child, affine=affine))
    del module
    return module_output


class MyLayerNorm(torch.nn.Module):
    """LayerNorm with variable input shape."""

    def __init__(self, bn):
        super().__init__()
        self.eps = bn.eps

    def forward(self, x):
        """Forward pass."""
        return torch.nn.functional.layer_norm(
            x, x.shape[1:], None, None, self.eps)

    @classmethod
    def convert_mylayernorm(cls, module):
        module_output = module
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
            module_output = MyLayerNorm(module)
        for name, child in module.named_children():
            module_output.add_module(name, cls.convert_mylayernorm(child))
        del module
        return module_output

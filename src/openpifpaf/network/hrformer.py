import copy

import torch

try:
    import mmpose.models
except ImportError:
    pass


def hrformer_small_config(multiscale_output=False):
    return dict(
        type='HRFormer',
        in_channels=3,
        extra=dict(
            drop_path_rate=0.1,
            with_rpe=True,
            stage1=dict(
                num_modules=1,
                num_branches=1,
                block='BOTTLENECK',
                num_blocks=(2, ),
                num_channels=(64, ),
                num_heads=[2],
                num_mlp_ratios=[4]),
            stage2=dict(
                num_modules=1,
                num_branches=2,
                block='HRFORMERBLOCK',
                num_blocks=(2, 2),
                num_channels=(32, 64),
                num_heads=[1, 2],
                mlp_ratios=[4, 4],
                window_sizes=[7, 7]),
            stage3=dict(
                num_modules=4,
                num_branches=3,
                block='HRFORMERBLOCK',
                num_blocks=(2, 2, 2),
                num_channels=(32, 64, 128),
                num_heads=[1, 2, 4],
                mlp_ratios=[4, 4, 4],
                window_sizes=[7, 7, 7]),
            stage4=dict(
                num_modules=2,
                num_branches=4,
                block='HRFORMERBLOCK',
                num_blocks=(2, 2, 2, 2),
                num_channels=(32, 64, 128, 256),
                num_heads=[1, 2, 4, 8],
                mlp_ratios=[4, 4, 4, 4],
                window_sizes=[7, 7, 7, 7],
                multiscale_output=multiscale_output)),
        init_cfg=dict(
            type='Pretrained',
            checkpoint='https://download.openmmlab.com/mmpose/'
            'pretrain_models/hrformer_small-09516375_20220226.pth'),
    )


def hrformer_base_config(multiscale_output=False):
    return dict(
        type='HRFormer',
        in_channels=3,
        extra=dict(
            drop_path_rate=0.2,
            with_rpe=True,
            stage1=dict(
                num_modules=1,
                num_branches=1,
                block='BOTTLENECK',
                num_blocks=(2, ),
                num_channels=(64, ),
                num_heads=[2],
                mlp_ratios=[4]),
            stage2=dict(
                num_modules=1,
                num_branches=2,
                block='HRFORMERBLOCK',
                num_blocks=(2, 2),
                num_channels=(78, 156),
                num_heads=[2, 4],
                mlp_ratios=[4, 4],
                window_sizes=[7, 7]),
            stage3=dict(
                num_modules=4,
                num_branches=3,
                block='HRFORMERBLOCK',
                num_blocks=(2, 2, 2),
                num_channels=(78, 156, 312),
                num_heads=[2, 4, 8],
                mlp_ratios=[4, 4, 4],
                window_sizes=[7, 7, 7]),
            stage4=dict(
                num_modules=2,
                num_branches=4,
                block='HRFORMERBLOCK',
                num_blocks=(2, 2, 2, 2),
                num_channels=(78, 156, 312, 624),
                num_heads=[2, 4, 8, 16],
                mlp_ratios=[4, 4, 4, 4],
                window_sizes=[7, 7, 7, 7],
                multiscale_output=multiscale_output)),
        init_cfg=dict(
            type='Pretrained',
            checkpoint='https://download.openmmlab.com/mmpose/'
            'pretrain_models/hrformer_base-32815020_20220226.pth'),
    )


class HRFormerModuleWithUpsample(torch.nn.Module):
    """HRFormer module with forward modified to integrate upsampling."""
    def __init__(self, hrformermodule):
        super().__init__()
        self.hrformermodule = copy.deepcopy(hrformermodule)

    def forward(self, x):
        # pylint: disable=all
        """Adapted from https://github.com/open-mmlab/mmpose/blob/main/mmpose/models/backbones/hrnet.py#L194"""
        if self.hrformermodule.num_branches == 1:
            return [self.hrformermodule.branches[0](x[0])]
        for i in range(self.hrformermodule.num_branches):
            x[i] = self.hrformermodule.branches[i](x[i])
        x_fuse = []
        for i in range(len(self.hrformermodule.fuse_layers)):
            y = 0
            for j in range(self.hrformermodule.num_branches):
                if i == j:
                    y += x[j]
                elif j < i:
                    y += self.hrformermodule.fuse_layers[i][j](x[j])
                elif i < j:  # upsampling added here
                    y += torch.nn.functional.interpolate(
                        self.hrformermodule.fuse_layers[i][j](x[j]),
                        size=y.shape[2:],
                        mode='bilinear',
                        align_corners=False,
                    )
            x_fuse.append(self.hrformermodule.relu(y))
        return x_fuse


def adapt_hrformer(module):
    """Adapt HRFormer's upsampling to work with custom image size in OpenPifPaf."""
    module_output = module
    if isinstance(module, torch.nn.modules.Upsample):
        module_output = torch.nn.Sequential()  # remove Upsample module
    elif isinstance(module, mmpose.models.backbones.hrformer.HRFomerModule):
        module_output = HRFormerModuleWithUpsample(module)  # add upsampling to HRFormerModule
    for name, child in module.named_children():
        module_output.add_module(name, adapt_hrformer(child))
    del module
    return module_output


def hrformer(hrformer_config_fn=None,
             scale_level=0,
             concat_feature_maps=False,
             pretrained=True):
    multiscale_output = (scale_level != 0) or concat_feature_maps
    hrformer_config_dict = hrformer_config_fn(multiscale_output=multiscale_output)
    hrformer_backbone = mmpose.models.build_backbone(hrformer_config_dict)
    if pretrained:
        hrformer_backbone.init_weights()
    hrformer_backbone = adapt_hrformer(hrformer_backbone)
    if concat_feature_maps:
        fmp_index = [scale_level, 1, 2, 3]
        fmp_index[scale_level] = 0
        fmp = mmpose.models.FeatureMapProcessor(select_index=fmp_index, concat=True)
        out_features = sum(hrformer_config_dict['extra']['stage4']['num_channels'])
    else:
        fmp = mmpose.models.FeatureMapProcessor(select_index=scale_level)
        out_features = hrformer_config_dict['extra']['stage4']['num_channels'][scale_level]
    return torch.nn.Sequential(hrformer_backbone, fmp), out_features


def hrformersmall(scale_level=0, pretrained=True):
    hrformer_backbone, out_features = hrformer(hrformer_small_config,
                                               scale_level=scale_level,
                                               concat_feature_maps=False,
                                               pretrained=pretrained)
    return hrformer_backbone, out_features


def hrformersmallcat(scale_level=0, pretrained=True):
    hrformer_backbone, out_features = hrformer(hrformer_small_config,
                                               scale_level=scale_level,
                                               concat_feature_maps=True,
                                               pretrained=pretrained)
    return hrformer_backbone, out_features


def hrformerbase(scale_level=0, pretrained=True):
    hrformer_backbone, out_features = hrformer(hrformer_base_config,
                                               scale_level=scale_level,
                                               concat_feature_maps=False,
                                               pretrained=pretrained)
    return hrformer_backbone, out_features


def hrformerbasecat(scale_level=0, pretrained=True):
    hrformer_backbone, out_features = hrformer(hrformer_base_config,
                                               scale_level=scale_level,
                                               concat_feature_maps=True,
                                               pretrained=pretrained)
    return hrformer_backbone, out_features

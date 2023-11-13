try:
    import mmpretrain.models
except ImportError:
    pass


def adapt_convnextv2(backbone):
    """Adapt ConvNeXt V2's downsampling to work with custom image size in OpenPifPaf."""
    for i in range(backbone.num_stages):
        if i == 0:
            backbone.downsample_layers[0][0].padding = (2, 2)
        else:
            backbone.downsample_layers[i][1].padding = (1, 1)
    return backbone


def convnextv2(config=None, pretrained=True):
    backbone = mmpretrain.models.build_backbone(config)
    if pretrained:
        backbone.init_weights()
    backbone = adapt_convnextv2(backbone)
    return backbone


def convnextv2base(pretrained=True):
    convnextv2_base_config = dict(
        type='mmpretrain.ConvNeXt',
        arch='base',
        out_indices=[3],
        drop_path_rate=0.4,
        layer_scale_init_value=0.,  # disable layer scale when using GRN
        gap_before_final_norm=False,
        use_grn=True,  # V2 uses GRN
        init_cfg=dict(
            type='Pretrained',
            checkpoint='https://download.openmmlab.com/mmclassification/v0/convnext-v2/'
            'convnext-v2-base_3rdparty-fcmae_in1k_20230104-8a798eaf.pth',
            prefix='backbone.',
        ),
    )
    out_features = 1024
    return convnextv2(config=convnextv2_base_config, pretrained=pretrained), out_features

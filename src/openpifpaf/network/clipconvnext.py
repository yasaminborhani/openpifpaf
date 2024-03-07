try:
    import open_clip
except ImportError:
    pass


def adapt_clipconvnext(backbone):
    """Adapt CLIPConvNeXt's downsampling to work with custom image size in OpenPifPaf."""
    backbone.visual.trunk.stem[0].padding = (2, 2)
    backbone.visual.trunk.stages[1].downsample[1].padding = (1, 1)
    backbone.visual.trunk.stages[2].downsample[1].padding = (1, 1)
    backbone.visual.trunk.stages[3].downsample[1].padding = (1, 1)
    return backbone


def clipconvnext(model_name=None, pretraining_dataset=None):
    backbone, _, _ = open_clip.create_model_and_transforms(model_name,
                                                           pretrained=pretraining_dataset)
    backbone = adapt_clipconvnext(backbone)
    return backbone


def clipconvnextbase(pretrained=True):
    model_name, pretraining_dataset = 'convnext_base_w_320', 'laion_aesthetic_s13b_b82k_augreg'
    out_features = 1024
    if not pretrained:
        pretraining_dataset = None
    backbone = clipconvnext(model_name=model_name, pretraining_dataset=pretraining_dataset)
    return backbone, out_features

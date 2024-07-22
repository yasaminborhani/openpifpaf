"""
Interface for custom data.

This module handles datasets and is the class that you need to inherit from for your custom dataset.
This class gives you all the handles so that you can train with a new –dataset=mydataset.
The particular configuration of keypoints and skeleton is specified in the headmeta instances
"""

import argparse
from typing import List, Optional, Tuple

import torch
import torch.utils.data
import numpy as np
try:
    from pycocotools.coco import COCO
except ImportError:
    print('COCO not available')
    COCO = None

import openpifpaf
import openpifpaf.transforms

from .constants import get_constants #, training_weights_local_centrality
from .metrics import MeanPixelError
from .dataset import CocoDataset

class OpenLaneKp(openpifpaf.datasets.DataModule):
    """
    DataModule for the OpenLane Dataset.
    """
    debug = False
    pin_memory = False
    
    # test annotation json dir missing
    
    train_annotations = 'data_openlane/annotations/openlane_keypoints_sample_training.json'  
    val_annotations = 'data_openlane/annotations/openlane_keypoints_sample_validation.json' 
    eval_annotations = val_annotations 
    train_image_dir = '/work/scitas-share/datasets/Vita/civil-459/OpenLane/raw/images/training/' 
    val_image_dir = '/work/scitas-share/datasets/Vita/civil-459/OpenLane/raw/images/validation/' 
    eval_image_dir = val_image_dir

    n_images = None
    square_edge = 385 
    extended_scale = False
    orientation_invariant = 0.0
    blur = 0.0
    augmentation = True
    rescale_images = 1.0
    upsample_stride = 1
    min_kp_anns = 1
    b_min = 1  # 1 pixel

    eval_annotation_filter = True
    eval_long_edge = 256  
    eval_orientation_invariant = 0.0
    eval_extended_scale = False

    # lane-specific configuration
    hflip = None
    weights: Optional[List[float]] = None 
    lane_categories =  [ 
                        'unkown',             # 0
                        'white-dash',         # 1
                        'white-solid',        # 2
                        'double-white-dash',  # 3
                        'double-white-solid', # 4
                        'white-ldash-rsolid', # 5
                        'white-lsolid-rdash', # 6
                        'yellow-dash',        # 7
                        'yellow-solid',       # 8
                        'double-yellow-dash', # 9
                        'double-yellow-solid',# 10
                        'yellow-ldash-rsolid',# 11
                        'yellow-lsolid-rdash',# 12
                        'left-curbside',      # 20
                        'right-curbside'      # 21
                        ]
    lane_keypoints = None
    lane_skeleton: Optional[List[Tuple[int, int]]] = None
    lane_sigmas = None
    lane_pose = None
    lane_score_weights = None 

    def __init__(self):
        super().__init__()

        assert self.lane_keypoints is not None
        assert self.lane_sigmas is not None
        assert self.lane_skeleton is not None

        if self.weights is not None:
            caf_weights = [] 
            for bone1, bone2 in self.lane_skeleton:  # pylint: disable=not-an-iterable
                caf_weights.append(max(self.weights[bone1 - 1],
                                       self.weights[bone2 - 1])) 
            w_np = np.array(caf_weights)
            caf_weights = list(w_np / np.sum(w_np) * len(caf_weights))
        else:
            caf_weights = None

        cif = openpifpaf.headmeta.Cif('cif', 'openlane',
                                      keypoints=self.lane_keypoints,
                                      sigmas=self.lane_sigmas,
                                      pose=self.lane_pose,
                                      draw_skeleton=self.lane_skeleton,
                                      score_weights=self.lane_score_weights,
                                      training_weights=self.weights)
        caf = openpifpaf.headmeta.Caf('caf', 'openlane',
                                      keypoints=self.lane_keypoints,
                                      sigmas=self.lane_sigmas,
                                      pose=self.lane_pose,
                                      skeleton=self.lane_skeleton,
                                      training_weights=caf_weights)

        cif.upsample_stride = self.upsample_stride
        caf.upsample_stride = self.upsample_stride
        self.head_metas = [cif, caf]

    @classmethod
    def cli(cls, parser: argparse.ArgumentParser):
        group = parser.add_argument_group('data module OpenLane')

        group.add_argument('--openlane-train-annotations',
                           default=cls.train_annotations)
        group.add_argument('--openlane-val-annotations',
                           default=cls.val_annotations)
        group.add_argument('--openlane-train-image-dir',
                           default=cls.train_image_dir)
        group.add_argument('--openlane-val-image-dir',
                           default=cls.val_image_dir)

        group.add_argument('--openlane-square-edge',
                           default=cls.square_edge, type=int,
                           help='square edge of input images')
        assert not cls.extended_scale
        group.add_argument('--openlane-extended-scale',
                           default=False, action='store_true',
                           help='augment with an extended scale range')
        group.add_argument('--openlane-orientation-invariant',
                           default=cls.orientation_invariant, type=float,
                           help='augment with random orientations')
        group.add_argument('--openlane-blur',
                           default=cls.blur, type=float,
                           help='augment with blur')
        # assert cls.augmentation
        group.add_argument('--openlane-no-augmentation',
                           dest='apollo_augmentation',
                           default=True, action='store_false',
                           help='do not apply data augmentation')
        group.add_argument('--openlane-rescale-images',
                           default=cls.rescale_images, type=float,
                           help='overall rescale factor for images')
        group.add_argument('--openlane-upsample',
                           default=cls.upsample_stride, type=int,
                           help='head upsample stride')
        group.add_argument('--openlane-min-kp-anns',
                           default=cls.min_kp_anns, type=int,
                           help='filter images with fewer keypoint annotations')
        group.add_argument('--openlane-bmin',
                           default=cls.b_min, type=int,
                           help='b minimum in pixels')
      

        # evaluation
        assert cls.eval_annotation_filter
        group.add_argument('--openlane-no-eval-annotation-filter',
                           dest='openlane_eval_annotation_filter',
                           default=True, action='store_false')
        group.add_argument('--openlane-eval-long-edge', default=cls.eval_long_edge, type=int,
                           help='set to zero to deactivate rescaling')
        assert not cls.eval_extended_scale
        group.add_argument('--openlane-eval-extended-scale', default=False, action='store_true')
        group.add_argument('--openlane-eval-orientation-invariant',
                           default=cls.eval_orientation_invariant, type=float)
        
    @classmethod
    def configure(cls, args: argparse.Namespace):
        # extract global information
        cls.debug = args.debug
        cls.pin_memory = args.pin_memory

        # OpenLane specific
        cls.train_annotations = args.openlane_train_annotations
        cls.val_annotations = args.openlane_val_annotations
        cls.eval_annotations = cls.val_annotations
        cls.train_image_dir = args.openlane_train_image_dir
        cls.val_image_dir = args.openlane_val_image_dir
        cls.eval_image_dir = cls.val_image_dir

        cls.square_edge = args.openlane_square_edge
        cls.extended_scale = args.openlane_extended_scale
        cls.orientation_invariant = args.openlane_orientation_invariant
        cls.blur = args.openlane_blur
        # cls.augmentation = args.openlane_augmentation  # loaded by the dest name
        cls.rescale_images = args.openlane_rescale_images
        cls.upsample_stride = args.openlane_upsample
        cls.min_kp_anns = args.openlane_min_kp_anns
        cls.b_min = args.openlane_bmin
        
        (cls.lane_keypoints, cls.lane_skeleton, cls.lane_sigmas, cls.lane_pose,
        cls.lane_categories, cls.lane_score_weights) = get_constants()
        
          
        # evaluation
        cls.eval_annotation_filter = args.openlane_eval_annotation_filter
        cls.eval_long_edge = args.openlane_eval_long_edge
        cls.eval_orientation_invariant = args.openlane_eval_orientation_invariant
        cls.eval_extended_scale = args.openlane_eval_extended_scale
        
    def _preprocess(self):
        encoders = (openpifpaf.encoder.Cif(self.head_metas[0], bmin=self.b_min),
                    openpifpaf.encoder.Caf(self.head_metas[1], bmin=self.b_min))

        if not self.augmentation:
            return openpifpaf.transforms.Compose([
                openpifpaf.transforms.NormalizeAnnotations(),
                openpifpaf.transforms.RescaleAbsolute(self.square_edge),
                openpifpaf.transforms.CenterPad(self.square_edge),
                openpifpaf.transforms.EVAL_TRANSFORM,
                openpifpaf.transforms.Encoders(encoders),
            ])

        # not sure how the following coefficients determined
        if self.extended_scale:
            rescale_t = openpifpaf.transforms.RescaleRelative(
                scale_range=(0.2 * self.rescale_images,
                             2.0 * self.rescale_images),
                power_law=True, stretch_range=(0.75, 1.33))
        else:
            rescale_t = openpifpaf.transforms.RescaleRelative(
                scale_range=(0.33 * self.rescale_images,
                             1.33 * self.rescale_images),
                power_law=True, stretch_range=(0.75, 1.33))

        return openpifpaf.transforms.Compose([
            openpifpaf.transforms.NormalizeAnnotations(),
            # openpifpaf.transforms.AnnotationJitter(),
            # openpifpaf.transforms.RandomApply(
                # openpifpaf.transforms.HFlip(self.car_keypoints, self.hflip), 0.5),
            rescale_t,
            openpifpaf.transforms.RandomApply(openpifpaf.transforms.Blur(), self.blur),
            openpifpaf.transforms.RandomChoice(
                [openpifpaf.transforms.RotateBy90(),
                 openpifpaf.transforms.RotateUniform(30.0)],
                [self.orientation_invariant, 0.2],
            ),
            openpifpaf.transforms.Crop(self.square_edge, use_area_of_interest=True),
            openpifpaf.transforms.CenterPad(self.square_edge),
            openpifpaf.transforms.MinSize(min_side=32.0),
            openpifpaf.transforms.TRAIN_TRANSFORM,
            openpifpaf.transforms.Encoders(encoders),
        ])

    def train_loader(self):
        train_data = CocoDataset(
            image_dir=self.train_image_dir,
            ann_file=self.train_annotations,
            preprocess=self._preprocess(),
            annotation_filter=True,
            min_kp_anns=self.min_kp_anns,
            category_ids=[0,1,2,3,4,5,6,7,8,9,10,11,12,20,21], 
            # category_ids=[0],
        )
        return torch.utils.data.DataLoader(
            train_data, batch_size=self.batch_size, shuffle=not self.debug,
            pin_memory=self.pin_memory, num_workers=self.loader_workers, drop_last=True,
            collate_fn=openpifpaf.datasets.collate_images_targets_meta)

    def val_loader(self):
        val_data = CocoDataset(
            image_dir=self.val_image_dir,
            ann_file=self.val_annotations,
            preprocess=self._preprocess(),
            annotation_filter=True,
            min_kp_anns=self.min_kp_anns,
            category_ids=[0,1,2,3,4,5,6,7,8,9,10,11,12,20,21],
            # category_ids=[0],
        )
        return torch.utils.data.DataLoader(
            val_data, batch_size=self.batch_size, shuffle=False,
            pin_memory=self.pin_memory, num_workers=self.loader_workers, drop_last=True,
            collate_fn=openpifpaf.datasets.collate_images_targets_meta)

    @classmethod
    def common_eval_preprocess(cls):
        rescale_t = None
        if cls.eval_extended_scale:
            assert cls.eval_long_edge
            rescale_t = [
                openpifpaf.transforms.DeterministicEqualChoice([
                    openpifpaf.transforms.RescaleAbsolute(cls.eval_long_edge),
                    openpifpaf.transforms.RescaleAbsolute((cls.eval_long_edge - 1) // 2 + 1),
                ], salt=1)
            ]
        elif cls.eval_long_edge:
            rescale_t = openpifpaf.transforms.RescaleAbsolute(cls.eval_long_edge)

        if cls.batch_size == 1:
            padding_t = openpifpaf.transforms.CenterPadTight(16)
        else:
            assert cls.eval_long_edge
            padding_t = openpifpaf.transforms.CenterPad(cls.eval_long_edge)

        orientation_t = None
        if cls.eval_orientation_invariant:
            orientation_t = openpifpaf.transforms.DeterministicEqualChoice([
                None,
                openpifpaf.transforms.RotateBy90(fixed_angle=90),
                openpifpaf.transforms.RotateBy90(fixed_angle=180),
                openpifpaf.transforms.RotateBy90(fixed_angle=270),
            ], salt=3)

        return [
            openpifpaf.transforms.NormalizeAnnotations(),
            rescale_t,
            padding_t,
            orientation_t,
        ]

    def _eval_preprocess(self):
        return openpifpaf.transforms.Compose([
            *self.common_eval_preprocess(),
            openpifpaf.transforms.ToAnnotations([
                openpifpaf.transforms.ToKpAnnotations(
                    self.lane_categories,
                    keypoints_by_category={0: self.head_metas[0].keypoints,
                                          1: self.head_metas[0].keypoints,
                                          2: self.head_metas[0].keypoints,
                                          3: self.head_metas[0].keypoints,
                                          4: self.head_metas[0].keypoints,
                                          5: self.head_metas[0].keypoints,
                                          6: self.head_metas[0].keypoints,
                                          7: self.head_metas[0].keypoints,
                                          8: self.head_metas[0].keypoints,
                                          9: self.head_metas[0].keypoints,
                                          10: self.head_metas[0].keypoints,
                                          11: self.head_metas[0].keypoints,
                                          12: self.head_metas[0].keypoints,
                                          20: self.head_metas[0].keypoints,
                                          21: self.head_metas[0].keypoints,
                                        },
                    skeleton_by_category={0: self.head_metas[1].skeleton,
                                          1: self.head_metas[1].skeleton,
                                          2: self.head_metas[1].skeleton,
                                          3: self.head_metas[1].skeleton,
                                          4: self.head_metas[1].skeleton,
                                          5: self.head_metas[1].skeleton,
                                          6: self.head_metas[1].skeleton,
                                          7: self.head_metas[1].skeleton,
                                          8: self.head_metas[1].skeleton,
                                          9: self.head_metas[1].skeleton,
                                          10: self.head_metas[1].skeleton,
                                          11: self.head_metas[1].skeleton,
                                          12: self.head_metas[1].skeleton,
                                          20: self.head_metas[1].skeleton,
                                          21: self.head_metas[1].skeleton,
                                        },
                ),
                openpifpaf.transforms.ToCrowdAnnotations(self.lane_categories),
            ]),
            openpifpaf.transforms.EVAL_TRANSFORM,
        ])

    def eval_loader(self):
        eval_data = CocoDataset(
            image_dir=self.eval_image_dir,
            ann_file=self.eval_annotations,
            preprocess=self._eval_preprocess(),
            annotation_filter=self.eval_annotation_filter,
            min_kp_anns=self.min_kp_anns if self.eval_annotation_filter else 0,
            category_ids=[0,1,2,3,4,5,6,7,8,9,10,11,12,22,20,21] if self.eval_annotation_filter else [],
            # category_ids=[0] if self.eval_annotation_filter else [],
        )
        return torch.utils.data.DataLoader(
            eval_data, batch_size=self.batch_size, shuffle=False,
            pin_memory=self.pin_memory, num_workers=self.loader_workers, drop_last=False,
            collate_fn=openpifpaf.datasets.collate_images_anns_meta)

    def metrics(self):
        # TODO: make sure that 24kp flag is activated when evaluating a 24kp model
        if COCO is None:
            return []
        # return [openpifpaf.metric.Coco(
        #     COCO(self.eval_annotations),
        #     max_per_image=20,
        #     category_ids=[0,1,2,3,4,5,6,7,8,9,10,11,12,22,20,21],
        #     iou_type='keypoints',
        #     # keypoint_oks_sigmas=self.lane_sigmas,
        #     keypoint_oks_sigmas=None,
        # ), MeanPixelError()]

        return [openpifpaf.metric.Coco(
            COCO(self.eval_annotations),
            max_per_image=20,
            category_ids=[0,1,2,3,4,5,6,7,8,9,10,11,12,22,20,21],
            iou_type='keypoints',
            # keypoint_oks_sigmas=self.lane_sigmas,
            keypoint_oks_sigmas=self.lane_sigmas,
        ),MeanPixelError()]


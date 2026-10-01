"""Regression checks for the six-keypoint bicycle evaluator."""

import argparse
import json

import numpy as np
import pytest

from openpifpaf.plugins.cocobicycle.cocobicyclekp import CocoBicycleKp


class Prediction:
    def __init__(self, keypoints):
        self.keypoints = keypoints

    def json_data(self):
        return {'category_id': 1, 'score': 0.99,
                'keypoints': self.keypoints, 'bbox': [10, 10, 180, 100]}


def test_bicycle_eval_flags_are_independent_of_coco_flags():
    parser = argparse.ArgumentParser()
    parser.set_defaults(debug=False, pin_memory=False)
    CocoBicycleKp.cli(parser)
    args = parser.parse_args(['--cocobicyclekp-eval-long-edge', '769'])
    CocoBicycleKp.configure(args)
    assert CocoBicycleKp.eval_long_edge == 769


@pytest.mark.parametrize('has_prediction,expected_ap', [(True, 1.0), (False, 0.0)])
def test_bicycle_metric_uses_six_keypoints(tmp_path, has_prediction, expected_ap):
    from pycocotools.coco import COCO  # pylint: disable=import-outside-toplevel

    points = [[20 + 10 * i, 30 + 5 * (i % 3), 2] for i in range(6)]
    flat = np.asarray(points).reshape(-1).tolist()
    annotations = {
        'info': {},
        'images': [{'id': 1, 'width': 512, 'height': 512}],
        'categories': [{'id': 1, 'name': 'bicycle',
                        'keypoints': [str(i) for i in range(6)], 'skeleton': []}],
        'annotations': [{'id': 1, 'image_id': 1, 'category_id': 1,
                         'keypoints': flat, 'num_keypoints': 6,
                         'bbox': [10, 10, 180, 100], 'area': 18000, 'iscrowd': 0}],
    }
    path = tmp_path / 'bicycle.json'
    path.write_text(json.dumps(annotations))
    assert COCO(str(path)).getImgIds() == [1]
    module = CocoBicycleKp()
    module.eval_annotations = str(path)
    metric = module.metrics()[0]
    metric.accumulate([Prediction(flat)] if has_prediction else [], {'image_id': 1})
    assert metric.stats()['stats'][0] == pytest.approx(expected_ap)
    assert len(metric.eval.params.kpt_oks_sigmas) == 6
    np.testing.assert_allclose(metric.eval.params.kpt_oks_sigmas, [0.089] * 6)
    if not has_prediction:
        assert len(metric.predictions[0]['keypoints']) == 18

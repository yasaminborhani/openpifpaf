"""Lane evaluation must keep empty images and avoid duplicate image IDs."""

import json

import pytest

from openpifpaf.plugins.openpifpaf_culane.dataset import CocoDataset as CulaneDataset
from openpifpaf.plugins.openpifpaf_openlane.dataset import CocoDataset as OpenLaneDataset


@pytest.mark.parametrize('dataset_type', [OpenLaneDataset, CulaneDataset])
def test_full_split_includes_each_image_once(tmp_path, dataset_type):
    annotations = {
        'info': {},
        'images': [{'id': 1, 'file_name': 'one.jpg'},
                   {'id': 2, 'file_name': 'two.jpg'}],
        'categories': [{'id': 1, 'name': 'lane'}, {'id': 2, 'name': 'other lane'}],
        'annotations': [{'id': 1, 'image_id': 1, 'category_id': 1,
                         'iscrowd': 0, 'keypoints': [10, 10, 2, 20, 20, 2]}],
    }
    path = tmp_path / 'lanes.json'
    path.write_text(json.dumps(annotations))
    full = dataset_type(str(tmp_path), str(path), category_ids=[],
                        annotation_filter=False)
    assert sorted(full.ids) == [1, 2]
    assert len(full) == 2
    positive = dataset_type(str(tmp_path), str(path), category_ids=[1, 2],
                            annotation_filter=True, min_kp_anns=1)
    assert positive.ids == [1]

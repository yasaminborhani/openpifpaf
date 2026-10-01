"""Restore a trained bicycle branch when its encoder/FPN agree numerically."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import torch
import openpifpaf

p = argparse.ArgumentParser()
p.add_argument('--joint-checkpoint', type=Path, required=True)
p.add_argument('--bicycle-checkpoint', type=Path, required=True)
p.add_argument('--output', type=Path, required=True)
a = p.parse_args()
assert not a.output.exists(), 'Existing checkpoint must not be overwritten'
torch.set_num_threads(4)
joint_data = torch.load(a.joint_checkpoint, map_location='cpu', weights_only=False)
bicycle_data = torch.load(a.bicycle_checkpoint, map_location='cpu', weights_only=False)
joint = joint_data['model']
bicycle = bicycle_data['model']
assert {m.dataset for m in joint.head_metas} == {'cocokp', 'animal', 'apollo', 'openlane'}
assert {m.dataset for m in bicycle.head_metas} == {'cocobicyclekp'}
assert len(joint.head_nets) == 8 and len(bicycle.head_nets) == 2
assert joint_data['epoch'] == 450 and bicycle_data['epoch'] > 450
left, right = joint.base_net.state_dict(), bicycle.base_net.state_dict()
assert left.keys() == right.keys(), 'Encoder/FPN state keys differ'
close = lambda x,y: torch.allclose(x, y, rtol=1e-5, atol=1e-7) if x.is_floating_point() else torch.equal(x,y)
assert all(close(left[k], right[k]) for k in left), 'Encoder/FPN tensors differ beyond floating-point tolerance'
maximum_difference = max((left[k]-right[k]).abs().max().item() for k in left)
assert maximum_difference <= 2e-6, maximum_difference
assert joint.base_net.fpn is not None
assert all(hasattr(h, 'att') for h in list(joint.head_nets) + list(bicycle.head_nets))
original_heads = list(joint.head_nets)
joint.set_head_nets(original_heads + list(bicycle.head_nets))
assert len(joint.head_nets) == 10 and {m.dataset for m in joint.head_metas} == {'cocokp', 'animal', 'apollo', 'openlane', 'cocobicyclekp'}
joint.eval()

def sha256(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(16 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()

report = {
    'created_utc': datetime.now(timezone.utc).isoformat(),
    'construction': 'Restored the trained bicycle CIF/CAF modules onto the original four-task model after verifying every shared encoder/FPN state tensor agrees within the recorded floating-point tolerance. Original four-task encoder/FPN and all trained head tensors are retained.',
    'joint_checkpoint': str(a.joint_checkpoint),
    'joint_checkpoint_epoch': joint_data['epoch'],
    'bicycle_checkpoint': str(a.bicycle_checkpoint),
    'bicycle_checkpoint_epoch': bicycle_data['epoch'],
    'joint_checkpoint_sha256': sha256(a.joint_checkpoint),
    'bicycle_checkpoint_sha256': sha256(a.bicycle_checkpoint),
    'shared_encoder_fpn_identical': all(torch.equal(left[k], right[k]) for k in left),
    'shared_encoder_fpn_numerically_equivalent': True,
    'shared_encoder_fpn_maximum_absolute_difference': maximum_difference,
    'shared_encoder_fpn_comparison_rtol': 1e-5,
    'shared_encoder_fpn_comparison_atol': 1e-7,
    'shared_state_tensor_count': len(left),
    'original_four_task_head_tensors_retained': True,
    'trained_bicycle_head_tensors_retained': True,
    'new_training_performed': False,
    'five_task_joint_fine_tuning_performed': False,
    'single_encoder_forward': True,
    'output_checkpoint': str(a.output),
    'heads': [{'dataset': m.dataset, 'name': m.name, 'index': m.head_index,
               'keypoints': len(m.keypoints), 'stride': m.stride} for m in joint.head_metas],
}
a.output.parent.mkdir(parents=True, exist_ok=True)
torch.save({'model': joint, 'epoch': joint_data['epoch'], 'meta': {'five_branch_restoration': report}}, a.output)
report['output_checkpoint_sha256'] = sha256(a.output)
a.output.with_suffix('.provenance.json').write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps(report, indent=2), flush=True)

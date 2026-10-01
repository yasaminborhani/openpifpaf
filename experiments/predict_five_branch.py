"""Run the released five-branch PoseDriver checkpoint on ordinary image files.

The task-specific decoder settings and drawing thresholds match the audited
nuScenes Figure 7 inference. The five decoders run serially on fields from one
shared network forward per image.
"""

import argparse
from collections import Counter
import json
from pathlib import Path

from PIL import Image, ImageDraw
import torch

import openpifpaf
from openpifpaf.decoder import utils
from openpifpaf.decoder.multi import Multi


TASK_SETTINGS = {
    'cocokp': (0.2, 0.0),
    'animal': (0.01, 0.0),
    'apollo': (0.2, 0.0),
    'openlane': (0.2, 0.05),
    'cocobicyclekp': (0.2, 0.0),
}
TASK_NAMES = {
    'cocokp': 'human',
    'animal': 'animal',
    'apollo': 'car',
    'openlane': 'lane',
    'cocobicyclekp': 'bicycle',
}
COLORS = {
    'human': '#2979ff', 'animal': '#42d44e', 'car': '#c857ed',
    'lane': '#ff972f', 'bicycle': '#ff4848',
}


class TaggedMulti(Multi):
    """Apply the saved threshold pair to each sequential CIF/CAF decoder."""

    def __call__(self, fields):
        predictions = []
        old_seed = utils.CifSeeds.get_threshold()
        old_instance = utils.nms.Keypoints.get_instance_threshold()
        try:
            for task_decoder in self.decoders:
                dataset = task_decoder.cif_metas[0].dataset
                seed, instance = TASK_SETTINGS[dataset]
                utils.CifSeeds.set_threshold(seed)
                utils.nms.Keypoints.set_instance_threshold(instance)
                for annotation in task_decoder(fields):
                    annotation.pose_driver_dataset = dataset
                    predictions.append(annotation)
        finally:
            utils.CifSeeds.set_threshold(old_seed)
            utils.nms.Keypoints.set_instance_threshold(old_instance)
        return predictions


def visible_geometry(row, skeleton):
    if float(row['score']) < 0.2:
        return [], [], []
    flat = row['keypoints']
    points = [flat[i:i + 3] for i in range(0, len(flat), 3)]
    visible = {i for i, point in enumerate(points) if float(point[2]) >= 0.3}
    edges = [(int(a) - 1, int(b) - 1) for a, b in skeleton
             if int(a) - 1 in visible and int(b) - 1 in visible]
    adjacency = {i: set() for edge in edges for i in edge}
    for a, b in edges:
        adjacency[a].add(b)
        adjacency[b].add(a)
    retained = set()
    remaining = set(adjacency)
    minimum = 3 if row['task'] == 'bicycle' else 4
    while remaining:
        start = remaining.pop()
        component = {start}
        stack = [start]
        while stack:
            for neighbor in adjacency[stack.pop()] - component:
                component.add(neighbor)
                remaining.discard(neighbor)
                stack.append(neighbor)
        if len(component) >= minimum:
            retained.update(component)
    return points, [edge for edge in edges if set(edge) <= retained], retained


def draw_overlay(image, rows, skeletons):
    canvas = image.convert('RGB').copy()
    draw = ImageDraw.Draw(canvas)
    counts = Counter()
    width = max(2, round(canvas.width / 500))
    for row in rows:
        points, edges, retained = visible_geometry(row, skeletons[row['dataset']])
        if not edges:
            continue
        counts[row['task']] += 1
        color = COLORS[row['task']]
        for a, b in edges:
            draw.line([(float(points[a][0]), float(points[a][1])),
                       (float(points[b][0]), float(points[b][1]))],
                      fill=color, width=width)
        for index in retained:
            x, y = map(float, points[index][:2])
            draw.ellipse((x - width, y - width, x + width, y + width), fill=color)
    return canvas, dict(counts)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('images', nargs='+', type=Path)
    parser.add_argument('--output-dir', required=True, type=Path)
    openpifpaf.decoder.cli(parser, workers=0)
    openpifpaf.network.Factory.cli(parser)
    openpifpaf.Predictor.cli(parser)
    parser.set_defaults(
        long_edge=1600, loader_workers=0, decoder_workers=0, batch_size=1,
        force_complete_pose=True, head_consolidation='keep',
        cf4_attention_swin_head=True, swin_use_fpn=True,
        decoder=[f'cifcaf:{index}' for index in range(5)],
        seed_threshold=0.2, instance_threshold=0.05,
    )
    args = parser.parse_args()
    if not args.checkpoint or not Path(args.checkpoint).is_file():
        parser.error('--checkpoint must name a local file')
    if args.batch_size != 1 or args.decoder_workers != 0:
        parser.error('this release command requires batch size 1 and serial decoding')
    missing = [str(path) for path in args.images if not path.is_file()]
    if missing:
        parser.error('missing images: ' + ', '.join(missing))
    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    openpifpaf.decoder.configure(args)
    openpifpaf.network.Factory.configure(args)
    openpifpaf.Predictor.configure(args)

    predictor = openpifpaf.Predictor(checkpoint=args.checkpoint, visualize_image=False)
    decoders = predictor.processor.decoders
    datasets = {decoder.cif_metas[0].dataset for decoder in decoders}
    if datasets != set(TASK_SETTINGS) or len(decoders) != 5:
        raise RuntimeError('expected one decoder for each of the five PoseDriver tasks')
    predictor.processor = TaggedMulti(decoders)
    skeletons = {decoder.cif_metas[0].dataset: decoder.caf_metas[0].skeleton
                 for decoder in decoders}
    forward_count = [0]

    def count_forward(_module, _inputs, _output):
        forward_count[0] += 1

    handle = predictor.model.base_net.register_forward_hook(count_forward)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    records = []
    try:
        for index, (annotations, _, _) in enumerate(
                predictor.images([str(path) for path in args.images])):
            source = args.images[index]
            stem = f'{index + 1:04d}_{source.stem}'
            rows = []
            for annotation in annotations:
                row = annotation.json_data()
                row['dataset'] = annotation.pose_driver_dataset
                row['task'] = TASK_NAMES[annotation.pose_driver_dataset]
                rows.append(row)
            with Image.open(source) as image:
                overlay, counts = draw_overlay(image, rows, skeletons)
            overlay.save(args.output_dir / f'{stem}.png')
            (args.output_dir / f'{stem}.json').write_text(
                json.dumps({'source': source.name, 'predictions': rows}, indent=2) + '\n')
            records.append({'source': source.name, 'stem': stem, 'rendered_counts': counts})
            print(f'{index + 1}/{len(args.images)} {source.name}: {counts}', flush=True)
    finally:
        handle.remove()
    if forward_count[0] != len(args.images):
        raise RuntimeError(f'expected one encoder pass per image, got {forward_count[0]}')
    (args.output_dir / 'manifest.json').write_text(json.dumps({
        'checkpoint': Path(args.checkpoint).name,
        'images': records,
        'encoder_forward_calls': forward_count[0],
        'decoder_profiles': {name: {'seed_threshold': pair[0],
                                    'instance_threshold': pair[1]}
                             for name, pair in TASK_SETTINGS.items()},
        'long_edge': args.long_edge,
        'force_complete_pose': args.force_complete_pose,
        'rendering': {'instance_threshold': 0.2, 'keypoint_threshold': 0.3,
                      'minimum_connected_keypoints': 4,
                      'bicycle_minimum_connected_keypoints': 3},
    }, indent=2) + '\n')


if __name__ == '__main__':
    main()

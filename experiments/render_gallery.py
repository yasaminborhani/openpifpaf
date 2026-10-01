"""Render the released nuScenes prediction JSON using a local nuScenes download."""

import argparse
import json
from pathlib import Path

from PIL import Image

from predict_five_branch import draw_overlay


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--nuscenes-root', required=True, type=Path,
                        help='nuScenes root containing samples/CAM_FRONT')
    parser.add_argument('--output-dir', required=True, type=Path)
    parser.add_argument('--gallery-root', type=Path,
                        default=Path(__file__).resolve().parents[1] / 'gallery/nuscenes')
    args = parser.parse_args()
    manifest = json.loads((args.gallery_root / 'manifest.json').read_text())
    skeletons = json.loads((Path(__file__).parent / 'checkpoints/skeletons.json').read_text())
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for sample in manifest['samples']:
        record = json.loads((args.gallery_root / sample['prediction_file']).read_text())
        image_path = args.nuscenes_root / 'samples/CAM_FRONT' / sample['image_file']
        if not image_path.is_file():
            parser.error(f'missing nuScenes image: {image_path}')
        with Image.open(image_path) as image:
            overlay, counts = draw_overlay(image, record['predictions'], skeletons)
        output = args.output_dir / (Path(sample['prediction_file']).stem + '.png')
        overlay.save(output)
        print(f'{output.name}: {counts}', flush=True)


if __name__ == '__main__':
    main()

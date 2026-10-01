#!/usr/bin/env bash
set -euo pipefail

: "${POSEDRIVER_CHECKPOINT:?Set POSEDRIVER_CHECKPOINT}"
: "${POSEDRIVER_TEST_IMAGE:?Set POSEDRIVER_TEST_IMAGE}"
: "${POSEDRIVER_TEST_OUTPUT:?Set POSEDRIVER_TEST_OUTPUT}"

cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"

python3 -m pip install --no-build-isolation -e .
python3 -m pip install pytest
python3 - <<'PY'
import openpifpaf
from pathlib import Path
path = Path(openpifpaf.__file__).resolve()
assert Path.cwd() in path.parents, path
print('Installed source:', path)
PY
python3 -m openpifpaf.predict --help >/dev/null
python3 experiments/predict_five_branch.py --help >/dev/null
python3 -m pytest -o addopts='' -q tests/test_posedriver_bicycle_metric.py tests/test_posedriver_lane_dataset.py
python3 experiments/predict_five_branch.py \
  --checkpoint "$POSEDRIVER_CHECKPOINT" \
  --output-dir "$POSEDRIVER_TEST_OUTPUT" \
  "$POSEDRIVER_TEST_IMAGE"
python3 - <<'PY'
import json
import os
from pathlib import Path
folder = Path(os.environ['POSEDRIVER_TEST_OUTPUT'])
manifest = json.loads((folder / 'manifest.json').read_text())
assert manifest['encoder_forward_calls'] == 1
assert len(manifest['images']) == 1
assert len(list(folder.glob('*.png'))) == 1
assert len(list(folder.glob('*.json'))) == 2
print('One shared encoder forward and five-branch outputs verified')
PY

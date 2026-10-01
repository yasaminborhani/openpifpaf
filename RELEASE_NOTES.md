# PoseDriver research release

This release packages PoseDriver source code, selected training/evaluation commands, the Figure 7 five-branch inference path, a model checkpoint asset, and ten saved nuScenes prediction records.

The checkpoint is the restored Swin-L + FPN model assembled from the four-task epoch-450 model and the bicycle epoch-550 adaptation heads. It was not trained on nuScenes or jointly fine-tuned on five tasks. The repository includes the source checkpoint hashes and the numerical agreement report in `experiments/checkpoints/swinl_5branches_shared_backbone.provenance.json`.

The exact historical Table 9 bicycle AP scoring output and fold-1 evaluation JSON remain unavailable; the included evaluator regression tests use synthetic six-keypoint examples. The original Figure 6 checkpoint and historical CULane Table 4 matching settings are also unconfirmed. The paper's YOLO26 numbers are from published documentation, not a model in this release.

Dataset camera images are not bundled. The gallery prediction JSON and rendering command can be used with a local nuScenes download.

# PoseDriver experiment commands

The `commands/` directory contains six selected historical shell scripts: three four-task backbone runs and three bicycle runs. The Swin four-task and frozen-backbone bicycle commands identify the two sources of the released Figure 7 model. The scripts preserve training and evaluation flags from the saved records; they are not a claim that all paper tables can currently be regenerated from public data.

Set these variables before running a command **inside an allocated compute workload**:

```bash
export POSEDRIVER_WORKDIR=/path/to/writable/workspace
export POSEDRIVER_DATA_ROOT=/path/to/prepared/datasets
export POSEDRIVER_LANE_ANNOT_ROOT=/path/to/openlane_24_keypoint_annotations
bash experiments/commands/cocoanicarlane-swinl-fpn-l200-mixedbs64-lrx01-aug-att.sh
bash experiments/commands/bike-swinl-100epochs-fold1-att-fpn-bs32-transfer-freeze-aug-2.sh
```

The commands expect the original experiment data layout, including augmented COCO and AnimalPose annotations, ApolloScape 24-point annotations, converted OpenLane 24-point annotations, and the bicycle fold-1 split. Obtain the source datasets from their providers; create or supply those derived annotations in the paths named by the scripts. The bicycle annotation set is linked from the root README. The original `modified_fold_1_val.json` used in historical bicycle scoring has not been recovered, so the bicycle commands are provenance for the saved training settings rather than an independently verified Table 9 reproduction recipe.

The [five-branch assembly script](restore_five_branch_checkpoint.py) takes the epoch-450 four-task checkpoint and epoch-550 bicycle checkpoint, compares shared encoder/FPN tensors, and writes a combined checkpoint plus provenance JSON. The preassembled model in GitHub Releases is the one used for Figure 7. It is not a new five-task training run.

The [five-branch prediction command](predict_five_branch.py) takes ordinary image files. To recreate the ten saved Figure 7 overlays from a local nuScenes download, see [the gallery instructions](../gallery/nuscenes/README.md). The Python evaluator regression test is `tests/test_posedriver_bicycle_metric.py`.

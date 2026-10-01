# PoseDriver

PoseDriver is research code for detecting human, animal, car, bicycle, and lane skeletons with a shared visual encoder and task-specific CIF/CAF heads. This repository extends [OpenPifPaf-VITA](https://github.com/vita-epfl/openpifpaf); its original history, license, and third-party notices are retained. The paper is **“PoseDriver: A Unified Approach to Multi-Category Skeleton Detection for Autonomous Driving.”**

## What this release contains

- The model, dataset plugins, training and evaluation entry points, and selected experiment commands.
- A portable [five-branch inference command](experiments/predict_five_branch.py) and [checkpoint assembly script](experiments/restore_five_branch_checkpoint.py).
- The [Figure 7 prediction gallery](gallery/nuscenes/README.md), with ten nuScenes sample IDs and saved predictions. Raw nuScenes images are obtained from nuScenes itself.
- Provenance and SHA-256 for the released Swin-L + FPN five-branch checkpoint in [the checkpoint record](experiments/checkpoints/swinl_5branches_shared_backbone.provenance.json). Download the `swinl_5branches_shared_backbone.pt` asset from this repository's GitHub Releases page; model weights are intentionally outside Git.

The five-branch checkpoint was assembled from the epoch-450 four-task Swin-L + FPN model and the bicycle CIF/CAF heads learned in an epoch-550 frozen-backbone adaptation. The shared encoder/FPN tensors were checked for numerical equivalence before assembly. The checkpoint evaluates all five branches from **one encoder forward pass**; it was not jointly fine-tuned on five datasets and was not trained on nuScenes. See the provenance JSON for source checkpoint hashes and verification tolerances.

## Installation

The audited nuScenes run used Python 3.10.11, PyTorch 2.1.2+cu121, and one Tesla V100 GPU. A CUDA-capable Linux environment with a C++17 compiler is recommended because OpenPifPaf builds a C++ extension. Install a compatible PyTorch/torchvision pair first, then run:

```bash
python -m pip install --no-build-isolation -e '.[backbones,train]'
python -m pip install pytest
python -m openpifpaf.predict --help
```

The run used the EPFL image `registry.rcp.epfl.ch/vita/opp_all_2:latest` with digest `sha256:d0c983a5c1d2a447a2500648aafc1d603fb93691f3033894a22832769327d598`. This image identifier records the environment used for the audit; public users need not use that private registry.
That image contains an older system OpenPifPaf installation. When using it, run `export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"` from this repository after installation so Python imports the release checkout.

## Five-branch prediction

Download the released checkpoint and run:

```bash
python experiments/predict_five_branch.py \
  --checkpoint /path/to/swinl_5branches_shared_backbone.pt \
  --output-dir /path/to/output \
  /path/to/image1.jpg /path/to/image2.jpg
```

The command defaults to the audited 1600-pixel long edge, batch size 1, and sequential task decoders. It writes one prediction JSON and one overlay PNG per image, plus a manifest with thresholds and the counted shared-encoder forward passes. The drawing filter uses instance confidence 0.20, keypoint confidence 0.30, at least four connected points for human/animal/car/lane skeletons, and at least three for bicycles. These are visualization filters, not benchmark scoring thresholds. A PyTorch checkpoint containing Python objects should be loaded only from a trusted source; verify its SHA-256 against the checkpoint record.

For a single task, the existing `openpifpaf.predict` and `openpifpaf.eval` commands accept `--task`/`--dataset` to select the matching CIF/CAF head pair. See [the experiment notes](experiments/README.md) for training, evaluation, datasets, and limitations.

## Results and data boundaries

The saved nuScenes predictions are **qualitative**; nuScenes was not a PoseDriver training or joint-annotation benchmark. The bicycle annotations are [available separately on Kaggle](https://www.kaggle.com/datasets/javadkhorramdel/byccdt/data). The repo does not include COCO, AnimalPose, ApolloScape, OpenLane, CULane, or nuScenes images.

The corrected bicycle evaluator uses six OKS sigmas. However, the original successful scoring output and the historical fold-1 evaluation JSON for the paper's Table 9 are not available in this checkout. The released tests verify the evaluator on synthetic six-keypoint cases; they do not independently reproduce those historical AP values. The original Figure 6 checkpoint and some historical CULane matching details also remain unconfirmed.

The YOLO26 values in the paper are published comparison numbers, not results from a YOLO implementation in this repository. Experimental YOLO backbone code and weights are excluded from this release.

## License and attribution

The code retains the GNU AGPLv3-or-later license and the exceptions listed in [LICENSE](LICENSE). The `docs/` license files cover bundled third-party architecture code. Cite PoseDriver when using this work, and cite [OpenPifPaf](https://github.com/vita-epfl/openpifpaf) for the underlying framework. Dataset images and annotations remain subject to their providers' terms.

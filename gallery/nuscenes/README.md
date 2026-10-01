# PoseDriver Figure 7 prediction gallery

This directory contains prediction JSON for the ten nuScenes examples selected for Figure 7, plus `manifest.json` with scene names, sample tokens, and camera image filenames. The predictions came from the restored Swin-L + FPN five-branch checkpoint. NuScenes was used only for inference; it was not used for PoseDriver training or bicycle adaptation. Several predicted skeletons are partial, and these images are qualitative examples rather than an AP benchmark.

Obtain nuScenes from [the dataset provider](https://www.nuscenes.org/) and place it in a directory containing `samples/CAM_FRONT`. Render the overlays locally with:

```bash
python experiments/render_gallery.py \
  --nuscenes-root /path/to/nuscenes \
  --output-dir /path/to/posedriver-gallery
```

The renderer uses the exact saved task skeletons, the visualization confidence thresholds, and no black description bar. The JSON files contain model predictions and public dataset identifiers, not nuScenes camera images. The model asset is documented in the root README.

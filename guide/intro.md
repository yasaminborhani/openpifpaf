(introduction)=
# Introduction

Continuously tested on Linux, MacOS and Windows:
[![Tests](https://github.com/vita-epfl/openpifpaf/actions/workflows/tests.yml/badge.svg?branch=main)](https://github.com/vita-epfl/openpifpaf/actions?query=workflow%3ATests)
[![deploy-guide](https://github.com/vita-epfl/openpifpaf/actions/workflows/deploy-guide.yml/badge.svg)](https://github.com/vita-epfl/openpifpaf/actions?query=workflow%3Adeploy-guide)
[![Downloads](https://pepy.tech/badge/openpifpaf-vita)](https://pepy.tech/project/openpifpaf-vita)
<br />
[__New__ 2021 paper](https://arxiv.org/abs/2103.02440):

> __OpenPifPaf: Composite Fields for Semantic Keypoint Detection and Spatio-Temporal Association__<br />
> _[Sven Kreiss](https://www.svenkreiss.com), [Lorenzo Bertoni](https://scholar.google.com/citations?user=f-4YHeMAAAAJ&hl=en), [Alexandre Alahi](https://scholar.google.com/citations?user=UIhXQ64AAAAJ&hl=en)_, 2021.
>
> Many image-based perception tasks can be formulated as detecting, associating
> and tracking semantic keypoints, e.g., human body pose estimation and tracking.
> In this work, we present a general framework that jointly detects and forms
> spatio-temporal keypoint associations in a single stage, making this the first
> real-time pose detection and tracking algorithm. We present a generic neural
> network architecture that uses Composite Fields to detect and construct a
> spatio-temporal pose which is a single, connected graph whose nodes are the
> semantic keypoints (e.g., a person's body joints) in multiple frames. For the
> temporal associations, we introduce the Temporal Composite Association Field
> (TCAF) which requires an extended network architecture and training method
> beyond previous Composite Fields. Our experiments show competitive accuracy
> while being an order of magnitude faster on multiple publicly available datasets
> such as COCO, CrowdPose and the PoseTrack 2017 and 2018 datasets. We also show
> that our method generalizes to any class of semantic keypoints such as car and
> animal parts to provide a holistic perception framework that is well suited for
> urban mobility such as self-driving cars and delivery robots.

Previous [CVPR 2019 paper](http://openaccess.thecvf.com/content_CVPR_2019/html/Kreiss_PifPaf_Composite_Fields_for_Human_Pose_Estimation_CVPR_2019_paper.html).

Have fun with [our latest real-time interactive demo](https://vitademo.epfl.ch/movements/)!


## Demo

![example image with overlaid pose predictions](../docs/coco/000000081988.jpg.predictions.jpeg)

Image credit: "[Learning to surf](https://www.flickr.com/photos/fotologic/6038911779/in/photostream/)" by fotologic which is licensed under [CC-BY-2.0].<br />
Created with:
```sh
python3 -m openpifpaf.predict docs/coco/000000081988.jpg --image-output
```

![example image with overlaid wholebody pose predictions](https://raw.githubusercontent.com/vita-epfl/openpifpaf/main/docs/soccer.jpeg.predictions.jpeg)
Image credit: [Photo](https://de.wikipedia.org/wiki/Kamil_Vacek#/media/Datei:Kamil_Vacek_20200627.jpg) by [Lokomotive74](https://commons.wikimedia.org/wiki/User:Lokomotive74) which is licensed under [CC-BY-4.0](https://creativecommons.org/licenses/by/4.0/).<br />
Created with:
```sh
python3 -m openpifpaf.predict docs/wholebody/soccer.jpeg --checkpoint=shufflenetv2k30-wholebody --line-width=2 --image-output
```

More demos:
* [openpifpafwebdemo](https://github.com/openpifpaf/openpifpafwebdemo) project (best performance). [Live](https://vitademo.epfl.ch).
* OpenPifPaf [running in your browser: openpifpaf.github.io/openpifpafwebdemo](https://openpifpaf.github.io/openpifpafwebdemo/) (experimental)
* the {ref}`python3 -m openpifpaf.video <cli-help-video>` command (requires OpenCV)
* [Google Colab demo](https://colab.research.google.com/drive/1H8T4ZE6wc0A9xJE4oGnhgHpUpAH5HL7W)

```{image} ../docs/wave3.gif
:height: "250"
```


## Install

This version of OpenPifPaf (`openpifpaf-vita`) cannot co-exist with the original one ([`openpifpaf`](https://github.com/openpifpaf/openpifpaf)) in the same environment.
If you have previously installed the package `openpifpaf`, remove it before installation to avoid conflicts.

This project was forked from [OpenPifPaf v0.13.1](https://github.com/openpifpaf/openpifpaf/releases/tag/v0.13.1) and developed separately from version v0.14.0 on.

Do not clone this repository.
Make sure there is no folder named `openpifpaf-vita` in your current directory, and run:

```sh
pip3 install openpifpaf-vita
```

You need to install `matplotlib` to produce visual outputs:

```sh
pip3 install matplotlib
```

To modify OpenPifPaf itself, please follow {ref}`modify-code`.

For a live demo, we recommend to try the
[openpifpafwebdemo](https://github.com/openpifpaf/openpifpafwebdemo) project.
Alternatively, {ref}`python3 -m openpifpaf.video <cli-help-video>` (requires OpenCV)
provides a live demo as well.


## Pre-trained Models

Performance metrics on the COCO val set obtained with a GTX1080Ti:

| Name               | AP       | AP0.5    | AP0.75   | APM      | APL      | t_{total} [ms] | t_{NN} [ms] | t_{dec} [ms] |     size |
|-------------------:|:--------:|:--------:|:--------:|:--------:|:--------:|---------------:|------------:|-------------:|---------:|
| [mobilenetv3small] | __47.1__ | 73.9     | 49.5     | 40.1     | 58.0     | 26             | 9           | 14           |    5.8MB |
| [mobilenetv3large] | __58.4__ | 82.3     | 63.4     | 52.3     | 67.9     | 34             | 19          | 12           |   15.0MB |
| [resnet50]         | __68.1__ | 87.8     | 74.4     | 65.4     | 73.0     | 53             | 38          | 12           |   97.4MB |
| [shufflenetv2k16]  | __68.1__ | 87.6     | 74.5     | 63.0     | 76.0     | 40             | 28          | 10           |   38.9MB |
| [shufflenetv2k30]  | __71.8__ | 89.4     | 78.1     | 67.0     | 79.5     | 81             | 71          | 8            |  115.0MB |

[mobilenetv3large]: https://github.com/openpifpaf/torchhub/releases/download/v0.13/mobilenetv3large-210820-184901-cocokp-slurm725985-edge513-o10s-6c76cbfb.pkl
[mobilenetv3small]: https://github.com/openpifpaf/torchhub/releases/download/v0.13/mobilenetv3small-210822-213409-cocokp-slurm726252-edge513-o10s-803b24ae.pkl
[resnet50]: https://github.com/openpifpaf/torchhub/releases/download/v0.13/resnet50-210830-150728-cocokp-slurm728641-edge513-o10s-ecd30da4.pkl
[shufflenetv2k16]: https://github.com/openpifpaf/torchhub/releases/download/v0.13/shufflenetv2k16-210820-232500-cocokp-slurm726069-edge513-o10s-7189450a.pkl
[shufflenetv2k30]: https://github.com/openpifpaf/torchhub/releases/download/v0.13/shufflenetv2k30-210821-003923-cocokp-slurm726072-edge513-o10s-5fe1c400.pkl

Command to reproduce this table: {ref}`python -m openpifpaf.benchmark --checkpoints resnet50 shufflenetv2k16 shufflenetv2k30 <cli-help-benchmark>`.

Performance metrics on the COCO val set obtained with a NVIDIA A100:

| Name                    | AP       | AP0.5    | AP0.75   | APM      | APL      | t_{total} [ms] | t_{NN} [ms] | t_{dec} [ms] |     size |
|------------------------:|:--------:|:--------:|:--------:|:--------:|:--------:|---------------:|------------:|-------------:|---------:|
| [clipconvnextbase]      | __69.3__ | 87.8     | 75.5     | 63.2     | 78.4     | 110            | 86          | 15           |    688MB |
| [convnextv2base]        | __70.5__ | 87.9     | 76.9     | 64.2     | 79.7     | 138            | 113         | 17           |    338MB |
| [hrformerbasecat]       | __73.4__ | 89.9     | 80.0     | 69.1     | 80.2     | 312            | 239         | 62           |    331MB |
| [shufflenetv2k30*]      | __73.8__ | 90.1     | 80.2     | 69.7     | 80.4     | 112            | 73          | 30           |    115MB |
| [swin_l_input_upsample] | __75.8__ | 90.9     | 82.6     | 72.1     | 81.8     | 635            | 606         | 23           |    750MB |

[clipconvnextbase]: https://github.com/vita-epfl/openpifpaf-torchhub/releases/download/v0.14.2/cocokp-clipconvnextbase.pt.epoch250
[convnextv2base]: https://github.com/vita-epfl/openpifpaf-torchhub/releases/download/v0.14.2/cocokp-convnextv2base.pt.epoch250
[hrformerbasecat]: https://github.com/vita-epfl/openpifpaf-torchhub/releases/download/v0.14.2/cocokp-hrformerbasecat.pt.epoch250
[shufflenetv2k30*]: https://github.com/vita-epfl/openpifpaf-torchhub/releases/download/v0.14.2/cocokp-shufflenetv2k30.pt.epoch350
[swin_l_input_upsample]: https://github.com/vita-epfl/openpifpaf-torchhub/releases/download/v0.14.2/cocokp-swinl-inputupsample.pt.epoch250

Pretrained model files are shared in the __[vita-epfl/openpifpaf-torchhub](https://github.com/vita-epfl/openpifpaf-torchhub/releases)__ and __[openpifpaf/torchhub](https://github.com/openpifpaf/torchhub/releases)__
repositories and linked from the checkpoint names in the table above.
The pretrained models are downloaded automatically when
using the command line option `--checkpoint checkpointasintableabove`.


## Related Projects

* {doc}`Keypoint Communities <plugins_wholebody>`: plugin for wholebody human poses, [HuggingFace Demo](https://huggingface.co/spaces/akhaliq/Keypoint_Communities), [PapersWithCode](https://paperswithcode.com/paper/keypoint-communities).
* [neuralet](https://neuralet.com/article/pose-estimation-on-nvidia-jetson-platforms-using-openpifpaf/): TensorRT execution, including Docker images for NVidia Jetson.
* [fall detection using pose estimation](https://towardsdatascience.com/fall-detection-using-pose-estimation-a8f7fd77081d): illustrated with many example video clips.
* [physio pose](https://medium.com/@_samkitjain/physio-pose-a-virtual-physiotherapy-assistant-7d1c17db3159): "A virtual physiotherapy assistant".
* [monstereo](https://github.com/vita-epfl/monstereo): "MonStereo: When Monocular and Stereo Meet at the Tail of 3D Human Localization".
* [monoloco](https://github.com/vita-epfl/monoloco): "Monocular 3D Pedestrian Localization and Uncertainty Estimation".
* [openpifpafwebdemo](https://github.com/openpifpaf/openpifpafwebdemo): Web server and frontend. Docker image. Kubernetes config. [Live](https://vitademo.epfl.ch).
* [GitHub dependency graph](https://github.com/vita-epfl/openpifpaf/network/dependents): auto-detected Github repositories that use OpenPifPaf.

Open an [issue](https://github.com/vita-epfl/openpifpaf/issues) to suggest more projects.


## Executable Guide

This is a [jupyter-book](https://jupyterbook.org/intro.html) or "executable book".
Many sections of this book, like {doc}`predict_cli`, are generated from the code
shown on the page itself:
most pages are based on Jupyter Notebooks
that can be downloaded or launched interactively in the cloud by clicking on the rocket
at the top and selecting a cloud provider like _Binder_.
The code on the page is all the code required to reproduce that particular page.


(citation)=
## Citation

Reference {cite}`kreiss2021openpifpaf`,
[arxiv.org/abs/2103.02440](https://arxiv.org/abs/2103.02440)
```
@article{kreiss2021openpifpaf,
  title = {{OpenPifPaf: Composite Fields for Semantic Keypoint Detection and Spatio-Temporal Association}},
  author = {Sven Kreiss and Lorenzo Bertoni and Alexandre Alahi},
  journal = {IEEE Transactions on Intelligent Transportation Systems},
  pages = {1-14},
  month = {March},
  year = {2021}
}
```

Reference {cite}`kreiss2019pifpaf`,
[arxiv.org/abs/1903.06593](https://arxiv.org/abs/1903.06593)
```
@InProceedings{kreiss2019pifpaf,
  author = {Kreiss, Sven and Bertoni, Lorenzo and Alahi, Alexandre},
  title = {{PifPaf: Composite Fields for Human Pose Estimation}},
  booktitle = {The IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
  month = {June},
  year = {2019}
}
```

[CC-BY-2.0]: https://creativecommons.org/licenses/by/2.0/


## Commercial License

This software is available for licensing via the EPFL Technology Transfer
Office ([https://tto.epfl.ch/](https://tto.epfl.ch/), [info.tto@epfl.ch](mailto:info.tto@epfl.ch)).

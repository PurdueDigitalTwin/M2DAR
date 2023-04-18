## M<sup>2</sup>DAR: : Multi-View Multi-Scale Driver Action Recognition with Vision Transformer

Yunsheng Ma,
Liangqi Yuan,
Amr Abdelraouf,
Kyungtae Han,
Rohit Gupta,
Zihao Li,
Ziran Wang

![vis](.github/vis4.png)

## Introduction

Ensuring traffic safety and preventing accidents is a critical goal in daily driving. Computer vision technologies can
be leveraged to achieve this goal by detecting distracted driving behaviors. In this paper, we present M<sup>2</sup>DAR,
a multi-view, multi-scale framework for naturalistic driving action recognition and localization in untrimmed videos.

## Methodology

![framework](.github/framework.png)
M<sup>2</sup>DAR is a weight-sharing, Multi-Scale Transformer-based action recognition network that learns robust
hierarchical representations. It features a novel election algorithm consisting of aggregation, filtering, merging, and
selection processes to refine the preliminary results from the action recognition module across multiple views.

## Installation

This project is based on the [PySlowFast](https://github.com/facebookresearch/SlowFast) codebase (Many thanks!). Please
follow the instructions [there](https://github.com/facebookresearch/SlowFast/blob/main/INSTALL.md) to set up the
environment. Additionally, install the decord library by running the following command:

```bash
pip install decord
```

## Data preparation

1. Download the AI-City Challenge 2023 Track 3 Dataset and put it into `data/orignal`. The expected contents of this
   folder are:
    - `data/original/A1`
    - `data/original/A2`
    - `data/original/Distracted_Activity_Class_definition.txt`

2. Create an annotation file `annotation_A1.csv` by running the following command:

```bash
python tools/misc/create_anno.py
```

3. Move all videos in both the A1 and A2 subsets in a single folder `data/preprocessed/A1_A2_videos` using the following
   command:

```bash
python tools/misc/relocate_video.py `annotation_A1.csv`
```

4. Process the annotation file to fix errors in the original annotation file such as action end time < start time or
   wrong action labels by running the following command:

```bash
python tools/misc/process_anno.py
```

5. Cut the videos into clips using the following commands:

```bash
mkdir data/preprocessed/A1_clips
parallel -j 4 <data/A1_cut.sh
```

## Train

[TODO]

## Evaluation

[TODO]

## Citing M<sup>2</sup>DAR

If you use this code for your research, please cite our paper:

```
@inproceedings{ma_m2dar_2023,
	title = {{M2DAR}: {Multi}-{View} {Multi}-{Scale} {Driver} {Action} {Recognition} with {Vision} {Transformer}},
	booktitle = {2023 {IEEE}/{CVF} {Conference} on {Computer} {Vision} and {Pattern} {Recognition} {Workshops} ({CVPRW})},
	author = {Ma, Yunsheng and Yuan, Liangqi and Abdelraouf, Amr and Han, Kyungtae and Gupta, Rohit and Li, Zihao and Wang, Ziran},
	month = jun,
	year = {2023},
}
```

## Acknowledgement

This repo is based on [PySlowFast](https://github.com/facebookresearch/SlowFast)
and [AiCityAction](https://github.com/JunweiLiang/aicity_action) repos. Many thanks!
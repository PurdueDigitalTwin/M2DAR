## M<sup>2</sup>DAR: Multi-View Multi-Scale Driver Action Recognition with Vision Transformer

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

This project is based on [PySlowFast](https://github.com/facebookresearch/SlowFast)
and [AiCityAction](https://github.com/JunweiLiang/aicity_action) codebases. Please
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
python tools/data_converter/create_anno.py
```

3. Move all videos in both the A1 and A2 subsets in a single folder `data/preprocessed/A1_A2_videos` using the following
   command:

```bash
python tools/data_converter/relocate_video.py
```

4. Process the annotation file to fix errors in the original annotation file such as action end time < start time or
   wrong action labels by running the following command:

```bash
python tools/data_converter/process_anno.py
```

5. Cut the videos into clips using the following commands:

```bash
mkdir data/preprocessed/A1_clips
parallel -j 4 <data/A1_cut.sh
```

6. Split the `processed_annotation_A1.csv` into train and val subsets using the following commands:

```bash
python tools/data_converter/split_anno.py
```

7. Make annotation files for training on the whole A1 set:

```bash
$ mkdir data/annotations/splits/full
$ cat data/annotations/splits/splits_1/train.csv data/annotations/splits/splits_1/val.csv >data/annotations/splits/full/train.csv
$ cp data/annotations/splits/splits_1/val.csv data/annotations/splits/full/

```

8. Download the MViTv2-S pretrained model
   from [here](https://drive.google.com/file/d/1UwwCAS1fgS0dzxgiYxF_rITXwC_8Xx8r/view?usp=sharing) and put it
   as `MViTv2_S_16x4_pretrained.pyth` under `data/ckpts`.

## Training

To train the model, set the working directory to be the **project root path**, then run:

```bash
export PYTHONPATH=$PWD/:$PYTHONPATH
python tools/run_net.py --cfg configs/MVITV2_S_16x4_448.yaml

```

## Inference

1.Copy the `video_ids.csv` from the original dataset to the annotation directory:

```bash
cp data/original/A2/video_ids.csv data/annotations
mv data/annotations/video_ids.csv data/annotations/A2_video_ids.csv

```

2. Create the video list using the following:

```bash
python tools/data_converter/create_test_video_lst.py
```

3. Download the backbone model checkpoint
   from [here](https://drive.google.com/file/d/1Il1o9NMR6x8Cw4Q6B3AItZzvLFT_gtpE/view?usp=sharing), which is the one
   that achieved an overlap score of 0.5921 on the A2 test set and put it
   at `data/ckpts/MViTv2_S_16x4_aicity_200ep.pyth`.

4. Run the DAR module using the following commands, which will generate preliminary results stored at `data/runs/A2` in
   pickle format:

```bash
python tools/inference/driver_action_rec.py

```

5. Run the Election algorithm to refine the preliminary findings from the DAR module using the following commands:

```bash
python tools/inference/election.py
```

The submission file is at `data/inferences/submit.txt`, which achieves an overlap score of 0.5921 on the A2 test set.

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

## Acknowledgements

This repo is based on the [PySlowFast](https://github.com/facebookresearch/SlowFast)
and [AiCityAction](https://github.com/JunweiLiang/aicity_action) repos. Many thanks!

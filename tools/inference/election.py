import argparse
import os
import pickle
from typing import List

import numpy as np
import matplotlib.pyplot as plt
import copy
import matplotlib

from itertools import product
from collections import defaultdict
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("--pred_pickle_path",
                    default="data/runs/A2")
parser.add_argument("--vid_csv",
                    default="data/annotations/A2_video_ids.csv")
parser.add_argument("--threshold_path",
                    default="data/thresholds")
parser.add_argument("--output_path",
                    default="data/inferences")
parser.add_argument("--num_classes", default=16, type=int)
parser.add_argument("--visualize", default=True, type=bool)


class Election:
    def __init__(self, agg_weights: List[List[float]], fltr_thresholds: List[float], mrg_thresholds: List[int]):
        self.fps = 30.0
        self.agg_weights = agg_weights
        self.fltr_thresholds = fltr_thresholds
        self.mrg_thresholds = mrg_thresholds

    def aggregation(self, pr_matrix, action_id):
        weights = self.agg_weights[action_id]
        video_flen = min([r.shape[0] for r in pr_matrix])
        pr_matrix = [r[:video_flen] for r in pr_matrix]
        pr_matrix = np.vstack(pr_matrix)
        pr_matrix = weights[0] * pr_matrix[0, :] + weights[1] * pr_matrix[1, :] + weights[2] * pr_matrix[2, :]
        return pr_matrix

    def filtering(self, probs, action_id):
        video_flen = len(probs)
        th = self.fltr_thresholds[action_id]
        probs[probs < th] = 0.
        candidates = []
        start = -1
        for i in range(video_flen):
            pr = probs.item(i)
            if pr > 0.:
                if start == -1:
                    start = i
                elif start > -1 and i == video_flen - 1:
                    candidates.append((start, i))
                    break
            else:
                if start > -1:
                    candidates.append((start, i))
                    start = -1
        if len(candidates) == 0.:
            candidates.append((0, 0))
        return candidates

    def merging(self, candidates, action_id):
        if len(candidates) == 1:
            return candidates

        th = self.mrg_thresholds[action_id]
        merged = []
        pre_clip = candidates[0]
        for i in range(1, len(candidates)):
            cur_clip = candidates[i]
            gap = cur_clip[0] - pre_clip[1]
            if gap < th:
                pre_clip = [pre_clip[0], cur_clip[1]]
            else:
                merged.append(pre_clip)
                pre_clip = cur_clip
            if i == len(candidates) - 1:
                merged.append(pre_clip)
        return merged

    def selection(self, candidates, probs):
        candidates.sort(key=lambda c: np.sum(probs[c[0]: c[1] + 1]), reverse=True)
        start_sec, end_sec = round(candidates[0][0] / self.fps), round(candidates[0][1] / self.fps)
        assert end_sec >= start_sec
        return start_sec, end_sec

    def __call__(self, pr_matrix, action_id):
        probs = self.aggregation(pr_matrix, action_id)
        candidates = self.filtering(probs, action_id)
        candidates = self.merging(candidates, action_id)
        start_sec, end_sec = self.selection(candidates, probs)
        return start_sec, end_sec


def main(args):
    action_ids = list(range(args.num_classes))
    test_vids = {}
    all_videos = []
    for line in open(args.vid_csv, "r").readlines()[1:]:
        vid, file1, file2, file3 = line.strip().split(",")
        test_vids[vid] = [file1, file2, file3]
        all_videos += [file1, file2, file3]
    pickle_data = {}
    for file_id in all_videos:
        pred_file = os.path.join(args.pred_pickle_path, "%s.pkl" % file_id)
        with open(pred_file, "rb") as f:
            pred = pickle.load(f)
        pickle_data[file_id] = get_prob_matrix_tcm(pred, args.num_classes)

    if args.output_path is not None:
        os.makedirs(args.output_path, exist_ok=True)

    with open(os.path.join(args.threshold_path, 'aggregation.csv'), 'r') as f:
        agg_ws = [r.strip().split(',') for r in f.readlines()]
    agg_ths = list(map(lambda r: list(map(lambda w: float(w.strip()), r)), agg_ws))
    with open(os.path.join(args.threshold_path, 'filtering.csv'), 'r') as f:
        fltr_ths = [r.strip() for r in f.readlines()]
    fltr_ths = list(map(lambda p: float(p.strip()), fltr_ths))
    with open(os.path.join(args.threshold_path, 'merging.csv'), 'r') as f:
        mrg_ths = [r.strip() for r in f.readlines()]
    mrg_ths = list(map(lambda d: int(d.strip()), mrg_ths))
    election = Election(agg_ths, fltr_ths, mrg_ths)
    outputs = []
    for vid in test_vids:
        for action_id in action_ids:
            preds = []
            for idx, fid in enumerate(test_vids[vid]):
                preds.append(pickle_data[fid][:, action_id])
            ps, pe = election(preds, action_id)
            outputs.append((vid, action_id, ps, pe))
    outputs = [o for o in outputs if not o[2] == o[3] == 0]

    print("total actions %s" % len(outputs))
    with open(os.path.join(args.output_path, 'submit.txt'), "w") as f:
        for vid, action_id, start, end in outputs:
            f.writelines("%s %s %d %d\n" % (vid, action_id, start, end))


def get_prob_matrix_tcm(pred_list, num_classes):
    # frame_idx are 0-indexed
    frame_idxs = [t[0] for t in pred_list]
    frame_idxs += [t[1] for t in pred_list]
    min_frame_idx = min(frame_idxs)
    max_frame_idx = max(frame_idxs)
    frame_num = max_frame_idx - min_frame_idx

    # construct a list per frame_idx for all scores
    # assume scores are between 0. to 1.
    # len == num_frame, each is a list of predictions of all classes
    score_list_per_frame = [
        [np.zeros((num_classes), dtype="float32")]
        for i in range(frame_num)]

    # t1 - t0 == 64
    for t0, t1, cls_data in pred_list:
        for t in range(t0, t1):
            save_idx = t - min_frame_idx
            score = cls_data  # num_class
            assert len(score) == num_classes
            score = score / score.sum()
            score_list_per_frame[save_idx].append(score)

    # average the scores at each frame idx
    # get the chunks in (t0, t1) with scores >= thres
    avg_score_per_frame = []
    for i in range(len(score_list_per_frame)):
        # stack all the scores first
        if len(score_list_per_frame[i]) > 1:
            score_list_per_frame[i].pop(0)  # remove the zero padding
        # [T, num_class]
        stacked_scores = np.vstack(score_list_per_frame[i])
        # [num_class]
        this_frame_scores = np.mean(stacked_scores, axis=0)
        # this_frame_idx = min_frame_idx + i
        avg_score_per_frame.append(this_frame_scores)
    return np.vstack(avg_score_per_frame)  # [T, num_classes]


if __name__ == "__main__":
    main(parser.parse_args())

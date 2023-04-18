# coding=utf-8
"""
    Given the converted anno of aicity track3, get splits train.csv and val.csv

Rightside_user_id_24491_1.24491.0.17.mp4 -1
Rightside_user_id_24491_1.24491.18.45.mp4 3
Rightside_user_id_24491_1.24491.45.54.mp4 14
Rightside_user_id_24491_1.24491.74.105.mp4 2

    the annotation action_id should be zero-indexed
"""

import os
import argparse
from collections import defaultdict

parser = argparse.ArgumentParser()
parser.add_argument("--anno_file",
                    default="./data/annotations/processed_annotation_A1.csv")
parser.add_argument("--out_path",
                    default="./data/annotations/splits")
parser.add_argument("--method", type=int, choices=[1, 2], default=1,
                    help="1:, NA to 0, 2: NA and empty to 0")


def main(args):
    data = defaultdict(list)

    for line in open(args.anno_file).readlines():
        video_file, action_id = line.strip().split()
        user_id = video_file.split(".")[1]
        action_id = int(action_id)
        if action_id in [-1, -2]:
            if args.method == 1:
                if action_id == -2:
                    continue
                else:
                    action_id = 0
            else:
                action_id = 0

        assert action_id in range(16), action_id
        data[user_id].append((video_file, action_id))

    print("total user %s" % len(data))

    # each user as a validation set
    for i, user_id in enumerate(data.keys()):
        target_path = os.path.join(args.out_path, "splits_%s" % (i + 1))
        val_data = data[user_id]
        train_data = []
        for t_user_id in data:
            if t_user_id != user_id:
                train_data += data[t_user_id]
        print("train %s, val %s" % (len(train_data), len(val_data)))

        os.makedirs(target_path, exist_ok=True)
        train_file = os.path.join(target_path, "train.csv")
        val_file = os.path.join(target_path, "val.csv")
        with open(train_file, "w") as f:
            for one in train_data:
                f.writelines("%s %s\n" % (one[0], one[1]))
        with open(val_file, "w") as f:
            for one in val_data:
                f.writelines("%s %s\n" % (one[0], one[1]))


if __name__ == "__main__":
    main(parser.parse_args())

import argparse
import csv
import numpy as np
from collections import defaultdict
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--original_path",
                    default="./data/original/A1")
parser.add_argument("--out_anno_file",
                    default="./annotations/annotation_A1.csv")

fields = ["User ID", "FileName", "Camera View", "Activity Type", "Start Time", "End Time", "Label", "Appearance Block"]


def process_time(timestamp: str):
    timestamp = timestamp.replace('AM', '')
    timestamp = timestamp.replace('PM', '')
    timestamp = timestamp.strip()
    h, m, s = timestamp.split(':')
    return ':'.join([str(int(m)), str(int(s))])


def process_label(label: str):
    label = label.replace('Class', '').strip()
    return label


def main(args):
    original_path = Path(args.original_path)
    all_action_annotations = []
    for user_id_path in sorted(list(original_path.iterdir())):
        if not user_id_path.is_dir():
            continue
        user_id = str(user_id_path).split('/')[-1]
        uid5 = user_id[-5:]
        print(uid5)
        for csv_path in user_id_path.iterdir():
            csv_path = str(csv_path)
            if csv_path.endswith(".csv"):
                with open(csv_path, 'r') as f:
                    csvreader = csv.reader(f)
                    next(csvreader)
                    filename = ''
                    for row in csvreader:
                        if len(row[0]):
                            filename = row[0].capitalize().split('_')
                            filename.insert(-2, 'NoAudio')
                            filename = '_'.join(filename)

                        _, camera_view, activity_type, start_time, end_time, label, appearance_block = row
                        label = process_label(label)
                        if len(label) == 0:
                            continue

                        all_action_annotations.append(
                            [uid5, filename, camera_view, activity_type, process_time(start_time),
                             process_time(end_time), label, appearance_block])

    with open(args.out_anno_file, 'w') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(fields)
        csvwriter.writerows(all_action_annotations)


if __name__ == "__main__":
    main(parser.parse_args())

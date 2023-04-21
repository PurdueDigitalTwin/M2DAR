import argparse
import csv

parser = argparse.ArgumentParser()
parser.add_argument("--video_ids_file_path",
                    default="data/annotations/A2_video_ids.csv")
parser.add_argument("--out_file",
                    default="data/annotations/A2_videos.lst")


def main(args):
    all_a2_videos = []
    with open(args.video_ids_file_path, 'r') as f:
        csvreader = csv.reader(f)
        next(csvreader)
        for row in csvreader:
            all_a2_videos.extend(row[1:])
    all_a2_videos = [[p] for p in all_a2_videos]
    with open(args.out_file, 'w') as f:
        csvwriter = csv.writer(f)
        csvwriter.writerows(all_a2_videos)


if __name__ == "__main__":
    main(parser.parse_args())

import argparse
import subprocess
from pathlib import Path
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("--original_path",
                    default="data/original")
parser.add_argument("--dest_path",
                    default="data/preprocessed/A1_A2_videos")


def main(args):
    all_video_paths = []
    original_path = Path(args.original_path)
    for subset in ["A1", "A2"]:
        subset_path = original_path / subset
        print(subset_path)
        for user_id_path in subset_path.iterdir():
            if not user_id_path.is_dir():
                continue
            for video_path in user_id_path.iterdir():
                video_path = str(video_path)
                if video_path.endswith(".MP4"):
                    all_video_paths.append(video_path)
    assert len(all_video_paths) == (25 + 5) * 6
    for vid_path in tqdm(all_video_paths):
        subprocess.call([
            'cp',
            vid_path,
            args.dest_path
        ])


if __name__ == "__main__":
    main(parser.parse_args())

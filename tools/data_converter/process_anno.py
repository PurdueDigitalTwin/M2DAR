# coding=utf-8


import argparse
import os
import sys
import decord
import numpy as np
from collections import defaultdict
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("--anno_file",
                    default='./data/annotations/annotation_A1.csv')
parser.add_argument("--video_path",
                    default="./data/preprocessed/A1_A2_videos")
parser.add_argument("--out_anno_file",
                    default="./data/annotations/processed_annotation_A1.csv")
parser.add_argument("--clip_cmds",
                    default="./data/A1_cut.sh")  # ffmpeg bash file for cutting the videos into clips
parser.add_argument("--target_path",
                    default="./data/preprocessed/A1_clips")
parser.add_argument("--resolution", default="-2:540", help="ffmpeg -vf scale=")


def time2int(time_str):
    # 00:18 to integer seconds
    minutes, seconds = time_str.split(":")
    minutes = int(minutes)
    seconds = int(seconds)
    return minutes * 60 + seconds


def int2time(secs):
    # seconds to 00:00
    m, s = divmod(secs, 60)
    if s >= 10.0:
        return "%02d:%.3f" % (m, s)
    else:
        return "%02d:0%.3f" % (m, s)


def fix_user_id(fn_user_id, user_id, view):
    if fn_user_id == "30932" and user_id == "61962" and view == "Right_side_window":
        return user_id, user_id
    else:
        return fn_user_id, user_id


def process_file_name(file_name: str, user_id: str, view: str):
    if view == 'Rearview':
        view = 'Rear_view'
    elif view == 'Dashboard':
        pass
    elif view == 'Rightside Window':
        view = 'Right_side_window'
    else:
        raise NotImplementedError

    fn_noaudio, fn_user_id, perform_id = file_name.split('_')[-3:]
    fn_user_id, user_id = fix_user_id(fn_user_id, user_id, view)
    assert fn_user_id == user_id, f"fn: {fn_user_id}, uid: {user_id}, fname: {file_name}"
    return f"{view}_user_id_{fn_user_id}_{fn_noaudio}_{perform_id}"


def fix_label(video_file_name, start, end, label):
    # Remove annotation
    if '30932_NoAudio_7' in video_file_name and start == 357 and end == 361 and label == 14:
        return 0, 0, -1
    if '96269_NoAudio_5' in video_file_name and start == 436 and end == 460 and label == 2:
        return 0, 0, -1
    if '60167_NoAudio_7' in video_file_name and start == 501 and end == 494 and label == 0:
        return 0, 0, -1
    if '96269_NoAudio_7' in video_file_name and start == 515 and end == 515 and label == 0:
        return 0, 0, -1
    if '86356_NoAudio_5' in video_file_name and start == 153 and end == 177 and label == 15:
        return 0, 0, -1

    # Fix action label
    if '86356_NoAudio_5' in video_file_name and start == 127 and end == 134 and label == 4:
        return time2int("2:12"), time2int("2:30"), 0
    if '86952_NoAudio_11' in video_file_name:
        if end == 518:
            return time2int("8:34"), time2int("8:36"), 0
    return start, end, label


def fix_action_len(video_file_name, start, end, label):
    if '30932_NoAudio_7' in video_file_name and start == 268 and end == 267 and label == 9:
        return time2int("4:26"), time2int("4:28")

    if '59581_NoAudio_7' in video_file_name:
        if start == 351 and end == 334 and label == 11:
            return time2int("5:17"), time2int("5:34")
        if start == 537 and end == 496 and label == 5:
            return time2int("7:57"), end

    if '60167_NoAudio_7' in video_file_name:
        if start == 97 and end == 42 and label == 9:
            return start, time2int("1:42")
        if start == time2int("7:50") and end == time2int("8:10") and label == 0:
            return time2int("8:00"), end

    if '63513_NoAudio_5' in video_file_name and start == 19 and end == 11 and label == 4:
        return time2int("0:08"), time2int("0:12")

    if '86952_NoAudio_9' in video_file_name:
        if start == 47 and end == 0 and label == 3:
            return start, time2int("1:00")
        if start == 266 and end == 308 and label == 14:
            return time2int("4:46"), end

    if '47457_NoAudio_5' in video_file_name and start == 296 and end == 359 and label == 13:
        return start, time2int("4:59")

    if '83756_NoAudio_7' in video_file_name and start == 44 and end == 73 and label == 4:
        return start, time2int("1:11")

    if '96269_NoAudio_7' in video_file_name:
        if start == 48 and end == 109 and label == 10:
            return 46, 48
        if start == 158 and end == 237 and label == 6:
            return start, time2int("2:57")

    return start, end


def main(args):
    data = defaultdict(list)
    users = {}
    action_lengths = []
    action_id_to_count = defaultdict(int)
    vid_to_seg = defaultdict(dict)  # video_file to segment, make sure no overlap
    # compute some stats
    # 1. the action id num, the length stats
    for row in open(args.anno_file, "r").readlines()[1:]:
        user_id, video_file_name, view, _, start, end, action_id, block = row.strip().split(",")
        users[user_id] = 1
        # original video has "NoAudio" but annotation does not
        video_file_name = "%s.MP4" % process_file_name(video_file_name.strip(), user_id.strip(), view.strip())

        start = time2int(start)
        end = time2int(end)

        # action_id could be 0-15
        action_id = action_id.strip()
        action_id = int(action_id)
        action_id_to_count[action_id] += 1

        # assert no overlap
        start, end, action_id = fix_label(video_file_name, start, end, action_id)
        if action_id == -1:
            continue

        assert (start, end) not in vid_to_seg[video_file_name], f"{row}{video_file_name}, {start}, {end}, {action_id}"
        vid_to_seg[video_file_name][(start, end)] = 1

        start, end = fix_action_len(video_file_name, start, end, action_id)
        action_len = end - start
        assert action_len > 0, f"{row}{video_file_name}, {start}, {end}, {action_id}"
        action_lengths.append(action_len)

        data[video_file_name].append((user_id, video_file_name, start, end, action_id))

    print(action_id_to_count)
    print("user num: %s, action length min/max/median: %s, %s, %s" % (
        len(users),
        min(action_lengths), max(action_lengths), np.median(action_lengths)))

    # get the max length of each video, and check non-annotated segment length
    total_empty, total_length = 0, 0
    data_empty = {}  # video_file -> empty segments
    for video_file in tqdm(data):
        video = os.path.join(args.video_path, video_file)
        vcap = decord.VideoReader(video)
        num_frame = len(vcap)
        max_length = int(num_frame / 30.0)
        anno_max_length = data[video_file][-1][3]  # end
        user_id = data[video_file][0][0]

        anno_segments = [(None, None, 0, 0, 0)] + data[video_file]
        anno_segments.sort(key=lambda r: r[2])  # sort according to start time
        # print(anno_segments)

        if max_length > anno_max_length:
            print("%s anno ends on %s, has %s total" % (video_file, anno_max_length, max_length))
            anno_segments += [(None, None, max_length, 0, 0)]
        elif max_length < anno_max_length:
            print("warning for %s, %s, %s" % (video_file, anno_segments[-1], max_length))
            # some annotation might be longer than the video

        empty_segments = []
        for s1, s2 in zip(anno_segments[0:-1], anno_segments[1:]):
            last_end = s1[3]
            next_start = s2[2]

            gap = next_start - last_end
            if gap > 0:
                empty_segments.append((user_id, video_file, last_end, next_start, "empty"))
                total_empty += gap
            elif gap < 0:
                print(s1, s2)
                sys.exit()

        data_empty[video_file] = empty_segments
        total_length += max_length
    print("total length %s, empty %s" % (total_length, total_empty))

    # write the annotation file
    video_clips = []  # video_file_name.user_id.start.end.mp4
    with open(args.out_anno_file, "w") as f:

        for video_file in data:
            anno_segs = data[video_file]
            empty_segs = data_empty[video_file]
            for user_id, _, start, end, action_id in anno_segs + empty_segs:
                video_id = "%s.%s.%d.%d.MP4" % (
                    os.path.splitext(video_file)[0],
                    user_id, start, end)
                if action_id == "NA":
                    action_id = -1
                elif action_id == "empty":
                    action_id = -2
                action_id = int(action_id)
                video_clips.append((video_file, int2time(start), int2time(end), video_id))

                f.writelines("%s %d\n" % (video_id, action_id))

    # write the cutting command
    with open(args.clip_cmds, "w") as f:
        for ori_video, start, end, target_clip in video_clips:
            f.writelines("ffmpeg -nostdin -y -i %s -vf scale=%s -c:v libx264 -ss %s -to %s %s\n" % (
                os.path.join(args.video_path, ori_video),
                args.resolution,
                start, end,
                os.path.join(args.target_path, target_clip)))


if __name__ == "__main__":
    main(parser.parse_args())

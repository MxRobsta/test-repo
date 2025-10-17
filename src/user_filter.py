import csv
import hydra
import json
import numpy as np
from omegaconf import DictConfig
import os
from pathlib import Path
import soundfile as sf

from utils import load_refaudio, rms_norm

TARGET_FS = 16000
MIN_DURATION = TARGET_FS * 2
TARGET_SEGMENTS = 12
PADDING = 5 * TARGET_FS
TMP_PATH = "scratch/tmp.wav"


def user_filter(
    sessions_file: str,
    segments_file: str,
    session_ref_template: str,
    rainbow_template: str,
    filtered_store: str,
):

    if not Path(filtered_store).exists():
        filtered_segments = []
    else:
        with open(filtered_store, "r") as file:
            filtered_segments = json.load(file)

    seen_sesspids = [f"{x['session']}_{x['pid']}" for x in filtered_segments]

    RNG = np.random.default_rng()

    sessions_file = sessions_file.format(dataset="dev")
    with open(sessions_file, "r") as file:
        session_csv = list(csv.DictReader(file))

    while True:
        session_info = RNG.choice(session_csv)
        session = session_info["session"]
        wearer_positions = [
            int(session_info["ha_pos"]),
            int(session_info["aria_pos"]),
        ]
        pid = RNG.choice(
            [session_info[f"pos{i}"] for i in range(1, 5) if i not in wearer_positions]
        )
        device = RNG.choice(["ha", "aria"])

        if f"{session}_{pid}" not in seen_sesspids:
            break

    print(f"Filtering for:\n\nSession: {session}\nPID: {pid}\nDevice: {device}")
    os.system("play " + rainbow_template.format(dataset="dev", pid=pid))
    input("\nPress ENTER to begin")

    filtered_segments.append(
        {
            "session": session,
            "pid": pid,
            "device": device,
            "segments": [],
        }
    )

    segments_file = segments_file.format(
        dataset="dev", session=session, device=device, pid=pid
    )

    with open(segments_file, "r") as file:
        speech_segments = list(
            csv.DictReader(file, fieldnames=["index", "start", "end"])
        )

    RNG.shuffle(speech_segments)

    wearer_pos = int(session_info[f"{device}_pos"])
    ref_audio, fs = load_refaudio(
        session_ref_template,
        session,
        device,
        [f"pos{i}" for i in range(1, 5) if i != wearer_pos],
    )

    while len(filtered_segments[-1]["segments"]) < TARGET_SEGMENTS:

        this_seg = speech_segments.pop(0)
        index = int(this_seg["index"])
        start = int(this_seg["start"])
        end = int(this_seg["end"])

        if (end - start) < MIN_DURATION:
            continue

        print(f"Segments remaining: {len(speech_segments)}")
        print(
            f"Segments found: {len(filtered_segments[-1]['segments'])}/{TARGET_SEGMENTS}\n"
        )

        pad_start = start - PADDING
        audio_snippet = rms_norm(ref_audio[pad_start:end], 0.05)
        sf.write(TMP_PATH, audio_snippet, TARGET_FS)
        os.system("play " + TMP_PATH)
        x = input()

        if len(x) == 0:
            continue

        start_time = start / 16000
        end_time = end / 16000

        filtered_segments[-1]["segments"].append(
            {
                "key": f"{session}.{device}.{pid}.{index}",
                "index": index,
                "speech": {
                    "start_sample": start,
                    "end_sample": end,
                    "start_time": start_time,
                    "end_time": end_time,
                },
                "context": {"start_time": pad_start // 16000, "end_time": end_time},
            }
        )

        with open(filtered_store, "w") as file:
            json.dump(filtered_segments, file, indent=4)


@hydra.main(version_base=None, config_path="../config", config_name="main")
def main(cfg: DictConfig):
    user_filter(
        cfg.sessions_file,
        cfg.segments_file,
        cfg.ref_session_file,
        cfg.rainbow_file,
        cfg.filtered_store,
    )


if __name__ == "__main__":
    main()

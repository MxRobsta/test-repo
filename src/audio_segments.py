import csv
import hydra
import json
import numpy as np
from omegaconf import DictConfig
import os
from pathlib import Path
import soundfile as sf

from utils import load_audio, rms_norm, load_refaudio

TARGET_SR = 16000


def get_ref(ftemplate, sess_info, device):

    devpos = int(sess_info[f"{device}_pos"])
    others = [f"pos{i}" for i in range(1, 5) if i != devpos]

    assert len(others) == 3

    audio, fs = load_refaudio(
        ftemplate, sess_info["session"], device, others, normalize=0.05
    )

    return audio


@hydra.main(version_base=None, config_path="../config", config_name="main")
def main(cfg: DictConfig):
    segments_fpath = cfg.filtered_store
    animation_ftemplate = cfg.animations_file
    experiment = cfg.experiment
    sessions_file = cfg.sessions_file.format(dataset="dev")

    with open(sessions_file, "r") as file:
        sess_csv = {a["session"]: a for a in csv.DictReader(file)}

    if experiment == "passthrough" or experiment == "noisy":
        session_ftemplate = cfg.noisy_session_file
    elif experiment == "ref":
        session_ftemplate = cfg.ref_session_file
    elif experiment == "ct":
        session_ftemplate = cfg.ct_session_file
    else:
        session_ftemplate = cfg.exp_session_file

    output_audio_ftemplate = cfg.exp_segment_audio
    output_video_ftemplate = cfg.exp_segment_video

    with open(segments_fpath, "r") as file:
        sessions = json.load(file)

    for sess_info in sessions:
        session = sess_info["session"]
        device = sess_info["device"]
        pid = sess_info["pid"]

        if experiment in ["ref", "ct"]:
            session_audio = get_ref(session_ftemplate, sess_csv[session], device)
        else:
            infpath = session_ftemplate.format(
                dataset="dev", exp=experiment, session=session, device=device, pid=pid
            )
            session_audio, _ = load_audio(infpath, TARGET_SR, 0.05)

        if session_audio.ndim == 2:
            if device == "aria":
                session_audio = session_audio[:, 2]
            else:
                session_audio = np.sum(session_audio[:, :2], axis=1)

        for segment in sess_info["segments"]:

            seg_audio_fpath = Path(
                output_audio_ftemplate.format(
                    dataset="dev",
                    exp=experiment,
                    session=session,
                    device=device,
                    pid=pid,
                    seg=segment["index"],
                )
            )

            start = int(segment["start_time"] * TARGET_SR)
            end = int(segment["end_time"] * TARGET_SR)

            snippet = rms_norm(session_audio[start:end], 0.05)

            if not seg_audio_fpath.parent.exists():
                seg_audio_fpath.parent.mkdir(parents=True)

            sf.write(seg_audio_fpath, snippet, TARGET_SR)

            for anim in cfg.animation_types:
                seg_video_fpath = Path(
                    output_video_ftemplate.format(
                        dataset="dev",
                        exp=experiment,
                        session=session,
                        device=device,
                        pid=pid,
                        seg=segment["index"],
                        anim=anim,
                    )
                )
                anim_fpath = Path(
                    animation_ftemplate.format(
                        dataset="dev",
                        exp=experiment,
                        session=session,
                        device=device,
                        pid=pid,
                        seg=segment["index"],
                        anim=anim,
                    )
                )
                if not seg_video_fpath.exists() or cfg.overwrite:
                    os.system(
                        f"ffmpeg -y -hide_banner -loglevel error -i {anim_fpath} -i {seg_audio_fpath} -c:v copy -c:a aac {seg_video_fpath}"
                    )


if __name__ == "__main__":
    main()

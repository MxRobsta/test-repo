import hydra
import json
import numpy as np
from omegaconf import DictConfig
import os
from pathlib import Path
import soundfile as sf

from utils import load_audio, rms_norm

TARGET_SR = 16000


@hydra.main(version_base=None, config_path="../config", config_name="main")
def main(cfg: DictConfig):
    segments_fpath = cfg.filtered_store
    animation_ftemplate = cfg.animations_file
    experiment = cfg.experiment

    if experiment == "passthrough" or experiment == "noisy":
        session_ftemplate = cfg.noisy_session_file
    elif experiment == "ref":
        session_ftemplate = cfg.ref_session_file
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

        infpath = session_ftemplate.format(
            dataset="dev", exp=experiment, session=session, device=device, pid=pid
        )
        session_audio, fs = load_audio(infpath, TARGET_SR, 0.05)

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

            if seg_audio_fpath.exists():
                continue

            start = int(segment["context"]["start_time"] * TARGET_SR)
            end = int(segment["context"]["end_time"] * TARGET_SR)

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
                os.system(
                    f"ffmpeg -hide_banner -loglevel error -i {anim_fpath} -i {seg_audio_fpath} -c:v copy -c:a aac {seg_video_fpath}"
                )


if __name__ == "__main__":
    main()

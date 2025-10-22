import hydra
import json
import logging
import numpy as np
from omegaconf import DictConfig
from pathlib import Path
from tqdm import tqdm

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib import patches
from functools import partial

from utils import load_refaudio, rms_norm

PLOT_FS = 500

AUDIO_FS = 16000


def framewise_rms(snippet):
    hop = int(AUDIO_FS / PLOT_FS)
    window = 10 * hop
    ham = np.hamming(window)
    audio_len = snippet.shape[0]

    rms = []

    for i in range(0, audio_len - window, hop):
        frame = snippet[i : i + window] * ham
        rms.append(np.log(np.sqrt(np.mean(np.square(frame))) + 1e-5))
    rms = np.convolve(rms, np.hamming(100), mode="same")
    return rms


def animate_waveform(snippet, target_start, fpath):

    fig, ax = plt.subplots(figsize=[6, 2])
    (line1,) = ax.plot([], [], "tab:blue")

    dt = 0.01
    signal_seconds = snippet.shape[0] / PLOT_FS
    t = np.linspace(0, signal_seconds, len(snippet))

    # print(min(snippet), max(snippet))

    def init():
        ymin, ymax = min(snippet), max(snippet)
        ymax = ymax + (ymax - ymin) * 0.2
        text_top = ymax * 0.95 + ymin * 0.05

        ax.set_ylim(ymin, ymax)
        ax.set_xlim(0, signal_seconds)
        ax.axis("off")

        rect = patches.Rectangle(
            (target_start, ymin),
            signal_seconds - target_start,
            ymax - ymin,
            color="gray",
            alpha=0.15,
        )
        ax.add_patch(rect)
        ax.axvline(target_start, color="r")

        ax.plot(t, snippet, "tab:blue", alpha=0.2)

        mid = (target_start + signal_seconds) / 2

        ax.text(mid, text_top, "Transcribe here", ha="center", va="top")

        return (line1,)

    def update(frame, ln, x, y):
        start = int(frame * PLOT_FS)
        end = int((frame + dt) * PLOT_FS) + 1
        end = min(end, snippet.shape[-1])

        x += list(t[start:end])
        y += list(snippet[start:end])
        ln.set_data(x, y)
        return (ln,)

    ani = FuncAnimation(
        fig,
        partial(update, ln=line1, x=[], y=[]),
        frames=np.arange(0, signal_seconds, dt),
        init_func=init,
        blit=True,
        interval=dt * 1000,
        repeat=False,
    )

    if not Path(fpath).parent.exists():
        Path(fpath).parent.mkdir(parents=True)

    # plt.show()
    ani.save(fpath, writer="ffmpeg", fps=1 / dt)
    plt.close(fig)


@hydra.main(version_base=None, config_path="../config", config_name="main")
def main(cfg: DictConfig):
    segments_fpath = cfg.filtered_store
    audio_fpath = cfg.ref_session_file
    animation_ftemplate = cfg.animations_file

    with open(segments_fpath, "r") as file:
        sessions = json.load(file)

    logging.disable(logging.INFO)

    for sess_info in sessions:
        session = sess_info["session"]
        device = sess_info["device"]
        pid = sess_info["pid"]
        sum_waveform, fs = load_refaudio(
            audio_fpath,
            session,
            device,
            [f"pos{i}" for i in range(1, 5)],
            target_sr=PLOT_FS,
        )
        target_waveform, _ = load_refaudio(
            audio_fpath, session, device, pid, target_sr=PLOT_FS
        )

        target_power, _ = load_refaudio(
            audio_fpath, session, device, pid, target_sr=AUDIO_FS
        )

        for seg in tqdm(sess_info["segments"], desc=f"{session} {pid}"):
            start = int(seg["start_time"] * PLOT_FS)
            end = int(seg["end_time"] * PLOT_FS)
            target_start = seg["speech"]["start_time"] - seg["start_time"]

            snippet = rms_norm(sum_waveform[start:end], 0.05)
            fpath = animation_ftemplate.format(
                session=session,
                device=device,
                pid=pid,
                seg=seg["index"],
                anim="sumwave",
            )
            if not Path(fpath).exists() or cfg.overwrite:
                animate_waveform(snippet, target_start, fpath)

            snippet = rms_norm(target_waveform[start:end], 0.05)

            fpath = animation_ftemplate.format(
                session=session,
                device=device,
                pid=pid,
                seg=seg["index"],
                anim="targetwave",
            )
            if not Path(fpath).exists() or cfg.overwrite:
                animate_waveform(snippet, target_start, fpath)

            start = int(seg["start_time"] * AUDIO_FS)
            end = int(seg["end_time"] * AUDIO_FS)
            snippet = target_power[start:end]
            thing = framewise_rms(snippet)

            fpath = animation_ftemplate.format(
                session=session,
                device=device,
                pid=pid,
                seg=seg["index"],
                anim="targetpower",
            )

            if not Path(fpath).exists() or cfg.overwrite:
                animate_waveform(thing, target_start, fpath)


if __name__ == "__main__":
    main()

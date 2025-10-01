import hydra
import json
import matplotlib.pyplot as plt
from omegaconf import DictConfig
import pandas as pd
from pathlib import Path
from ppgs import PHONEMES
import soundfile as sf
import streamlit as st
import torch

SCORES = ["stoi", "fwsegsnr", "ppg_js"]


@st.cache_data
def read_json(fpath):
    with open(fpath, "r") as file:
        data = json.load(file)
    return data


@st.cache_data
def read_audio(fpath):
    audio, fs = sf.read(fpath)
    return audio, fs


@st.cache_data
def read_pt(fpath):
    return torch.load(fpath).to(torch.float).numpy().squeeze(0)


@st.cache_data
def get_df(segments):
    keep_keys = ["noisy", "ppg_js", "stoi", "fwsegsnr", "Csig", "Cbak", "Covl"]

    thing = []
    for seg in segments:
        output = {"noisy": Path(seg["noisy"]).stem}
        for a in keep_keys[1:]:
            if a == "fwsegsnr":
                output["fw"] = f"{seg[a]:.2f}"
            else:
                output[a] = f"{seg[a]:.2f}"
        thing.append(output)
    df = pd.DataFrame.from_dict(thing)
    df.reset_index(drop=True, inplace=True)
    return df


@hydra.main(version_base=None, config_path="../../config/vis", config_name="ppgs")
def main(cfg: DictConfig):

    st.set_page_config(layout="wide")

    segments = sorted(read_json(cfg.filtered_store), key=lambda x: x["noisy"])

    left_col, right_col = st.columns(2)

    left_col.dataframe(get_df(segments), height=len(segments) * 30)

    active_seg = right_col.selectbox(
        "Segment", segments, format_func=lambda x: Path(x["noisy"]).stem
    )

    noisy, nfs = read_audio(active_seg["noisy"])
    ref, rfs = read_audio(active_seg["ref"])

    right_col.audio(noisy.T, format="audio/wav", sample_rate=nfs)
    right_col.audio(ref.T, format="audio/wav", sample_rate=rfs)

    cols = right_col.columns(len(SCORES))
    for score, col in zip(SCORES, cols):
        col.write(score)
        col.write(f"{active_seg[score]:.2f}")

    noisy_ppg = read_pt(active_seg["noisy_ppg"])
    ref_ppg = read_pt(active_seg["ref_ppg"])

    fig, (ax0, ax1) = plt.subplots(2, 1, sharex=True)

    ax0.imshow(noisy_ppg)
    ax1.imshow(ref_ppg)
    fig.tight_layout()

    duration = active_seg["duration"]
    xticklabs = [noisy_ppg.shape[1] * x / duration for x in range(int(duration) + 1)]

    ax1.set_xticks(xticklabs, range(int(duration) + 1))
    ax1.set_xlabel("Time / s")

    aspect = 0.75 * noisy_ppg.shape[1] / noisy_ppg.shape[0]
    PHONEMES[-1] = "<>"

    for ax in [ax0, ax1]:
        ax.set_aspect(aspect)
        ax.set_yticks(range(len(PHONEMES)), PHONEMES, fontsize=8)
        for i, label in enumerate(ax.get_yticklabels()):
            label.set_x(-((i % 2)) * 0.05)

        for x in range(0, noisy_ppg.shape[1], 10):
            ax.axvline(x, linewidth=0.3, color="w", alpha=0.5)

        for y in range(len(PHONEMES)):
            ax.axhline(y + 0.5, linewidth=0.3, color="w", alpha=0.5)

    fig.tight_layout()
    right_col.pyplot(fig)


if __name__ == "__main__":
    main()

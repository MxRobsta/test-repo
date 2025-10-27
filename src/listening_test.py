import json
from hydra import initialize, compose
from hydra.core.global_hydra import GlobalHydra
import matplotlib.pyplot as plt
import os
import streamlit as st

from wer import plot_wer

if "current" not in st.session_state:
    st.session_state.current = {"state": "instructions"}

SEGMENTS = []
N_TRAINS = 3


@st.cache_resource
def load_config():
    if GlobalHydra.instance().is_initialized():
        GlobalHydra.instance().clear()
    with initialize(config_path="../config", version_base=None):
        cfg = compose(config_name="main")
    return cfg


@st.cache_data
def load_segment_json(fpath):
    with open(fpath, "r") as file:
        segments = json.load(file)

    return segments


def get_current():
    return st.session_state.current


def encode_current():
    return json.dumps(st.session_state.current)


def append_save(response):
    player_name = st.session_state.responses["name"]

    current = get_current()
    seg = SEGMENTS[current["speaker"]]["segments"][current["sample"]]
    st.session_state.responses["segments"].append(
        {
            "key": seg["key"],
            "ground_truth": seg["ground_truth"],
            "response": response[0],
            "commment": response[1],
            "isTrain": current["isTrain"],
        }
    )

    with open(f"transcripts/{player_name}.json", "w") as file:
        json.dump(st.session_state.responses, file, indent=4)


def check_continue():
    player_name = st.session_state.responses["name"]

    fpath = f"transcripts/{player_name}.json"

    if not os.path.exists(fpath):
        return {"state": "rainbow", "speaker": 0}

    with open(fpath, "r") as file:
        responses = json.load(file)
    keys = [x["key"] for x in responses["segments"]]
    st.session_state.responses = responses
    for spk, info in enumerate(SEGMENTS):
        for sample, seg in enumerate(info["segments"]):
            if seg["key"] not in keys:
                isTrain = sample < N_TRAINS
                return {
                    "state": "training" if isTrain else "testing",
                    "speaker": spk,
                    "sample": sample,
                    "isTrain": False,
                }
    print("All segments already seen")
    return {"state": "end"}


def instructions(rainbow_ftemplate, segment_ftemplate):
    tab_md = [
        "Welcome",
        "Volume",
        "Task",
        "Transcript",
        "Training",
        "Summary",
        "Important",
        "Begin",
    ]

    stabs = st.tabs(tab_md)
    for tmd, tab in zip(tab_md, stabs):
        tmd = tmd.lower()
        with open(f"src/markdown/{tmd}.md", "r") as file:
            text = file.read()
        tab.markdown(text)

        if tmd == "begin":
            name = tab.text_input("Name")
            tab.button(
                "Continue", on_click=continue_test, args=[name], disabled=len(name) == 0
            )
        elif tmd in ["volume", "task", "transcript", "training"]:
            with tab:
                if tmd == "training":
                    show_rainbow(rainbow_ftemplate, True)
                show_sample(segment_ftemplate, tmd)


def pretrain():
    current = st.session_state.current
    speaker = current["speaker"]
    pid = SEGMENTS[speaker]["pid"]

    st.header(f"TRAINING for {pid}")


def pretest():
    current = st.session_state.current
    speaker = current["speaker"]
    pid = SEGMENTS[speaker]["pid"]

    st.header(f"TESTING for {pid}")


def show_rainbow(rainbow_ftemplate, dummy=False):
    if dummy:
        speaker = 0
    else:
        speaker = get_current()["speaker"]

    pid = SEGMENTS[speaker]["pid"]
    st.header("Clean speech sample for the target")
    st.write(
        "Please listen to the target speaker's voice (recorded without background noise or interfering speakers)."
    )
    st.audio(rainbow_ftemplate.format(dataset="dev", pid=pid))


def show_sample(segment_ftemplate, dummy_stage=None):
    if dummy_stage is not None:
        speaker, index = 0, 0
    else:
        current = get_current()
        speaker = current["speaker"]
        index = current["sample"]

    info = SEGMENTS[speaker]
    segment = info["segments"][index]

    session = info["session"]
    device = info["device"]
    pid = info["pid"]

    fpath = segment_ftemplate.format(
        dataset="dev",
        exp="baseline",
        session=session,
        device=device,
        pid=pid,
        seg=segment["index"],
        anim=st.session_state.anim_type,
    )

    current = get_current()
    if dummy_stage is None and current["isTrain"]:
        st.header(f"Training Sample {current['sample'] + 1}/{N_TRAINS}")
    elif dummy_stage is None:
        st.header(
            f"Speaker Sample {current['sample'] + 1 - N_TRAINS}/{len(info['segments']) - N_TRAINS}"
        )

    if dummy_stage in ["task", "transcript", "training", None]:

        if dummy_stage in ["transcript", "training", None]:
            st.subheader("Prior Transcript")
            lines = ["Target: " + segment["target_prior"]]
            lines += segment["other_prior"].split("\n\n")
        else:
            lines = []

        cola, colb = st.columns([1, 5])
        for line in lines:
            pre_spk, text = line.split(":")
            cola.write(pre_spk)
            if len(text) <= 1:
                text = "."
            colb.write(text)

        cola.write("**Response**")

        if dummy_stage is None:
            key = speaker * 1000 + index
            default = ""
        else:
            key = dummy_stage
            default = segment["ground_truth"]

        response = colb.text_input(
            "blah", "", label_visibility="collapsed", key=key, placeholder=default
        )
    else:
        response = ""

    st.video(fpath, "video/mp4")

    # comment = st.text_input("Comments here")
    return response, ""


def continue_test(response):
    current = get_current()
    state = current["state"]

    if state == "instructions":
        st.session_state.current["state"] = "rainbow"
        st.session_state.current["speaker"] = 0
        st.session_state.responses = {"name": response, "segments": []}
    elif state == "rainbow":
        st.session_state.current = {
            "state": "training",
            "speaker": current["speaker"],
            "sample": 0,
            "isTrain": True,
        }
    elif state == "training":
        append_save(response)
        if current["sample"] == N_TRAINS - 1:
            st.session_state.current = {
                "state": "testing",
                "speaker": current["speaker"],
                "sample": N_TRAINS,
                "isTrain": False,
            }
        else:
            current["sample"] += 1
            st.session_state.current = current
    elif state == "testing":
        append_save(response)
        speaker = current["speaker"]
        sample = current["sample"]
        total_samples = len(SEGMENTS[speaker]["segments"])
        total_speakers = len(SEGMENTS)
        if sample < total_samples - 1:
            current["sample"] += 1
            st.session_state.current = current
        elif speaker < total_speakers - 1:
            # Finished speaker, move on to next
            st.session_state.current = {
                "state": "rainbow",
                "speaker": speaker + 1,
            }
        else:
            # Finished all speakers, ending test
            st.session_state.current = {"state": "end"}


def end_window(cfg):
    cfg.listener = st.session_state.responses["name"]
    fig = plot_wer(cfg, True)
    st.pyplot(fig)
    st.write(
        "Please download your data using the button below and send it to rwhsutherland1@sheffield.ac.uk"
    )
    plt.savefig(f"transcripts/{cfg.listener}.png")


def main():
    global SEGMENTS

    cfg = load_config()

    SEGMENTS = load_segment_json(cfg.filtered_store)

    if "anim_type" not in st.session_state:
        st.session_state.anim_type = "targetwave"
    if "transcript_type" not in st.session_state:
        st.session_state.transcript_type = "all"

    state = get_current()["state"]

    if state == "instructions":
        response = instructions(cfg.test_rainbow_file, cfg.exp_segment_video)
    elif state == "rainbow":
        response = show_rainbow(cfg.test_rainbow_file)
    elif state == "training" or state == "testing":
        response = show_sample(cfg.exp_segment_video)
    elif state == "end":
        response = end_window(cfg)
    else:
        raise ValueError(f"Invalid state: {get_current()}")

    if state == "end":
        name = st.session_state.responses["name"]
        data = json.dumps(st.session_state.responses)
        st.download_button(
            label="Download Your Data", data=data, file_name=f"{name}.json"
        )
    elif state != "instructions":
        x = st.checkbox("Confirm you want to continue", key=encode_current())
        st.button("Continue", on_click=continue_test, args=[response], disabled=not x)


if __name__ == "__main__":

    main()

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


def instructions():
    st.title("Instructions")
    st.write(
        "Welcome the the CHiME-9 ECHI Listening Test. Here you'll get some basic instructions."
    )
    st.header("The Task - Measuring Speech Intelligibility")
    st.write(
        "The task here is to recognise speech of a target speaker in a multi-party corversation in a noisy (cafeteria-like) condition. You will be played snippets of a conversation, and asked to type out the speech from one of the people in the conversation."
    )
    st.write(
        "To avoid ambiguity, all of the audio samples will be grouped by target speakers of whom the recognised words are to be written down, so that you're recognising the same speaker in each batch. You will first be given a clean sample for this person's voice, and then 3 samples to get used to their voice in the conversation for practise (which will not contribute to the final rating). Then, you will be given 9 samples to listen to."
    )
    st.write(
        "Each sample is 5-10s long, but you won't be expected to transcribe the entire sample. The first part of the sample will be the immediately previous part of the conversation, and then you will see a red line indicating when the target speech begins, and you should type what you hear from the red line to the end of the audio (you will see such a visualisation on the next page). You may also have the transcript of the previous part of the conversation, depending on which settings you choose."
    )
    st.write(
        "To make this as similar to real life as possible, you will only be allowed to listen to each audio sample **once** (except for the clean voice and warm-up samples). In real life, you would only get one chance to hear the speech, so we want to reflect that here. **COMMENT SG: SHOULD WE ALLOW LISTENING TO THE PART PREVIOUS TO THE RED LINE MULTIPLE TIMES?**"
    )
    st.header("Protocol")
    st.markdown(
        """
        1. You will get a clean sample of the target speaker
        2. You will get 3 practise examples, which you can replay as many times as you like to get used to the target voice and scenario
        3. You will be given 9 samples to listen to and write what you understood for a target speaker's voice
        4. Return to step one, but with a new speaker
        """
    )

    st.write(
        "For each sample, there is a comments section where you can provide feedback on any aspect of that sample (or anything else)."
    )

    st.write(
        "The next page will give you some settings to toggle, and show you an example of what a test will look like."
    )

    name = st.text_input("Please enter your name here")
    st.session_state.responses = {"name": name, "segments": []}


def settings(segment_ftemplate, anim_types, transcript_types):

    st.header("Settings")
    st.write(
        "Here you can toggle some settings for the listening tests. A preview is shown below, and will update when you toggle the settings."
    )
    st.write("**Please also use this page to tune your volume to a comfortable level after you listened to the audio example below.**")

    cola, colb = st.columns(2)
    with cola:
        anim = st.selectbox("anim_type", tuple(anim_types), index=0)
        st.session_state.anim_type = anim
    with colb:
        tra = st.selectbox("transcript type", tuple(transcript_types), index=2)
        st.session_state.transcript_type = tra

    st.subheader("Example Sample")

    show_sample(segment_ftemplate, True)


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


def show_rainbow(rainbow_ftemplate):
    speaker = get_current()["speaker"]

    pid = SEGMENTS[speaker]["pid"]
    st.header("Clean speech sample for " + pid)
    st.write(
        "Please listen to the target speaker's voice (recorded without background noise or interfering speakers)."
    )
    st.audio(rainbow_ftemplate.format(dataset="dev", pid=pid))


def show_sample(segment_ftemplate, dummy=False):
    if dummy:
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
    if not dummy and current["isTrain"]:
        st.header(f"Training Sample {current['sample'] + 1}/{N_TRAINS}")
    elif not dummy:
        st.header(
            f"Speaker Sample {current['sample'] + 1 - N_TRAINS}/{len(info['segments']) - N_TRAINS}"
        )

    if st.session_state.transcript_type != "none":
        st.subheader("Prior Transcript")
        lines = ["Target: " + segment["target_prior"]]

        if st.session_state.transcript_type == "all":
            lines += segment["other_prior"].split("\n\n")

        cola, colb = st.columns([1, 5])
        for line in lines:
            pre_spk, text = line.split(":")
            cola.write(pre_spk)
            if len(text) == 1:
                text = "."
            colb.write(text)

    cola, colb = st.columns([1, 5])
    with cola:
        st.write("**Response**")

    response = colb.text_input(
        "blah", "", label_visibility="collapsed", key=speaker * 1000 + index
    )

    st.video(fpath, "video/mp4")

    comment = st.text_input("Comments here")
    return response, comment


def continue_test(response):
    current = get_current()
    state = current["state"]

    if state == "instructions":
        st.session_state.current["state"] = "settings"
    elif state == "settings":
        st.session_state.current = check_continue()
    elif state == "rainbow":
        st.session_state.current = {"state": "pretrain", "speaker": current["speaker"]}
    elif state == "pretrain":
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
                "state": "pretest",
                "speaker": current["speaker"],
                "sample": N_TRAINS,
                "isTrain": False,
            }
        else:
            current["sample"] += 1
            st.session_state.current = current
    elif state == "pretest":
        current["state"] = "testing"
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
        st.session_state.anim_type = cfg.animation_types[1]
    if "transcript_type" not in st.session_state:
        st.session_state.transcript_type = cfg.transcript_types[2]

    state = get_current()["state"]

    if state == "instructions":
        response = instructions()
    elif state == "settings":
        response = settings(
            cfg.exp_segment_video, cfg.animation_types, cfg.transcript_types
        )
    elif state == "rainbow":
        response = show_rainbow(cfg.test_rainbow_file)
    elif state == "pretrain":
        response = pretrain()
    elif state == "pretest":
        response = pretest()
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
    else:
        st.button("Continue", on_click=continue_test, args=[response])


if __name__ == "__main__":

    main()

import hydra
import json
from omegaconf import DictConfig
import streamlit as st

if "current" not in st.session_state:
    st.session_state.current = "instructions"

SEGMENTS = []


@st.cache_data
def load_segment_json(fpath):
    with open(fpath, "r") as file:
        segments = json.load(file)

    return segments


def get_current():
    return st.session_state.current


def decode_current():
    current = get_current()
    return current // 1000, current % 1000


def encode_current(speaker, index):
    return speaker * 1000 + index


def instructions():
    st.title("Instructions")
    st.write(
        "Welcome the the CHiME-9 ECHI Listening Test. Here you'll get some basic instructions."
    )
    st.header("The Task")
    st.write(
        "The task here is to recognise speech in noisy conversations. You will be played snippets of a conversation, and asked to type out the speech from one of the people in the conversation."
    )
    st.write(
        "To avoid ambiguity, all of the audio samples will be grouped so that you're recognising the same speaker in each batch. You will first be given a clean sample for this person's voice, and then 2 warm-up samples to get used to their voice in the conversation. Then, you will be given ~10 samples to listen to."
    )
    st.write(
        "Each sample is 10-15s long, but you won't be expected to transcribe the entire sample. The first part of the sample will be the immediately previous part of the conversation, and then you will see a red line indicating when the target speech begins, and you should type what you hear from the red line to the end of the audio. You will also have the transcript of the previous part of the conversation."
    )
    st.write(
        "To make this as similar to real life as possible, you will only be allowed to listen to each audio sample **once** (except for the vlean voice and warm-up samples). In real life, you would only get one chance to hear the speech, so we want to reflect that here."
    )
    st.header("Protocol")
    st.markdown(
        """
        1. You will get a clean sample of the target speaker
        2. You will get 2 warm-up examples, which you can replay as many times as you like to get used to the target voice and scenario
        3. You will be given ~10 samples to work through for this target voice
        4. Return to step one, but with a new speaker
        """
    )


def show_rainbow(rainbow_ftemplate):
    speaker = int(get_current().split()[1])

    pid = SEGMENTS[speaker]["pid"]
    st.header("Clean speech sample for " + pid)
    st.audio(rainbow_ftemplate.format(dataset="dev", pid=pid))


def show_sample(segment_ftemplate):
    speaker, index = decode_current()

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
    )

    cola, colb = st.columns(2)

    with cola:
        st.write("**Prior Transcript**")
        print(segment)
        if "prespeech" in segment:
            st.write(segment["prespeech"])

    with colb:
        st.write("**Response**")
        response = st.text_input(
            "blah", value="", label_visibility="collapsed", key=get_current()
        )

    st.video(fpath, "video/mp4")

    return response


def continue_test(response):
    state = get_current()
    print("response", response)

    if isinstance(state, str):
        if state == "instructions":
            st.session_state.current = "spk 0"
        elif state[:3] == "spk":
            speaker = int(state.split()[1])
            st.session_state.current = encode_current(speaker, 0)
        else:
            raise ValueError(f"Invalid state: {state}")
    elif isinstance(state, int):
        speaker, index = decode_current()

        current_segments = SEGMENTS[speaker]["segments"]
        if index == len(current_segments) - 1 and speaker == len(SEGMENTS) - 1:
            # Completed last speaker, finish study
            st.session_state.current = "end"
        elif index == len(current_segments) - 1:
            # Finished test for speaker, move on to the next one
            st.session_state.current = f"spk {speaker + 1}"
        else:
            # Move on to the next samples for the current speaker
            st.session_state.current = encode_current(speaker, index + 1)
    else:
        raise ValueError(f"Invalid state: {state}")


@hydra.main(version_base=None, config_path="../config", config_name="main")
def main(cfg: DictConfig):
    global SEGMENTS

    SEGMENTS = load_segment_json(cfg.filtered_store)

    current = get_current()

    if current == "instructions":
        response = instructions()
    elif isinstance(current, str) and current[:3] == "spk":
        response = show_rainbow(cfg.rainbow_file)
    elif isinstance(current, int):
        response = show_sample(cfg.exp_segment_video)
    elif current == "end":
        response = st.title("Thank you")
    else:
        raise ValueError(f"Invalid state: {current}")

    st.button("Continue", on_click=continue_test, args=[response])


if __name__ == "__main__":

    main()

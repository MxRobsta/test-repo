import hydra
import json
from omegaconf import DictConfig
import os


@hydra.main(version_base=None, config_path="../config", config_name="main")
def main(cfg: DictConfig):

    filtered_store = cfg.filtered_store
    listener = cfg.listener
    audio_type = "noisy"
    if listener is None:
        listener = "ground_truth"
        audio_type = "ref"

    with open(filtered_store, "r") as file:
        segments = json.load(file)

    for seg in segments:

        if len(seg["transcripts"]["ground_truth"].split()) > 15:
            continue

        if "transcripts" in seg and listener in seg["transcripts"]:
            continue

        if audio_type == "ref":
            x = ""
            while len(x) == 0:
                os.system("play --norm=-3 " + seg[audio_type])
                x = input("Transcription:   ")
        else:
            os.system("play --norm=-3 " + seg[audio_type])
            x = input("Transcription:   ")

        if x == "EXIT":
            break

        if "transcripts" not in seg:
            seg["transcripts"] = {}
        seg["transcripts"][listener] = x

    with open(filtered_store, "w") as file:
        json.dump(segments, file, indent=4)


if __name__ == "__main__":
    main()

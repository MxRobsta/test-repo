import hydra
import json
import numpy as np
from omegaconf import DictConfig
from pathlib import Path

from utils import load_csv, load_json, get_wearer_targets

GRANULARITY = 10


def get_vad_array(transcripts):
    max_len = max(max(x["end_time"] for x in y) for y in transcripts.values())
    max_sample = int(GRANULARITY * max_len)

    vad = {}
    for pid, transcript in transcripts.items():
        vad[pid] = np.zeros(max_sample) - 1

        for i, seg in enumerate(transcript):
            start = int(seg["start_time"] * GRANULARITY)
            end = int(seg["end_time"] * GRANULARITY)
            vad[pid][start:end] = i
    return vad


def extract_hits(transcripts, hits, target):

    testable = []
    for i, h in enumerate(hits):
        target_index = h["target"]
        priors = h["priors"]

        item = {
            "target_pid": target,
        }

        target_segment = transcripts[target][target_index]
        target_segment["pid"] = target
        item["target_segment"] = target_segment

        item["prior_segments"] = []
        for pid, ids in priors.items():
            for i in ids:
                seg = transcripts[pid][i]
                seg["pid"] = pid
                item["prior_segments"].append(seg)

        testable.append(item)

    return testable


def count_words(text: str):
    remove_chars = [".", ",", ";", ":", "?"]
    text = text.lower()

    for c in remove_chars:
        if c in text:
            text = text.replace(c, "")

    poss_words = text.split(" ")

    clean = []
    for w in poss_words:
        if len(w) > 0 and w[0] != "[" and w[-1] != "]":
            clean.append(w)
    count = len(clean)
    clean = " ".join(clean)
    return count, clean


def filter_n_words(transcript, min_words, max_words):
    suitable = []
    for seg in transcript:
        n_words, clean = count_words(seg["text"])
        if n_words >= min_words and n_words <= max_words:
            seg["clean"] = clean
            suitable.append(seg)

    return suitable


def filter_prewindow(targ_ts, target, vad, partner, wearers, prewindow):

    if isinstance(wearers, str):
        wearers = [wearers]

    wearer_vad = np.concatenate([vad[w] for w in wearers])

    hits = []

    for seg in targ_ts:
        index = seg["index"]
        targ_start = int(seg["start_time"] * GRANULARITY)
        pre_start = int(targ_start - prewindow * GRANULARITY)
        end = int(seg["end_time"] * GRANULARITY)

        if any(wearer_vad[pre_start:end] >= 0):
            # Ignore if device wearer is talking
            continue

        priors = {}
        for pid in [target, partner]:
            segments = list(set(vad[pid][pre_start:targ_start]))
            segments = [int(s) for s in segments if s >= 0]
            if len(segments) > 0:
                priors[pid] = segments

        if len(priors) == 0:
            continue

        hits.append({"target": index, "priors": priors})

    return hits


@hydra.main(version_base=None, config_path="../config", config_name="filter")
def main(cfg: DictConfig):

    devices = ["aria", "ha"]

    # load sessions file
    session_fpath = cfg.sessions_file.format(dataset=cfg.dataset)
    session_info = load_csv(session_fpath)

    min_words = cfg.min_words
    max_words = cfg.max_words
    prewindow = cfg.prewindow

    for sinfo in session_info:

        transcripts = {}
        pids = []
        for i in range(1, 5):
            pid = sinfo[f"pos{i}"]
            pids.append(pid)
            fpath = cfg.transcript_file.format(
                dataset=cfg.dataset, session=sinfo["session"], pid=f"pos{i}"
            )
            transcripts[pid] = load_json(fpath)

        wearers, targets = get_wearer_targets(sinfo)

        vad = get_vad_array(transcripts)

        for device in devices:
            for targ in targets:
                partner = targets[int(targ == targets[0])]
                dev_ts = transcripts[targ]
                dev_ts = filter_n_words(dev_ts, min_words, max_words)
                hits = filter_prewindow(
                    dev_ts, targ, vad, partner, list(wearers.values()), prewindow
                )

                segments = extract_hits(transcripts, hits, targ)

                out_fpath = Path(
                    cfg.testable_file.format(
                        dataset=cfg.dataset,
                        session=sinfo["session"],
                        device=device,
                        pid=targ,
                    )
                )
                out_fpath.parent.mkdir(parents=True, exist_ok=True)

                with open(out_fpath, "w") as file:
                    json.dump(segments, file, indent=4)

                print(
                    f"{sinfo['session']}.{device}.{targ}",
                    len(segments),
                    len(transcripts[targ]),
                )


if __name__ == "__main__":
    main()

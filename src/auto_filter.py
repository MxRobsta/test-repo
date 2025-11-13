import csv
import hydra
import json
from omegaconf import DictConfig
from pathlib import Path


def read_ts(fpath):
    with open(fpath, "r") as file:
        data = json.load(file)
    return data


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


def filter_prewindow(targ_ts, transcripts, wearers, prewindow):

    bad_segments = []
    for w in wearers:
        bad_segments += transcripts[w]
    rem = []

    for seg in targ_ts:
        start = seg["start_time"] - prewindow
        end = seg["end_time"]
        bad = False
        for bs in bad_segments:
            if bs["end_time"] > start and bs["start_time"] < start:
                bad = True
                break
            elif bs["start_time"] < end and bs["end_time"] > end:
                bad = True
                break

        if not bad:
            rem.append(seg)

    return rem


@hydra.main(version_base=None, config_path="../config", config_name="filter")
def main(cfg: DictConfig):

    devices = ["aria", "ha"]

    # load sessions file
    session_fpath = cfg.sessions_file.format(dataset=cfg.dataset)
    with open(session_fpath, "r") as file:
        session_info = list(csv.DictReader(file))

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
            transcripts[pid] = read_ts(fpath)

        wearers = {d: sinfo[f"pos{sinfo[f'{d}_pos']}"] for d in devices}
        targets = [p for p in pids if p not in wearers.values()]

        for device in ["aria", "ha"]:
            for targ in targets:
                dev_ts = transcripts[targ]
                dev_ts = filter_n_words(dev_ts, min_words, max_words)
                dev_ts = filter_prewindow(
                    dev_ts, transcripts, [wearers[device]], prewindow
                )

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
                    json.dump(dev_ts, file, indent=4)

                print(
                    f"{sinfo['session']}.{device}.{targ}",
                    len(dev_ts),
                    len(transcripts[targ]),
                )


if __name__ == "__main__":
    main()

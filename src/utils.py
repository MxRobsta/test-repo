import csv
import json
import numpy as np
from pathlib import Path
import soundfile as sf
import soxr
from typing import Tuple


def load_refaudio(
    ftemplate, session, device, pids, dataset="dev", target_sr=16000, normalize=None
) -> Tuple[np.ndarray, int]:
    if isinstance(pids, str):
        fpath = ftemplate.format(
            dataset=dataset, session=session, device=device, pid=pids
        )
        return load_audio(fpath, target_sr, normalize)

    for i, pid in enumerate(pids):
        fpath = ftemplate.format(
            dataset=dataset, session=session, device=device, pid=pid
        )

        audio, fs = load_audio(fpath, target_sr, normalize)

        if i == 0:
            output = audio
        else:
            output += audio

    return output, fs


def load_audio(fpath, target_sr, normalize=None) -> Tuple[np.ndarray, int]:
    audio, fs = sf.read(fpath)

    if fs != target_sr:
        audio = soxr.resample(audio, fs, target_sr)

    if normalize:
        audio = rms_norm(audio, normalize)

    return audio, target_sr


def rms_norm(audio, target_rms):
    audio_rms = np.sqrt(np.mean(np.square(audio))) + 1e-5

    audio *= target_rms / audio_rms

    return audio


def load_csv(fpath: str | Path):
    with open(fpath) as file:
        data = list(csv.DictReader(file))
    return data


def load_json(fpath: str | Path):
    with open(fpath) as file:
        data = json.load(file)
    return data


def save_json(fpath: str | Path, data):
    parent = Path(fpath).parent
    parent.mkdir(exist_ok=True, parents=True)
    with open(fpath, "w") as file:
        json.dump(data, file, indent=4)


def get_wearer_targets(session_info):
    devices = ["aria", "ha"]
    pids = [session_info[f"pos{i}"] for i in range(1, 5)]

    wearers = {d: session_info[f"pos{session_info[f'{d}_pos']}"] for d in devices}
    targets = [p for p in pids if p not in wearers.values()]

    return wearers, targets

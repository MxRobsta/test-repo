import numpy as np
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
    audio_rms = np.sqrt(np.mean(np.square(audio)))

    audio *= target_rms / audio_rms

    return audio

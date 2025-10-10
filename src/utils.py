import numpy as np
import soundfile as sf
import soxr


def load_refaudio(ftemplate, session, device, pids, dataset="dev", target_sr=16000):
    if isinstance(pids, str):
        fpath = ftemplate.format(
            dataset=dataset, session=session, device=device, pid=pids
        )
        audio, fs = sf.read(fpath)
        if fs != target_sr:
            audio = soxr.resample(audio, fs, target_sr)
            fs = target_sr
        return audio, fs

    for i, pid in enumerate(pids):
        fpath = ftemplate.format(
            dataset=dataset, session=session, device=device, pid=pid
        )

        audio, fs = sf.read(fpath)
        if fs != target_sr:
            audio = soxr.resample(audio, fs, target_sr)
            fs = target_sr

        if i == 0:
            output = audio
        else:
            output += audio

    return output, fs


def rms_norm(audio, target_rms):
    audio_rms = np.sqrt(np.mean(np.square(audio)))

    audio *= target_rms / audio_rms

    return audio

import csv
import hydra
import numpy as np
from omegaconf import DictConfig
from pathlib import Path
import soundfile as sf
import soxr
from tqdm import tqdm

SAMPLE_RATE = 16000


def segment_files(
    noisy_audio: np.ndarray,
    ref_audio: np.ndarray,
    segments: list[dict],
    noisy_seg_ftemplate: str,
    ref_seg_ftempalte: str,
):

    length = min(noisy_audio.shape[0], ref_audio.shape[0])
    for seg in segments:
        start = int(seg["start"])
        end = int(seg["end"])

        if end > length:
            break

        noisy_snip = noisy_audio[start:end]
        noisy_file = noisy_seg_ftemplate.format(seg=seg["index"], start=start, end=end)
        if not Path(noisy_file).parent.exists():
            Path(noisy_file).parent.mkdir(parents=True)
        sf.write(noisy_file, noisy_snip, SAMPLE_RATE)

        ref_snip = ref_audio[start:end]
        ref_file = ref_seg_ftempalte.format(seg=seg["index"], start=start, end=end)
        if not Path(ref_file).parent.exists():
            Path(ref_file).parent.mkdir(parents=True)
        sf.write(ref_file, ref_snip, SAMPLE_RATE)


def segment_session(
    session_info: dict[str],
    device: str,
    noisy_session_ftemplate: str,
    ref_session_ftemplate: str,
    segments_ftemplate: str,
    noisy_seg_ftemplate: str,
    ref_seg_ftemplate: str,
):
    dataset = session_info["session"].split("_")[0]

    noisy_session_fpath = noisy_session_ftemplate.format(
        dataset=dataset, device=device, session=session_info["session"]
    )
    noisy_audio, fs = sf.read(noisy_session_fpath)
    if fs != SAMPLE_RATE:
        noisy_audio = soxr.resample(noisy_audio, fs, SAMPLE_RATE)

    for i in range(1, 5):
        if str(i) == session_info[f"{device}_pos"]:
            continue
        pid = session_info[f"pos{i}"]

        ref_session_fpath = ref_session_ftemplate.format(
            dataset=dataset, device=device, session=session_info["session"], pid=pid
        )
        ref_audio, fs = sf.read(ref_session_fpath)

        segments_fpath = segments_ftemplate.format(
            dataset=dataset, device=device, session=session_info["session"], pid=pid
        )
        with open(segments_fpath, "r") as file:
            segments = list(csv.DictReader(file, fieldnames=["index", "start", "end"]))

        noisy_seg_ftemplate2 = noisy_seg_ftemplate.format(
            dataset=dataset,
            device=device,
            session=session_info["session"],
            pid=pid,
            seg="{seg}",
            start="{start}",
            end="{end}",
        )
        ref_seg_ftemplate2 = ref_seg_ftemplate.format(
            dataset=dataset,
            device=device,
            session=session_info["session"],
            pid=pid,
            seg="{seg}",
            start="{start}",
            end="{end}",
        )

        segment_files(
            noisy_audio, ref_audio, segments, noisy_seg_ftemplate2, ref_seg_ftemplate2
        )


def segment_all(
    sessions_file: str,
    dataset: str,
    device: str,
    noisy_session_ftemplate: str,
    ref_session_ftemplate: str,
    segments_ftemplate: str,
    noisy_seg_ftemplate: str,
    ref_seg_ftemplate: str,
):

    with open(sessions_file.format(dataset=dataset), "r") as file:
        sessions = list(csv.DictReader(file))

    for session_info in tqdm(sessions):
        segment_session(
            session_info,
            device,
            noisy_session_ftemplate,
            ref_session_ftemplate,
            segments_ftemplate,
            noisy_seg_ftemplate,
            ref_seg_ftemplate,
        )


@hydra.main(version_base=None, config_path="../config", config_name="segment_audio")
def main(cfg: DictConfig):

    if isinstance(cfg.devices, str):
        cfg.devices = [cfg.devices]

    for device in cfg.devices:
        segment_all(
            cfg.sessions_file,
            cfg.dataset,
            device,
            cfg.noisy_session_file,
            cfg.ref_session_file,
            cfg.segments_file,
            cfg.noisy_seg_file,
            cfg.ref_seg_file,
        )


if __name__ == "__main__":
    main()

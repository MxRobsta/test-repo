import hydra
import json
from omegaconf import DictConfig
from pathlib import Path
import soundfile as sf
import soxr
from tqdm import tqdm


@hydra.main(version_base=None, config_path="../config", config_name="main")
def main(cfg: DictConfig):
    filtered_store = cfg.filtered_store
    filtered_file = cfg.filtered_file
    exp_session_ftemplate = cfg.exp_session_file

    with open(filtered_store, "r") as file:
        segments = json.load(file)

    for seg in tqdm(segments):
        exp_fpath = exp_session_ftemplate.format(
            exp=cfg.experiment,
            session=seg["session"],
            device=seg["device"],
            pid=seg["target"],
        )

        audio, fs = sf.read(exp_fpath)
        if fs != 16000:
            audio = soxr.resample(audio, fs, 16000)

        start = seg["start"] - seg["start_pad"]
        end = seg["end"] + seg["end_pad"]

        snipper = audio[start:end]
        output_fpath = filtered_file.format(
            system=cfg.experiment,
            device=seg["device"],
            session=seg["session"],
            pid=seg["target"],
            seg=seg["segment_index"],
            start=seg["start"],
            end=seg["end"],
        )
        if not Path(output_fpath).parent.exists():
            Path(output_fpath).parent.mkdir(parents=True)
        sf.write(output_fpath, snipper, 16000)

        seg[cfg.experiment] = output_fpath

    with open(filtered_store, "w") as file:
        json.dump(segments, file, indent=4)


if __name__ == "__main__":
    main()

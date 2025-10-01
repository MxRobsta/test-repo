import hydra
import json
import numpy as np
from omegaconf import DictConfig
from pathlib import Path
import soundfile as sf
import soxr
from tqdm import tqdm


@hydra.main(version_base=None, config_path="../config", config_name="main")
def main(cfg: DictConfig):

    RNG = np.random.default_rng(666)

    filtered_store = cfg.filtered_store
    noisy_session_ftemplate = cfg.noisy_session_file
    ref_session_ftemplate = cfg.ref_session_file

    seg_ftemplate = cfg.filtered_file

    stag_min = cfg.stagger_min * 16000
    stag_max = cfg.stagger_max * 16000

    with open(filtered_store, "r") as file:
        segments = json.load(file)

    for seg in tqdm(segments):
        session = seg["session"]
        device = seg["device"]
        target = seg["target"]
        split = session.split("_")[0]

        noisy_file = noisy_session_ftemplate.format(
            device=device, dataset=split, session=session
        )
        ref_file = ref_session_ftemplate.format(
            device=device, dataset=split, session=session, pid=target
        )

        noisy, nfs = sf.read(noisy_file)
        ref, rfs = sf.read(ref_file)

        if device == "aria":
            noisy = noisy[:, 2]
        else:
            noisy = noisy[:, 0] + noisy[:, 1]

        if nfs != rfs:
            noisy_file = soxr.resample(noisy, nfs, rfs)

        start = seg["start"]
        end = seg["end"]

        start_pad = RNG.integers(stag_min, stag_max)
        end_pad = RNG.integers(stag_min, stag_max)

        snip_start = max(start - start_pad, 0)
        snip_end = min(end + end_pad, noisy.shape[0])

        noisy_snippet = noisy[snip_start:snip_end]
        ref_snippet = ref[snip_start:snip_end]

        old_nfile = Path(seg["noisy"]).stem
        seg_index = old_nfile.split(".")[3]

        nfile = seg_ftemplate.format(
            system="noisy",
            session=session,
            device=device,
            pid=target,
            seg=seg_index,
            start=start,
            end=end,
        )
        sf.write(nfile, noisy_snippet, rfs)

        rfile = seg_ftemplate.format(
            system="noisy",
            session=session,
            device=device,
            pid=target,
            seg=seg_index,
            start=start,
            end=end,
        )
        sf.write(rfile, ref_snippet, rfs)

        seg["noisy"] = nfile
        seg["ref"] = rfile
        seg["segment_index"] = seg_index

        seg["start_pad"] = int(start_pad)
        seg["end_pad"] = int(end_pad)

        if "duration" in seg:
            del seg["duration"]

        seg["speech_duration"] = float((end - start) / rfs)
        seg["file_duration"] = float((snip_end - snip_start) / rfs)

    with open(filtered_store, "w") as file:
        json.dump(segments, file, indent=4)


if __name__ == "__main__":
    main()

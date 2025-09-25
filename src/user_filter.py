from glob import glob
import hydra
import json
import numpy as np
from omegaconf import DictConfig
from pathlib import Path
import os
import shutil


MIN_DURATION = 16000 * 2


def filter_segments(
    noisy_seg_file: str,
    ref_seg_file: str,
    filtered_file: str,
    filtered_store: str,
):
    noisy_match = noisy_seg_file.format(
        device="*", dataset="*", session="*", pid="*", seg="*", start="*", end="*"
    )

    noisy_files = glob(noisy_match)
    np.random.shuffle(noisy_files)

    if Path(filtered_store).exists():
        with open(filtered_store, "r") as file:
            selected = json.load(file)
    else:
        selected = []

    for noisy in noisy_files:
        if any(noisy == x["noisy"] for x in selected):
            continue

        fname = Path(noisy).stem
        session, device, pid, seg, start_end = fname.split(".")

        start, end = [int(a) for a in start_end.split("_")]

        if end - start < MIN_DURATION:
            continue

        split = session.split("_")[0]
        ref = ref_seg_file.format(
            device=device,
            dataset=split,
            session=session,
            pid=pid,
            seg=seg,
            start=start,
            end=end,
        )

        if not os.path.exists(ref):
            continue

        os.system("play " + noisy + " gain -n")
        os.system("play " + ref + " gain -n")
        x = input()

        if len(x) > 0:
            duration = (end - start) / 16000
            selected.append(
                {
                    "noisy": noisy,
                    "ref": ref,
                    "duration": duration,
                    "session": session,
                    "device": device,
                    "target": pid,
                    "start": start,
                    "end": end,
                }
            )
            print(f"You have currently selected {len(selected)} files.")
            with open(filtered_store, "w") as file:
                json.dump(selected, file, indent=4)

            filtered_fpath = filtered_file.format(
                device=device,
                dataset=split,
                session=session,
                pid=pid,
                seg=seg,
                start=start,
                end=end,
            )

            if not Path(filtered_fpath).parent.exists():
                Path(filtered_fpath).parent.mkdir(parents=True)

            shutil.copyfile(noisy, filtered_fpath)
            filtered_fpath = filtered_file.format(
                device=device + "_ref",
                dataset=split,
                session=session,
                pid=pid,
                seg=seg,
                start=start,
                end=end,
            )
            shutil.copyfile(ref, filtered_fpath)


@hydra.main(version_base=None, config_path="../config", config_name="filter")
def main(cfg: DictConfig):
    filter_segments(
        cfg.noisy_seg_file, cfg.ref_seg_file, cfg.filtered_file, cfg.filtered_store
    )


if __name__ == "__main__":
    main()

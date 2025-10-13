import hydra
import json
from omegaconf import DictConfig
from pathlib import Path


@hydra.main(version_base=None, config_path="../config", config_name="main")
def main(cfg: DictConfig):
    json_file = cfg.filtered_store
    tsv_dir = Path(cfg.tsv_dir)

    with open(json_file, "r") as file:
        sess_list = json.load(file)

    for sess in sess_list:
        output_path = tsv_dir / f"{sess['session']}.{sess['device']}.{sess['pid']}.tsv"

        tsv_segs = []
        for seg in sess["segments"]:
            start = seg["speech"]["start_time"]
            end = min(seg["speech"]["end_time"], start + 2)

            tsv_segs.append("\t".join([str(start), str(end), "segment"]) + "\n")

        with open(output_path, "w") as file:
            file.writelines(tsv_segs)


if __name__ == "__main__":
    main()
